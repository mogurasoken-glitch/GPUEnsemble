using System;
using System.Collections.Generic;
using System.Web.Script.Serialization;

namespace GPUEnsembleApp
{
    enum RunState { Stopped, Starting, Downloading, Ready, Failed }

    /// <summary>
    /// One long-running engine command (up, pull or worker): its state, its log and its progress.
    /// Events arrive on worker threads; the window marshals them to the UI thread.
    /// </summary>
    abstract class Service
    {
        ChildProcess? child;
        readonly List<string> log = new List<string>();
        public event Action? Changed;
        /// <summary>Only the download progress moved: the window updates its bar and text, nothing else.</summary>
        public event Action? Progressed;
        public event Action<string>? LogLine;

        public RunState State { get; protected set; } = RunState.Stopped;
        public double Done { get; private set; }
        public double Total { get; private set; }
        public int MinutesLeft { get; private set; }
        public string LastError { get; private set; } = "";
        public bool Running => child != null && !child.HasExited;

        public string[] Log { get { lock (log) return log.ToArray(); } }

        protected abstract IEnumerable<string> Args();
        protected virtual void OnLine(string line) { }

        public void Start()
        {
            if (Running) return;
            Done = Total = 0; MinutesLeft = 0; LastError = "";
            var p = new ChildProcess(Engine.StartInfo(Args()));
            p.Output += Line;
            p.Error += Line;
            p.Exited += code =>
            {
                if (!ReferenceEquals(child, p)) return;
                State = code == 0 || State == RunState.Stopped ? RunState.Stopped : RunState.Failed;
                child = null;
                Changed?.Invoke();
            };
            child = p;
            State = RunState.Starting;
            Add("> gpuensemble " + string.Join(" ", Args()));
            try { p.Start(); }
            catch (Exception e)
            {
                child = null;
                State = RunState.Failed;
                LastError = e.Message;
                Add("error: " + e.Message);
            }
            Changed?.Invoke();
        }

        public void Stop()
        {
            var p = child;
            if (p == null) return;
            State = RunState.Stopped;
            p.Stop();
            Changed?.Invoke();
        }

        protected void Send(string line) => child?.Send(line);
        protected void Raise() => Changed?.Invoke();

        void Add(string line)
        {
            lock (log)
            {
                log.Add(line);
                if (log.Count > 5000) log.RemoveRange(0, 1000);
            }
            LogLine?.Invoke(line);
        }

        void Line(string line)
        {
            Add(line);
            if (Lines.Progress(line, out var done, out var total, out var min))
            {
                Done = done; Total = total; MinutesLeft = min;
                if (State == RunState.Starting)
                {
                    State = RunState.Downloading;
                    Changed?.Invoke();
                }
                else
                    Progressed?.Invoke();
            }
            else if (line.StartsWith("error: ", StringComparison.Ordinal))
            {
                LastError = line.Substring(7);
                Changed?.Invoke();
            }
            OnLine(line);
        }
    }

    /// <summary><c>up --yes --remote auto [--model ID]</c>: the model server, with the GPUs other PCs lend.</summary>
    sealed class ServerService : Service
    {
        public string? ModelId { get; set; }
        public string? RunningModel { get; private set; }
        public string ChatUrl { get; private set; } = "http://127.0.0.1:8080/";

        protected override IEnumerable<string> Args()
        {
            yield return "up";
            yield return "--yes";
            yield return "--remote";
            yield return "auto";
            if (!string.IsNullOrEmpty(ModelId)) { yield return "--model"; yield return ModelId!; }
        }

        protected override void OnLine(string line)
        {
            var name = Lines.Starting(line);
            if (name != null) { RunningModel = name; State = RunState.Starting; Raise(); return; }
            var url = Lines.Ready(line);
            if (url != null) { ChatUrl = url; State = RunState.Ready; Raise(); }
        }
    }

    /// <summary><c>pull ID</c>: downloads one model.</summary>
    sealed class PullService : Service
    {
        public PullService(string id) { Id = id; }
        public string Id { get; }
        protected override IEnumerable<string> Args() { yield return "pull"; yield return Id; }
    }

    /// <summary><c>worker --lan --cache --serve</c>: lends this PC's GPUs; asks the user about each new main PC.</summary>
    sealed class WorkerService : Service
    {
        readonly List<KeyValuePair<string, string>> asking = new List<KeyValuePair<string, string>>();

        /// <summary>Main PCs that asked to use these GPUs and were neither let in nor refused: (name, address).</summary>
        public KeyValuePair<string, string>[] Asking { get { lock (asking) return asking.ToArray(); } }

        protected override IEnumerable<string> Args()
        {
            yield return "worker";
            yield return "--lan";
            yield return "--cache";
            yield return "--serve";
        }

        protected override void OnLine(string line)
        {
            if (State == RunState.Starting && line.StartsWith("Ctrl+C stops it", StringComparison.Ordinal)) { State = RunState.Ready; Raise(); }
            if (Lines.Asks(line, out var name, out var address))
            {
                lock (asking)
                {
                    asking.RemoveAll(a => a.Value == address);
                    asking.Add(new KeyValuePair<string, string>(name, address));
                }
                if (State == RunState.Starting) State = RunState.Ready;
                Raise();
            }
        }

        /// <summary>Lets that PC in (the worker remembers it), as typing <c>allow ADDRESS</c> in its window does.</summary>
        public void Allow(string address)
        {
            Send("allow " + address);
            Forget(address);
        }

        public void Forget(string address)
        {
            lock (asking) asking.RemoveAll(a => a.Value == address);
            Raise();
        }
    }

    /// <summary>
    /// <c>monitor --json --every 2</c> while a model runs: the real speed and each card's memory, heat,
    /// clocks and load (see the planner's monitor.py, the contract between the two). One process, one line
    /// per reading; it ends with the model or the app.
    /// </summary>
    sealed class MonitorService
    {
        ChildProcess? child;
        public Dictionary<string, object>? Last { get; private set; }
        public event Action? Updated;
        public bool Running => child != null && !child.HasExited;

        public void Start()
        {
            if (Running || Engine.StateFile != null) return;
            Last = null;
            var p = new ChildProcess(Engine.StartInfo(new[] { "monitor", "--json", "--every", "2" }));
            p.Output += line =>
            {
                try
                {
                    if (new JavaScriptSerializer().DeserializeObject(line) is Dictionary<string, object> d && ReferenceEquals(child, p))
                    {
                        Last = d;
                        Updated?.Invoke();
                    }
                }
                catch (Exception) { }  // a line that is not a reading
            };
            p.Error += _ => { };
            child = p;
            try { p.Start(); p.StandardInput.Close(); }
            catch (Exception) { child = null; }
        }

        public void Stop()
        {
            var p = child;
            child = null;
            Last = null;
            p?.Stop();
        }

        /// <summary>The reading of a card of this PC, by its nvidia-smi number.</summary>
        public Dictionary<string, object>? Gpu(int index)
        {
            if (Last == null || !Last.TryGetValue("gpus", out var v) || !(v is object[] a)) return null;
            foreach (var o in a)
                if (o is Dictionary<string, object> g && g.TryGetValue("index", out var i) && i != null && Convert.ToInt32(i) == index) return g;
            return null;
        }
    }
}
