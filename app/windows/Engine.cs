using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace GPUEnsembleApp
{
    /// <summary>Where the engine is: the zip's python\python.exe and app\ folder next to this exe.</summary>
    static class Engine
    {
        public static string BaseDir => AppDomain.CurrentDomain.BaseDirectory;

        /// <summary>python.exe of the zip; for a development build, GPUENSEMBLE_PYTHON or python on PATH.</summary>
        public static string Python
        {
            get
            {
                var env = Environment.GetEnvironmentVariable("GPUENSEMBLE_PYTHON");
                if (!string.IsNullOrEmpty(env)) return env!;
                var zip = Path.Combine(BaseDir, "python", "python.exe");
                return File.Exists(zip) ? zip : "python";
            }
        }

        public static string Home
        {
            get
            {
                var env = Environment.GetEnvironmentVariable("GPUENSEMBLE_HOME");
                return string.IsNullOrEmpty(env) ? Path.Combine(BaseDir, "data") : env!;
            }
        }

        public static ProcessStartInfo StartInfo(IEnumerable<string> args)
        {
            var psi = new ProcessStartInfo(Python, "-m gpuensemble_planner " + Quote(args))
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                WorkingDirectory = BaseDir,
            };
            psi.EnvironmentVariables["GPUENSEMBLE_HOME"] = Home;
            psi.EnvironmentVariables["GPUENSEMBLE_PROG"] = "gpuensemble.cmd";
            var models = Settings.ModelsDir;
            if (models != null) psi.EnvironmentVariables["GPUENSEMBLE_MODELS"] = models;
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            return psi;
        }

        static string Quote(IEnumerable<string> args)
        {
            var sb = new StringBuilder();
            foreach (var a in args)
            {
                if (sb.Length > 0) sb.Append(' ');
                if (a.Length > 0 && a.IndexOfAny(new[] { ' ', '\t', '"' }) < 0) { sb.Append(a); continue; }
                sb.Append('"').Append(a.Replace("\"", "\\\"")).Append('"');
            }
            return sb.ToString();
        }

        /// <summary>
        /// Runs <c>catalog --update</c>: the newest checked list of models from the repository, new models
        /// found on Hugging Face, and the start of each one's file (a few MB), so the next <c>state</c> lists them.
        /// Returns what it said.
        /// </summary>
        /// <param name="progress">Called with (files read, files to read) as the engine says "read N/M".</param>
        public static Task<string> UpdateCatalog(Action<int, int>? progress = null)
        {
            return Task.Run(() =>
            {
                if (StateFile != null) return "";
                var p = new ChildProcess(StartInfo(new List<string> { "catalog", "--update" }));
                var output = new StringBuilder();
                p.Output += line =>
                {
                    var m = System.Text.RegularExpressions.Regex.Match(line ?? "", @"^read (\d+)/(\d+)\s*$");
                    if (m.Success) { progress?.Invoke(int.Parse(m.Groups[1].Value), int.Parse(m.Groups[2].Value)); return; }
                    lock (output) output.AppendLine(line);
                };
                p.Error += line => { lock (output) output.AppendLine(line); };
                p.Start();
                p.StandardInput.Close();
                // Each file read is kept, so a run that is stopped here goes on from there the next time.
                p.WaitForExit(1_800_000);
                if (!p.HasExited) { p.Stop(); output.AppendLine("timeout"); }
                lock (output) return output.ToString();
            });
        }

        /// <summary>
        /// Runs <c>catalog --add SOURCE</c> (a model from Hugging Face on this PC's own list) or
        /// <c>catalog --remove ID</c>. Returns whether it worked and what it said.
        /// </summary>
        public static Task<(bool Ok, string Output)> Catalog(params string[] args)
        {
            return Task.Run(() =>
            {
                if (StateFile != null) return (false, "");
                var list = new List<string> { "catalog" };
                list.AddRange(args);
                var p = new ChildProcess(StartInfo(list));
                var output = new StringBuilder();
                p.Output += line => { lock (output) output.AppendLine(line); };
                p.Error += line => { lock (output) output.AppendLine(line); };
                p.Start();
                p.StandardInput.Close();
                // It reads the repository's file list and the start of one file: seconds, a few minutes at most.
                if (!p.WaitForExit(600_000)) { p.Stop(); lock (output) output.AppendLine("timeout"); }
                var ok = p.ExitCode == 0;
                lock (output) return (ok, output.ToString());
            });
        }

        /// <summary>Runs <c>state</c> (see the planner's appstate.py) and parses its JSON.</summary>
        /// <summary>
        /// With <c>--state FILE</c> the app shows that saved <c>state</c> output instead of asking the engine:
        /// for checking how a page looks with many PCs or GPUs that are not there.
        /// </summary>
        public static string? StateFile { get; set; }

        public static Task<StateResult> ReadState(bool askOtherPcs)
        {
            return Task.Run(() =>
            {
                if (StateFile != null)
                {
                    try
                    {
                        var json = new JavaScriptSerializer { MaxJsonLength = 16 * 1024 * 1024 };
                        var d = json.DeserializeObject(System.IO.File.ReadAllText(StateFile, Encoding.UTF8)) as Dictionary<string, object>;
                        return new StateResult(d == null ? null : new State(d), "read " + StateFile + "\n");
                    }
                    catch (Exception e) { return new StateResult(null, e.Message); }
                }
                var args = new List<string> { "state" };
                if (askOtherPcs) { args.Add("--remote"); args.Add("auto"); }
                var p = new ChildProcess(StartInfo(args));
                var output = new StringBuilder();
                var errors = new StringBuilder();
                p.Output += line => { lock (output) output.AppendLine(line); };
                p.Error += line => { lock (errors) errors.AppendLine(line); };
                p.Start();
                p.StandardInput.Close();
                p.WaitForExit(120_000);
                if (!p.HasExited) { p.Stop(); return new StateResult(null, "timeout\n" + errors); }
                try
                {
                    var json = new JavaScriptSerializer { MaxJsonLength = 16 * 1024 * 1024 };
                    var d = json.DeserializeObject(output.ToString()) as Dictionary<string, object>;
                    return new StateResult(d == null ? null : new State(d), errors.ToString());
                }
                catch (Exception e) { return new StateResult(null, e.Message + "\n" + errors); }
            });
        }
    }

    /// <summary>What the user chose in the window, kept in data\app-settings.json.</summary>
    static class Settings
    {
        static string FilePath => Path.Combine(Engine.Home, "app-settings.json");

        static Dictionary<string, object> Read()
        {
            try
            {
                if (File.Exists(FilePath) &&
                    new JavaScriptSerializer().DeserializeObject(File.ReadAllText(FilePath, Encoding.UTF8)) is Dictionary<string, object> d)
                    return d;
            }
            catch (Exception) { }
            return new Dictionary<string, object>();
        }

        /// <summary>The folder models are downloaded to and looked for in; null for the default (data\models).</summary>
        public static string? ModelsDir
        {
            get => Get("models_dir");
            set => Put("models_dir", value);
        }

        /// <summary>The window's language, "ja" or "en"; null to follow the language of Windows.</summary>
        public static string? Language
        {
            get => Get("language") is string s && (s == "ja" || s == "en") ? s : null;
            set => Put("language", value);
        }

        static string? Get(string key) => Read().TryGetValue(key, out var v) && v is string s && s.Length > 0 ? s : null;

        static void Put(string key, string? value)
        {
            var d = Read();
            if (string.IsNullOrEmpty(value)) d.Remove(key); else d[key] = value!;
            Directory.CreateDirectory(Engine.Home);
            File.WriteAllText(FilePath, new JavaScriptSerializer().Serialize(d), new UTF8Encoding(false));
        }
    }

    sealed class StateResult
    {
        public StateResult(State? state, string log) { State = state; Log = log; }
        public State? State { get; }
        public string Log { get; }
    }

    /// <summary>The planner's <c>state</c> JSON with typed getters.</summary>
    sealed class State
    {
        readonly Dictionary<string, object> d;
        public State(Dictionary<string, object> d) { this.d = d; }

        public string? Best => d.TryGetValue("best", out var v) ? v as string : null;
        public bool LlamaCpp => d.TryGetValue("llama_cpp", out var v) && v is bool b && b;
        public string ModelsDir => Str(d, "models_dir");
        public IEnumerable<Dictionary<string, object>> Gpus => List("gpus");
        public IEnumerable<Dictionary<string, object>> Workers => List("workers");
        public IEnumerable<Dictionary<string, object>> Models => List("models");
        public IEnumerable<string> Notes { get { foreach (var o in Arr("notes")) if (o is string s) yield return s; } }

        public Dictionary<string, object>? Model(string id)
        {
            foreach (var m in Models) if (Str(m, "id") == id) return m;
            return null;
        }

        object[] Arr(string key) => d.TryGetValue(key, out var v) && v is object[] a ? a : new object[0];

        IEnumerable<Dictionary<string, object>> List(string key)
        {
            foreach (var o in Arr(key)) if (o is Dictionary<string, object> x) yield return x;
        }

        public static string Str(Dictionary<string, object> x, string key) =>
            x.TryGetValue(key, out var v) && v != null ? Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture) ?? "" : "";

        public static double Num(Dictionary<string, object> x, string key) =>
            x.TryGetValue(key, out var v) && v != null ? Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture) : 0.0;

        public static bool Flag(Dictionary<string, object> x, string key) => x.TryGetValue(key, out var v) && v is bool b && b;
    }

    /// <summary>
    /// A child process of the engine whose output arrives line by line. It goes into a job object that
    /// ends it when this app ends for any reason (the engine puts llama-server into a job of its own,
    /// which ends with it), and <see cref="Stop"/> ends it at once.
    /// </summary>
    sealed class ChildProcess
    {
        readonly Process proc;
        public event Action<string>? Output;
        public event Action<string>? Error;
        public event Action<int>? Exited;

        public ChildProcess(ProcessStartInfo psi)
        {
            proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
            proc.OutputDataReceived += (_, e) => { if (e.Data != null) Output?.Invoke(e.Data); };
            proc.ErrorDataReceived += (_, e) => { if (e.Data != null) Error?.Invoke(e.Data); };
            proc.Exited += (_, __) =>
            {
                proc.WaitForExit(); // lets the last output lines through first
                Exited?.Invoke(proc.ExitCode);
            };
        }

        public StreamWriter StandardInput => proc.StandardInput;
        public bool HasExited { get { try { return proc.HasExited; } catch (InvalidOperationException) { return true; } } }

        public void Start()
        {
            proc.Start();
            Job.Add(proc);
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
        }

        public bool WaitForExit(int ms) => proc.WaitForExit(ms);

        /// <summary>The exit code, once the last output line has been passed on (-1 while it runs).</summary>
        public int ExitCode
        {
            get
            {
                try { if (!proc.HasExited) return -1; proc.WaitForExit(); return proc.ExitCode; }
                catch (InvalidOperationException) { return -1; }
            }
        }

        public void Send(string line)
        {
            try { proc.StandardInput.WriteLine(line); proc.StandardInput.Flush(); }
            catch (IOException) { }
            catch (InvalidOperationException) { }
        }

        public void Stop()
        {
            try { if (!proc.HasExited) proc.Kill(); } catch (InvalidOperationException) { } catch (System.ComponentModel.Win32Exception) { }
        }
    }

    /// <summary>One job object for every child: closing its last handle (this app ending) ends them all.</summary>
    static class Job
    {
        static IntPtr handle = IntPtr.Zero;
        static readonly object gate = new object();

        public static void Add(Process p)
        {
            lock (gate)
            {
                if (handle == IntPtr.Zero) handle = Create();
                if (handle != IntPtr.Zero) AssignProcessToJobObject(handle, p.Handle);
            }
        }

        static IntPtr Create()
        {
            var job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) return job;
            var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            info.BasicLimitInformation.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            int size = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
            var ptr = Marshal.AllocHGlobal(size);
            try
            {
                Marshal.StructureToPtr(info, ptr, false);
                SetInformationJobObject(job, 9 /* JobObjectExtendedLimitInformation */, ptr, (uint)size);
            }
            finally { Marshal.FreeHGlobal(ptr); }
            return job;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        static extern IntPtr CreateJobObject(IntPtr attributes, string? name);

        [DllImport("kernel32.dll")]
        static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);

        [DllImport("kernel32.dll")]
        static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct IO_COUNTERS
        {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
            public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }
    }

    /// <summary>What the engine's console lines say, for the parts the app shows.</summary>
    static class Lines
    {
        // hub._progress: "  name: 12.34 / 59.03 GiB   85.2 MB/s  about 9 min left"
        static readonly Regex progress = new Regex(@"(?<done>\d+(?:\.\d+)?) / (?<total>\d+(?:\.\d+)?) GiB\s+(?<rate>\d+(?:\.\d+)?) MB/s\s+about (?<min>\d+) min left");
        // llamacpp.run: "ready: http://127.0.0.1:8080/v1 (...)"
        static readonly Regex ready = new Regex(@"^ready: (?<url>https?://\S+?)/v1\b");
        // discovery.Approvals.request: "PC 'name' at 1.2.3.4 asks to use these GPUs. ..."
        static readonly Regex asks = new Regex(@"^PC '(?<name>.*)' at (?<address>[0-9a-fA-F:.]+) asks to use these GPUs");
        // cli._up: "starting llama-server with NAME; ..."
        static readonly Regex starting = new Regex(@"^starting llama-server with (?<name>[^;]+);");

        public static bool Progress(string line, out double done, out double total, out int minutes)
        {
            var m = progress.Match(line);
            done = total = 0; minutes = 0;
            if (!m.Success) return false;
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            done = double.Parse(m.Groups["done"].Value, inv);
            total = double.Parse(m.Groups["total"].Value, inv);
            minutes = int.Parse(m.Groups["min"].Value, inv);
            return true;
        }

        public static string? Ready(string line) { var m = ready.Match(line); return m.Success ? m.Groups["url"].Value + "/" : null; }

        public static bool Asks(string line, out string name, out string address)
        {
            var m = asks.Match(line);
            name = m.Success ? m.Groups["name"].Value : "";
            address = m.Success ? m.Groups["address"].Value : "";
            return m.Success;
        }

        public static string? Starting(string line) { var m = starting.Match(line); return m.Success ? m.Groups["name"].Value : null; }
    }
}
