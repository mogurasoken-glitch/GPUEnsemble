using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using static GPUEnsembleApp.Ui;

namespace GPUEnsembleApp
{
    enum Page { Home, Models, Link, Log }

    /// <summary>The window: a title bar, a menu on the left, one page on the right, and an icon in the notification area.</summary>
    sealed class MainForm : Form
    {
        readonly ServerService server = new ServerService();
        readonly WorkerService worker = new WorkerService();
        readonly MonitorService monitor = new MonitorService();
        PullService? pull;

        State? state;
        bool reading;
        string readError = "";

        Page page = Page.Home;
        readonly Panel content = new DoubleBufferedPanel { Dock = DockStyle.Fill, AutoScroll = true };
        readonly FlowLayoutPanel nav = new FlowLayoutPanel();
        readonly Label summary = new Label();
        readonly Dictionary<Page, NavItem> navItems = new Dictionary<Page, NavItem>();
        readonly TextBox logBox = new TextBox();
        readonly NotifyIcon tray = new NotifyIcon();
        readonly ToolStripMenuItem trayStatus = new ToolStripMenuItem { Enabled = false };
        readonly ToolStripMenuItem trayChat;
        readonly ToolStripMenuItem trayStartStop;
        readonly ToolStripMenuItem trayOpen;
        readonly ToolStripMenuItem trayExit;
        readonly ContextMenuStrip trayMenu;
        readonly LinkLabel language = new LinkLabel();
        readonly Timer refreshLater = new Timer { Interval = 300 };
        bool told, exiting;

        /// <summary>
        /// With <c>--capture FOLDER</c> the app saves a picture of each page there once it has read the state,
        /// then exits: for checking the look on a PC nobody is watching.
        /// </summary>
        public string? CaptureTo { get; set; }

        /// <summary>
        /// With <c>--start-model ID</c> the app starts that model as soon as it has read the state, as a click on
        /// its Start button would. With <c>--capture</c> too, the app asks the ready model one short question (the
        /// monitor has no speed before the first answer) and takes the pictures once the monitor has read the
        /// speed (or the start failed, or <see cref="CaptureWait"/> ran out): for checking the pages of a running
        /// model on a PC nobody is watching.
        /// </summary>
        public string? StartModel { get; set; }
        bool startedModel;
        static readonly TimeSpan CaptureWait = TimeSpan.FromMinutes(40);
        static readonly TimeSpan SpeedWait = TimeSpan.FromMinutes(5);
        DateTime startedAt, readyAt;
        bool asked;
        readonly Timer captureTimer = new Timer { Interval = 2000 };

        public MainForm()
        {
            Ui.Init(this);
            Text = "GPUEnsemble";
            Icon = LoadIcon();
            BackColor = Ground;
            Font = Body;
            StartPosition = FormStartPosition.CenterScreen;
            Size = new Size(Px(1160), Px(800));
            MinimumSize = new Size(Px(780), Px(540));

            content.BackColor = Ground;
            content.Padding = new Padding(Px(34), Px(28), Px(34), Px(28));
            content.Resize += (s, e) => { if (page != Page.Log) refreshLater.Start(); };

            nav.Dock = DockStyle.Left;
            nav.Width = Px(220);
            nav.BackColor = Surface;
            nav.FlowDirection = FlowDirection.TopDown;
            nav.WrapContents = false;
            nav.Padding = new Padding(Px(14), Px(20), Px(14), Px(14));
            nav.Paint += (s, e) => { using (var p = new Pen(Line)) e.Graphics.DrawLine(p, nav.Width - 1, 0, nav.Width - 1, nav.Height); };
            AddNav(Page.Home);
            AddNav(Page.Models);
            AddNav(Page.Link);
            AddNav(Page.Log);
            // The other language, named in itself, so that it can be found whichever language shows now.
            language.AutoSize = true;
            language.Margin = new Padding(Px(14), Px(18), 0, 0);
            language.LinkBehavior = LinkBehavior.HoverUnderline;
            language.LinkColor = Muted;
            language.ActiveLinkColor = Purple;
            language.UseMnemonic = false;
            language.TabStop = true;
            language.LinkClicked += (s, e) => SwitchLanguage();
            nav.Controls.Add(language);
            nav.Controls.Add(new Label { Text = "GPUEnsemble " + Version(), Font = Small, ForeColor = Muted, AutoSize = true,
                                         Margin = new Padding(Px(14), Px(6), 0, 0), UseMnemonic = false });

            var header = new Panel { Dock = DockStyle.Top, Height = Px(56), BackColor = Surface, Padding = new Padding(Px(22), 0, Px(22), 0) };
            var brand = new FlowLayoutPanel { Dock = DockStyle.Left, AutoSize = true, WrapContents = false, BackColor = Surface, Padding = new Padding(0, Px(16), 0, 0) };
            brand.Controls.Add(new Logo());
            var name = new Label { Text = "GPUEnsemble", Font = Brand, ForeColor = Ink, AutoSize = true, Margin = new Padding(0, Px(-2), 0, 0), UseMnemonic = false };
            brand.Controls.Add(name);
            summary.Dock = DockStyle.Right;
            summary.AutoSize = false;
            summary.Width = Px(420);
            summary.TextAlign = ContentAlignment.MiddleRight;
            summary.Font = Small;
            summary.ForeColor = Muted;
            summary.UseMnemonic = false;
            header.Controls.Add(summary);
            header.Controls.Add(brand);

            // Docked last to first: the page fills what the header, its line and the menu leave.
            Controls.Add(content);
            Controls.Add(nav);
            Controls.Add(new GradientLine());
            Controls.Add(header);

            logBox.Multiline = true;
            logBox.ReadOnly = true;
            logBox.ScrollBars = ScrollBars.Both;
            logBox.WordWrap = false;
            logBox.Font = Mono;
            logBox.BackColor = Surface;
            logBox.ForeColor = Ink;
            logBox.BorderStyle = BorderStyle.None;
            logBox.Dock = DockStyle.Fill;

            trayChat = new ToolStripMenuItem("", null, (s, e) => OpenChat());
            trayStartStop = new ToolStripMenuItem("", null, (s, e) => { if (server.Running) server.Stop(); else StartServer(null); });
            trayOpen = new ToolStripMenuItem("", null, (s, e) => ShowWindow());
            trayExit = new ToolStripMenuItem("", null, (s, e) => ExitApp());
            trayMenu = new ContextMenuStrip();
            trayMenu.Items.Add(trayStatus);
            trayMenu.Items.Add(new ToolStripSeparator());
            trayMenu.Items.Add(trayChat);
            trayMenu.Items.Add(trayOpen);
            trayMenu.Items.Add(trayStartStop);
            trayMenu.Items.Add(new ToolStripSeparator());
            trayMenu.Items.Add(trayExit);
            tray.Icon = Icon;
            tray.Text = "GPUEnsemble";
            tray.ContextMenuStrip = trayMenu;
            Relabel();
            tray.Visible = true;
            tray.DoubleClick += (s, e) => ShowWindow();

            refreshLater.Tick += (s, e) => { refreshLater.Stop(); Render(); };
            foreach (var svc in new Service[] { server, worker })
            {
                svc.Changed += () => OnUi(Changed);
                svc.Progressed += () => OnUi(Progressed);
                svc.LogLine += line => OnUi(() => AppendLog(line));
            }
            server.Changed += () =>
            {
                // A model that just downloaded, or a server that stopped, changes what the pages show.
                if (!server.Running) OnUi(() => ReadState());
                OnUi(() =>
                {
                    if (server.State == RunState.Ready) monitor.Start();
                    else if (!server.Running) monitor.Stop();
                });
            };
            // A reading changes only the parts that show it, as a download's progress does.
            monitor.Updated += () => OnUi(() => { if (page != Page.Log) foreach (var update in live.ToArray()) update(); });
            captureTimer.Tick += (s, e) => CaptureWhenRunning();

            Load += (s, e) => { Show(Page.Home); ReadState(); UpdateCatalog(); };
        }

        static Icon LoadIcon()
        {
            try { return Icon.ExtractAssociatedIcon(Application.ExecutablePath) ?? SystemIcons.Application; }
            catch (ArgumentException) { return SystemIcons.Application; }
        }

        /// <summary>The time a download has left, as the engine estimates it in whole minutes.</summary>
        static string TimeLeft(int minutes) => minutes <= 0 ? T("残り 1 分足らず", "under a minute left")
            : T($"残り約 {minutes} 分", $"about {minutes} min left");

        /// <summary>The app's version (the same as the engine's in a release zip).</summary>
        static string Version()
        {
            var v = typeof(MainForm).Assembly.GetName().Version;
            return v == null ? "" : $"{v.Major}.{v.Minor}.{v.Build}";
        }

        static string NavText(Page p) => p switch
        {
            Page.Home => T("ホーム", "Home"),
            Page.Models => T("モデル", "Models"),
            Page.Link => T("つながる PC", "Other PCs"),
            _ => T("ログ", "Log"),
        };

        /// <summary>The texts and fonts made once with the window, in the language that shows now.</summary>
        void Relabel()
        {
            Font = Body;
            summary.Font = Small;
            foreach (var kv in navItems)
            {
                kv.Value.Text = NavText(kv.Key);
                kv.Value.AccessibleName = kv.Value.Text;
                kv.Value.Invalidate();
            }
            language.Font = Small;
            language.Text = Japanese ? "English" : "日本語";
            language.AccessibleName = T("表示の言語を英語にする", "Show the window in Japanese");
            trayMenu.Font = Body;
            trayChat.Text = T("チャットを開く", "Open chat");
            trayOpen.Text = T("ウィンドウを開く", "Open window");
            trayExit.Text = T("終了", "Exit");
            UpdateTray();
            UpdateSummary();
        }

        /// <summary>Japanese to English or back, kept in data\app-settings.json for the next start.</summary>
        void SwitchLanguage()
        {
            SetLanguage(!Japanese);
            try { Settings.Language = Japanese ? "ja" : "en"; }
            catch (Exception e) when (e is System.IO.IOException || e is UnauthorizedAccessException) { }
            Relabel();
            Render();
        }

        void AddNav(Page p)
        {
            var item = new NavItem(NavText(p)) { Width = Px(190) };
            item.Click += (s, e) => Show(p);
            navItems[p] = item;
            nav.Controls.Add(item);
        }

        void OnUi(Action a)
        {
            if (IsDisposed || exiting) return;
            try { if (InvokeRequired) BeginInvoke(a); else a(); }
            catch (InvalidOperationException) { }
        }

        void Show(Page p)
        {
            page = p;
            foreach (var kv in navItems) kv.Value.Selected = kv.Key == p;
            Render();
        }

        /// <summary>
        /// The parts of the page that show download progress, each with how to bring itself up to date: a
        /// progress line changes only these, so the page is not rebuilt (and does not flicker) several times a second.
        /// </summary>
        readonly List<Action> live = new List<Action>();

        void Progressed()
        {
            UpdateTray();
            if (page == Page.Log) return;
            foreach (var update in live) update();
        }

        /// <summary>A bar and its text that follow a download.</summary>
        void LiveProgress(Control parent, Func<double> fraction, Func<string> text, int width, int height = 8)
        {
            var meter = new Meter(fraction(), DeepBlue, Purple, width, height);
            var label = Label(text(), Small, Muted);
            parent.Controls.Add(meter);
            parent.Controls.Add(label);
            live.Add(() =>
            {
                if (meter.IsDisposed) return;
                meter.Fraction = fraction();
                var t = text();
                if (label.Text != t) label.Text = t;
            });
        }

        void Changed()
        {
            UpdateTray();
            if (page != Page.Log) Render();
        }

        // --- engine ------------------------------------------------------------------------------

        /// <summary>Brings the list of models up to date in the background, then reads the state again.</summary>
        async void UpdateCatalog()
        {
            if (updatingCatalog) return;
            updatingCatalog = true;
            catalogRead = 0; catalogTotal = 0;
            if (page == Page.Models) Render();
            var log = await Engine.UpdateCatalog((done, total) => OnUi(() =>
            {
                catalogRead = done; catalogTotal = total;
                if (catalogNote != null && !catalogNote.IsDisposed) catalogNote.Text = CatalogNoteText();
            }));
            updatingCatalog = false;
            foreach (var line in log.Split('\n'))
                if (line.Trim().Length > 0) AppendLog("[catalog] " + line.TrimEnd());
            ReadState();
        }

        bool updatingCatalog, addingModel;

        /// <summary>
        /// Asks for a GGUF repository on Hugging Face (or a link to its page or one of its files) and puts
        /// that model on this PC's own list (<c>catalog --add</c>), so it can be downloaded and started like the others.
        /// </summary>
        async void AddModel()
        {
            if (addingModel) return;
            string source;
            using (var f = new Form
            {
                Text = T("モデルを追加", "Add a model"),
                FormBorderStyle = FormBorderStyle.FixedDialog,
                StartPosition = FormStartPosition.CenterParent,
                MinimizeBox = false,
                MaximizeBox = false,
                ShowInTaskbar = false,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                BackColor = Surface,
                Font = Body,
                Padding = new Padding(Px(16)),
            })
            {
                var col = Column(0);
                col.Controls.Add(Wrap(T("Hugging Face にある GGUF のリポジトリ名（例: JonathanColetti/Qwen3.8-27B-Uncensored-GGUF）か、そのページやファイルのリンクを入れてください。" +
                                        "ファイルのリンクならその量子化を、リポジトリだけなら Q4_K_M を使います。この PC の一覧にだけ加わり、手で確かめたモデルではありません。",
                                        "Enter a GGUF repository on Hugging Face (owner/name, e.g. JonathanColetti/Qwen3.8-27B-Uncensored-GGUF), or a link to its page or to one of its files. " +
                                        "A file link takes that quantization; a repository alone takes Q4_K_M. Only this PC's list gets it, and it is not a checked model."),
                                      Px(520), Small, Muted));
                var box = new TextBox { Width = Px(520), Font = Body, Margin = new Padding(0, Px(8), 0, Px(10)) };
                col.Controls.Add(box);
                var ok = new System.Windows.Forms.Button { Text = T("追加", "Add"), DialogResult = DialogResult.OK, AutoSize = true };
                var cancel = new System.Windows.Forms.Button { Text = T("やめる", "Cancel"), DialogResult = DialogResult.Cancel, AutoSize = true };
                col.Controls.Add(Row(ok, cancel));
                f.Controls.Add(col);
                f.AcceptButton = ok;
                f.CancelButton = cancel;
                if (f.ShowDialog(this) != DialogResult.OK) return;
                source = box.Text.Trim();
            }
            if (source.Length == 0) return;
            addingModel = true;
            AppendLog("[catalog] add " + source);
            if (page == Page.Models) Render();
            var r = await Engine.Catalog("--add", source);
            addingModel = false;
            foreach (var line in r.Output.Split('\n'))
                if (line.Trim().Length > 0) AppendLog("[catalog] " + line.TrimEnd());
            if (r.Ok) ReadState();
            else if (page == Page.Models) Render();
            MessageBox.Show(this, (r.Ok ? T("一覧に追加しました。", "Added to the list.") : T("追加できませんでした。", "Could not add it.")) +
                                  "\n\n" + r.Output.Trim(),
                "GPUEnsemble", MessageBoxButtons.OK, r.Ok ? MessageBoxIcon.Information : MessageBoxIcon.Warning);
        }

        /// <summary>Takes a model the user added off this PC's list (<c>catalog --remove</c>); its downloaded file stays.</summary>
        async void RemoveModel(string id, string name)
        {
            var go = MessageBox.Show(this, T($"{name} をこの PC の一覧から外しますか？ダウンロード済みのファイルは消しません。",
                                            $"Take {name} off this PC's list? A downloaded file is kept."),
                "GPUEnsemble", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (go != DialogResult.Yes) return;
            var r = await Engine.Catalog("--remove", id);
            foreach (var line in r.Output.Split('\n'))
                if (line.Trim().Length > 0) AppendLog("[catalog] " + line.TrimEnd());
            ReadState();
        }
        int catalogRead, catalogTotal;
        Label? catalogNote;

        string CatalogNoteText()
        {
            if (catalogTotal <= 0) return T("新しいモデルを Hugging Face で探しています…", "Looking for new models on Hugging Face...");
            return T($"モデルのファイルを読んでいます… {catalogRead}/{catalogTotal}（初回は数分かかります。ほかの操作はできます）",
                     $"Reading the models' files... {catalogRead}/{catalogTotal} (a few minutes the first time; the app can be used meanwhile)");
        }

        async void ReadState()
        {
            if (reading) return;
            reading = true;
            if (page != Page.Log) Render();
            var r = await Engine.ReadState(askOtherPcs: !worker.Running);
            reading = false;
            if (r.State != null) { state = r.State; readError = ""; }
            else readError = r.Log.Trim();
            foreach (var line in r.Log.Split('\n'))
                if (line.Trim().Length > 0) AppendLog("[state] " + line.TrimEnd());
            UpdateSummary();
            UpdateTray();
            if (page != Page.Log) Render();
            if (StartModel != null && !startedModel && state != null)
            {
                startedModel = true;
                startedAt = DateTime.UtcNow;
                StartServer(StartModel);
                if (CaptureTo != null) captureTimer.Start();
            }
            else if (CaptureTo != null && !startedModel) CapturePages(CaptureTo);
        }

        /// <summary>Takes the pictures of <c>--capture</c> with <c>--start-model</c> when there is a running model to show.</summary>
        void CaptureWhenRunning()
        {
            if (CaptureTo == null) { captureTimer.Stop(); return; }
            var ready = server.State == RunState.Ready;
            if (ready && readyAt == default(DateTime)) readyAt = DateTime.UtcNow;
            var m = monitor.Last;
            // The monitor counts from the end of the warm-up: a question sent before that would not count.
            if (ready && !asked && m != null && State.Flag(m, "ready")) { asked = true; AskOnce(server.ChatUrl); }
            var speed = ready && m != null && m.TryGetValue("speed", out var v) && v != null && !State.Flag(m, "busy");
            var gaveUp = (ready && DateTime.UtcNow - readyAt > SpeedWait) || DateTime.UtcNow - startedAt > CaptureWait;
            if (!speed && !gaveUp && server.Running) return;
            captureTimer.Stop();
            if (page != Page.Home) Show(Page.Home);
            CapturePages(CaptureTo);
        }

        /// <summary>One short question to the running model, so that the monitor has an answer to time.</summary>
        void AskOnce(string chatUrl)
        {
            const string body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 30.\"}],\"max_tokens\":64}";
            var client = new System.Net.WebClient();
            client.Headers[System.Net.HttpRequestHeader.ContentType] = "application/json";
            client.UploadStringCompleted += (s, e) =>
            {
                AppendLog(e.Error == null ? "[capture] the model answered the test question" : "[capture] test question failed: " + e.Error.Message);
                client.Dispose();
            };
            try { client.UploadStringAsync(new Uri(new Uri(chatUrl), "v1/chat/completions"), "POST", body); }
            catch (Exception ex) { AppendLog("[capture] test question failed: " + ex.Message); client.Dispose(); }
        }

        async void CapturePages(string folder)
        {
            CaptureTo = null;
            System.IO.Directory.CreateDirectory(folder);
            TopMost = true;  // the pictures come from the screen: nothing may cover the window
            // A saved state (--state) may hold many PCs, and a running model's home page is long: a tall window shows all of it.
            if (Engine.StateFile != null || startedModel)
            {
                var screen = Screen.FromControl(this).WorkingArea;
                WindowState = FormWindowState.Normal;
                Top = screen.Top;  // all of the taller window on the screen
                Height = Math.Min(Px(1900), screen.Height);
            }
            foreach (Page p in Enum.GetValues(typeof(Page)))
            {
                Show(p);
                Refresh();
                await System.Threading.Tasks.Task.Delay(1500);
                Refresh();
                // What is on the screen, as the user sees it (DrawToBitmap paints each control apart and
                // misses faults that only show on the screen); the drawing when the window is not visible.
                Bitmap bmp;
                try
                {
                    Activate();
                    var area = Rectangle.Intersect(VisibleBounds(), Screen.FromControl(this).Bounds);
                    bmp = new Bitmap(area.Width, area.Height);
                    using (var g = Graphics.FromImage(bmp)) g.CopyFromScreen(area.Location, Point.Empty, area.Size);
                }
                catch (Exception)
                {
                    bmp = new Bitmap(Width, Height);
                    DrawToBitmap(bmp, new Rectangle(0, 0, Width, Height));
                }
                using (bmp)
                    bmp.Save(System.IO.Path.Combine(folder, $"{(int)p + 1}-{p.ToString().ToLowerInvariant()}.png"), System.Drawing.Imaging.ImageFormat.Png);
            }
            ExitApp();
        }

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        struct RECT { public int Left, Top, Right, Bottom; }

        [System.Runtime.InteropServices.DllImport("dwmapi.dll")]
        static extern int DwmGetWindowAttribute(IntPtr hwnd, int attribute, out RECT value, int size);
        const int DWMWA_EXTENDED_FRAME_BOUNDS = 9;

        /// <summary>
        /// The window as drawn on the screen. <see cref="Control.Bounds"/> also holds the resize border Windows 10
        /// keeps invisible, through which a picture of the screen shows what is behind the window.
        /// </summary>
        Rectangle VisibleBounds()
        {
            try
            {
                if (DwmGetWindowAttribute(Handle, DWMWA_EXTENDED_FRAME_BOUNDS, out var r, 16) == 0 && r.Right > r.Left && r.Bottom > r.Top)
                    return Rectangle.FromLTRB(r.Left, r.Top, r.Right, r.Bottom);
            }
            catch (Exception) { }  // no DWM: the bounds as the form knows them
            return Bounds;
        }

        void StartServer(string? modelId)
        {
            if (server.Running || worker.Running) return;
            server.ModelId = modelId;
            server.Start();
            Show(Page.Home);
        }

        void StartPull(string id)
        {
            if (pull != null && pull.Running) return;
            pull = new PullService(id);
            pull.Changed += () => OnUi(() =>
            {
                if (pull != null && !pull.Running) ReadState();
                Changed();
            });
            pull.Progressed += () => OnUi(Progressed);
            pull.LogLine += line => OnUi(() => AppendLog(line));
            pull.Start();
        }

        void OpenChat()
        {
            try { Process.Start(new ProcessStartInfo(server.ChatUrl) { UseShellExecute = true }); }
            catch (Exception e) { MessageBox.Show(this, e.Message, "GPUEnsemble"); }
        }

        void AppendLog(string line)
        {
            if (logBox.IsDisposed) return;
            if (logBox.TextLength > 2_000_000) logBox.Text = logBox.Text.Substring(1_000_000);
            logBox.AppendText(line + Environment.NewLine);
        }

        void UpdateSummary()
        {
            if (state == null) { summary.Text = ""; return; }
            var gpus = state.Gpus.ToList();
            var pcs = gpus.Select(g => State.Str(g, "pc")).Distinct().Count();
            var vram = gpus.Sum(g => State.Num(g, "vram_gib"));
            summary.Text = gpus.Count == 0 ? T("GPU なし", "no GPU") :
                T($"GPU {gpus.Count} 枚 ・ PC {pcs} 台 ・ {Gib(vram)}", $"{gpus.Count} GPU{(gpus.Count > 1 ? "s" : "")} · {pcs} PC{(pcs > 1 ? "s" : "")} · {Gib(vram)}");
        }

        void UpdateTray()
        {
            string status;
            if (server.State == RunState.Ready) status = T($"{CurrentName()} が動いています", $"{CurrentName()} is running");
            else if (server.Running) status = T("起動しています…", "Starting…");
            else if (worker.Running) status = T("GPU を貸しています", "Lending GPUs");
            else status = T("止まっています", "Stopped");
            trayStatus.Text = status;
            var tip = "GPUEnsemble: " + status;
            tray.Text = tip.Length > 63 ? "GPUEnsemble" : tip;
            trayChat.Enabled = server.State == RunState.Ready;
            trayStartStop.Text = server.Running ? T("モデルを止める", "Stop the model") : T("おすすめのモデルを起動", "Start the recommended model");
            trayStartStop.Enabled = server.Running || !worker.Running;
        }

        /// <summary>The catalog id of the model running or starting, else the recommended one.</summary>
        string? CurrentId() => server.Running || server.State == RunState.Ready ? server.RunningModel ?? server.ModelId ?? state?.Best : state?.Best;

        string CurrentName()
        {
            var id = CurrentId();
            var m = id != null ? state?.Model(id) : null;
            return m != null ? State.Str(m, "name") : id ?? "";
        }

        // --- pages -------------------------------------------------------------------------------

        int Width0 => Math.Max(Px(340), content.ClientSize.Width - content.Padding.Horizontal - Px(24));

        [System.Runtime.InteropServices.DllImport("user32.dll")]
        static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, IntPtr lParam);
        const int WM_SETREDRAW = 0x000B;

        void Render()
        {
            live.Clear();
            // Nothing is drawn while the page is rebuilt, then it is drawn once: no half-built page shows.
            var freeze = content.IsHandleCreated && Visible;
            if (freeze) SendMessage(content.Handle, WM_SETREDRAW, IntPtr.Zero, IntPtr.Zero);
            try { RenderPage(); }
            finally
            {
                if (freeze)
                {
                    SendMessage(content.Handle, WM_SETREDRAW, (IntPtr)1, IntPtr.Zero);
                    content.Invalidate(true);
                }
            }
        }

        void RenderPage()
        {
            content.SuspendLayout();
            var scroll = content.AutoScrollPosition;
            // The log box lives on across pages: take it out before its card is thrown away with the page.
            logBox.Parent?.Controls.Remove(logBox);
            foreach (Control c in content.Controls.Cast<Control>().ToArray())
                c.Dispose();
            content.Controls.Clear();
            if (page == Page.Log)
            {
                var box = Card();
                box.Dock = DockStyle.Fill;
                box.AutoSize = false;
                box.Controls.Add(logBox);
                logBox.Dock = DockStyle.None;
                logBox.Margin = new Padding(0);
                logBox.Size = new Size(content.ClientSize.Width - content.Padding.Horizontal - Px(50), content.ClientSize.Height - content.Padding.Vertical - Px(44));
                content.Controls.Add(box);
                content.ResumeLayout();
                return;
            }
            var col = Column(0);
            col.Location = new Point(content.Padding.Left, content.Padding.Top);
            switch (page)
            {
                case Page.Home: HomePage(col); break;
                case Page.Models: ModelsPage(col); break;
                case Page.Link: LinkPage(col); break;
            }
            content.Controls.Add(col);
            Solidify(col);
            content.ResumeLayout();
            content.AutoScrollPosition = new Point(-scroll.X, -scroll.Y);
        }

        /// <summary>
        /// Gives every see-through panel, row and label of a page the solid color of what it stands on. A
        /// see-through panel does not paint its own area, so pixels left there from before (text that moved when
        /// the page was laid out or scrolled) stayed visible around the buttons until something repainted them.
        /// </summary>
        static void Solidify(Control c)
        {
            foreach (Control child in c.Controls)
            {
                if (!(child is Card) && !(child is Pill) && child.BackColor.A < 255) child.BackColor = Behind(child);
                Solidify(child);
            }
            if (!(c is Card) && c.BackColor.A < 255) c.BackColor = Behind(c);
        }

        static void Heading(Control col, string text) =>
            col.Controls.Add(new Label { Text = text, Font = H1, ForeColor = Ink, AutoSize = true, Margin = new Padding(0, 0, 0, Px(18)), UseMnemonic = false });

        static void Section(Control col, string text) =>
            col.Controls.Add(new Label { Text = text, Font = H2, ForeColor = Ink, AutoSize = true, Margin = new Padding(0, Px(14), 0, Px(10)), UseMnemonic = false });

        void Reading(Control col)
        {
            if (reading)
                col.Controls.Add(Label(T("この PC とつながる PC を調べています…", "Looking at this PC and the PCs it can use…"), Body, Muted));
            else if (state == null && readError.Length > 0)
            {
                var card = Card(Notice, NoticeLine);
                card.Controls.Add(Label(T("エンジンを動かせませんでした", "The engine did not run"), Bold));
                card.Controls.Add(Wrap(readError, Width0 - Px(44), Small, Muted));
                card.Controls.Add(Row(Button(T("もう一度", "Try again"), false, (s, e) => ReadState())));
                col.Controls.Add(card);
            }
        }

        /// <summary>
        /// How the GPUs line up with a model's split. <c>Stage[k]</c> is the card in <c>state.Gpus</c> that holds
        /// stage k (cards with the same name are taken in turn), <c>Order</c> the cards as the split goes from the
        /// first layer to the last, then the cards it leaves out, and a card's color is its place in that order:
        /// the bar, its legend, the GPU cards and the worker dots all read the same way, left to right.
        /// </summary>
        sealed class GpuMap
        {
            /// <summary>Above this many GPUs the pages group them by PC: one color and one row per PC.</summary>
            public const int ManyGpus = 8;

            public readonly List<int> Stage = new List<int>();
            public readonly List<int> Order = new List<int>();
            /// <summary>The PCs in the order of the split, then the others: each with its cards in <c>Order</c>.</summary>
            public readonly List<(string pc, List<int> gpus)> Pcs = new List<(string, List<int>)>();
            public readonly bool Many;
            readonly List<Dictionary<string, object>> gpus;
            readonly Dictionary<int, int> place = new Dictionary<int, int>();
            readonly Dictionary<string, int> pcPlace = new Dictionary<string, int>();

            public GpuMap(State? state, IEnumerable<Dictionary<string, object>> stages)
            {
                gpus = state != null ? state.Gpus.ToList() : new List<Dictionary<string, object>>();
                Many = gpus.Count > ManyGpus;
                var taken = new HashSet<int>();
                var list = stages.ToList();
                foreach (var s in list)
                {
                    var name = State.Str(s, "gpu");
                    var i = -1;
                    for (int j = 0; j < gpus.Count && i < 0; j++)
                        if (!taken.Contains(j) && State.Str(gpus[j], "name") == name) i = j;
                    if (i >= 0) taken.Add(i);
                    Stage.Add(i);
                }
                // Cards with layers first, in the order of their layers; then the rest as listed.
                foreach (var k in Enumerable.Range(0, list.Count).Where(k => Stage[k] >= 0 && HasLayers(list[k])).OrderBy(k => State.Num(list[k], "first")))
                    Add(Stage[k]);
                for (int j = 0; j < gpus.Count; j++) Add(j);
            }

            void Add(int i)
            {
                if (place.ContainsKey(i)) return;
                place[i] = Order.Count;
                Order.Add(i);
                var pc = Pc(i);
                if (!pcPlace.ContainsKey(pc))
                {
                    pcPlace[pc] = Pcs.Count;
                    Pcs.Add((pc, new List<int>()));
                }
                Pcs[pcPlace[pc]].gpus.Add(i);
            }

            public string Pc(int gpu) => gpu >= 0 && gpu < gpus.Count ? State.Str(gpus[gpu], "pc") : "";

            public Color PcColor(string pc) => pcPlace.TryGetValue(pc, out var c) ? Ui.GpuColor(c) : Ui.Idle;

            public Color Color(int gpu)
            {
                if (gpu < 0 || !place.TryGetValue(gpu, out var c)) return Ui.Idle;
                return Many ? PcColor(Pc(gpu)) : Ui.GpuColor(c);
            }
        }

        /// <summary>A PC's name as the pages show it: the engine calls the PC the app runs on "this PC".</summary>
        static string PcName(string pc) => pc == "this PC" ? T("この PC", "This PC") : pc;

        static bool HasLayers(Dictionary<string, object> s) => s.TryGetValue("first", out var f) && f != null;

        static IEnumerable<Dictionary<string, object>> Stages(Dictionary<string, object>? m) =>
            m != null && m.TryGetValue("stages", out var v) && v is object[] a ? a.OfType<Dictionary<string, object>>() : Enumerable.Empty<Dictionary<string, object>>();

        void HomePage(FlowLayoutPanel col)
        {
            Heading(col, T("ホーム", "Home"));
            var w = Width0;
            var card = Card();
            card.MinimumSize = new Size(w, 0);
            card.MaximumSize = new Size(w, 0);
            var id = CurrentId();
            var model = id != null ? state?.Model(id) : null;
            var inner = w - Px(48);

            switch (server.State)
            {
                case RunState.Ready:
                    card.Controls.Add(Status(T("動いています", "Running"), Good));
                    card.Controls.Add(Label(CurrentName(), Big));
                    if (model != null) LiveSpeed(card, model);
                    LiveText(card, MonitorText, inner, Body, Ink);
                    card.Controls.Add(Label(T($"チャット: {server.ChatUrl}　API: {server.ChatUrl}v1", $"Chat: {server.ChatUrl}   API: {server.ChatUrl}v1"), Small, Muted));
                    card.Controls.Add(Row(
                        Button(T("チャットを開く", "Open chat"), true, (s, e) => OpenChat()),
                        Button(T("停止", "Stop"), false, (s, e) => server.Stop())));
                    break;
                case RunState.Starting:
                case RunState.Downloading:
                    card.Controls.Add(Status(server.State == RunState.Downloading ? T("ダウンロードしています", "Downloading") : T("起動しています", "Starting"), DeepBlue));
                    card.Controls.Add(Label(CurrentName(), Big));
                    if (server.State == RunState.Downloading)
                    {
                        LiveProgress(card, () => server.Total > 0 ? server.Done / server.Total : 0,
                            () => T($"{server.Done:0.0} / {server.Total:0.0} GB　{TimeLeft(server.MinutesLeft)}", $"{server.Done:0.0} / {server.Total:0.0} GB, {TimeLeft(server.MinutesLeft)}"), inner, 10);
                    }
                    else
                        card.Controls.Add(Wrap(T("初回は GPU の準備に数分かかることがあります。経過はログに出ます。", "The first start can take a few minutes to prepare the GPUs. The log shows how far it is."), inner, Small, Muted));
                    card.Controls.Add(Row(
                        Button(T("中止", "Cancel"), false, (s, e) => server.Stop()),
                        Button(T("ログを見る", "Show log"), false, (s, e) => Show(Page.Log))));
                    break;
                default:
                    if (server.State == RunState.Failed)
                    {
                        card.Controls.Add(Status(T("エラーで止まりました", "Stopped with an error"), Bad));
                        if (server.LastError.Length > 0) card.Controls.Add(Wrap(server.LastError, inner, Body, Muted));
                    }
                    else
                        card.Controls.Add(Status(T("止まっています", "Stopped"), Idle));
                    if (worker.Running)
                    {
                        card.Controls.Add(Wrap(T("この PC は GPU を貸しています。ここでモデルを動かすには、「つながる PC」で貸すのをやめてください。",
                                                 "This PC lends its GPUs. To run a model here, stop lending on the Other PCs page."), inner, Body, Muted));
                        card.Controls.Add(Row(Button(T("つながる PC へ", "Other PCs"), false, (s, e) => Show(Page.Link))));
                    }
                    else if (model != null)
                    {
                        card.Controls.Add(Label(T("おすすめのモデル", "Recommended model"), Small, Muted));
                        card.Controls.Add(Label(State.Str(model, "name"), Big));
                        card.Controls.Add(Speed(State.Num(model, "speed"), Stages(model).Count(s => s.TryGetValue("first", out var f) && f != null), estimate: true));
                        if (!State.Flag(model, "downloaded"))
                            card.Controls.Add(Label(T($"初回は {Gib(State.Num(model, "file_gib"))} をダウンロードします", $"Downloads {Gib(State.Num(model, "file_gib"))} the first time"), Small, Muted));
                        card.Controls.Add(Row(
                            Button(T("起動", "Start"), true, (s, e) => StartServer(null)),
                            Button(T("ほかのモデルを選ぶ", "Choose another model"), false, (s, e) => Show(Page.Models))));
                    }
                    else if (state != null)
                        card.Controls.Add(Label(T("この PC で動かせるモデルが見つかりませんでした。", "No model in the catalog fits on this PC."), Body, Muted));
                    break;
            }

            // How the model is split over the GPUs, in their colors.
            var stages = Stages(model).ToList();
            var map = new GpuMap(state, stages);
            var parts = Enumerable.Range(0, stages.Count).Where(k => HasLayers(stages[k])).OrderBy(k => State.Num(stages[k], "first")).ToList();
            if (parts.Count > 0 && !worker.Running)
            {
                var split = Label(T("GPU ごとの分担（帯の幅は使うメモリの量）", "Each GPU's share (widths: memory it uses)"), Small, Muted);
                split.Margin = new Padding(0, Px(16), 0, 0);
                card.Controls.Add(split);
                // As wide as the memory each GPU uses for the model (in 0.1 GB), not its number of layers: a GPU whose
                // layers run their experts on the CPU holds many layers in little memory.
                var bar = parts.Select(k => ((int)Math.Round(State.Num(stages[k], "used_gib") * 10), map.Color(map.Stage[k]))).ToArray();
                card.Controls.Add(new SplitBar(bar, inner));
                var legend = Row();
                legend.Margin = new Padding(0);
                if (map.Many)
                    foreach (var run in PcRuns(map, stages, parts))
                    {
                        var item = Row(new Dot(map.PcColor(run.pc)) { Margin = new Padding(0, Px(5), Px(6), 0) },
                            Label($"{PcName(run.pc)}  {run.gib:0} GB ・ {T("層", "layers")} {run.first:0}-{run.last:0}", Small, Muted));
                        item.WrapContents = false;
                        item.Margin = new Padding(0, 0, Px(18), 0);
                        legend.Controls.Add(item);
                    }
                else foreach (var k in parts)
                {
                    var s = stages[k];
                    var item = Row(new Dot(map.Color(map.Stage[k])) { Margin = new Padding(0, Px(5), Px(6), 0) },
                        Label($"{ShortGpu(State.Str(s, "gpu"))}  {State.Num(s, "used_gib"):0.#} GB ・ {T("層", "layers")} {State.Num(s, "first"):0}-{State.Num(s, "last"):0}", Small, Muted));
                    item.WrapContents = false;
                    item.Margin = new Padding(0, 0, Px(18), 0);
                    legend.Controls.Add(item);
                }
                var cpu = model != null ? State.Num(model, "cpu_layers") : 0;
                if (cpu > 0) legend.Controls.Add(Label(T($"CPU  {cpu:0} 層", $"CPU  {cpu:0} layers"), Small, Muted));
                card.Controls.Add(legend);
            }
            col.Controls.Add(card);
            Reading(col);

            if (state == null) return;
            var useOf = new Dictionary<int, double>();
            for (int k = 0; k < stages.Count; k++)
                if (map.Stage[k] >= 0) useOf[map.Stage[k]] = (useOf.TryGetValue(map.Stage[k], out var u0) ? u0 : 0) + State.Num(stages[k], "used_gib");
            if (map.Many)
            {
                PcList(col, map, useOf, w);
                foreach (var note in state.Notes)
                    col.Controls.Add(Wrap(note, w, Small, Muted));
                col.Controls.Add(Row(Button(T("もう一度調べる", "Look again"), false, (s, e) => ReadState())));
                return;
            }
            Section(col, T("使える GPU", "GPUs"));
            var gpus = new FlowLayoutPanel { AutoSize = true, AutoSizeMode = AutoSizeMode.GrowAndShrink, MaximumSize = new Size(w + Px(14), 0), Margin = new Padding(0), BackColor = Color.Transparent };
            var gpuList = state.Gpus.ToList();
            foreach (var gi in map.Order)
                gpus.Controls.Add(GpuCard(gpuList[gi], map.Color(gi), useOf.TryGetValue(gi, out var used) ? used : (double?)null, Px(250), showPc: true));
            if (gpus.Controls.Count == 0)
                gpus.Controls.Add(Label(T("NVIDIA の GPU が見つかりませんでした。", "No NVIDIA GPU was found."), Body, Muted));
            col.Controls.Add(gpus);
            foreach (var note in state.Notes)
                col.Controls.Add(Wrap(note, w, Small, Muted));
            col.Controls.Add(Row(Button(T("もう一度調べる", "Look again"), false, (s, e) => ReadState())));
        }

        /// <summary>A text that follows the monitor's readings (and turns red when <paramref name="text"/> says so).</summary>
        void LiveText(Control parent, Func<(string text, bool bad)> text, int width, Font font, Color color)
        {
            var first = text();
            var label = Wrap(first.text, width, font, first.bad ? Bad : color);
            parent.Controls.Add(label);
            live.Add(() =>
            {
                if (label.IsDisposed) return;
                var t = text();
                if (label.Text != t.text) label.Text = t.text;
                var c = t.bad ? Bad : color;
                if (label.ForeColor != c) label.ForeColor = c;
            });
        }

        /// <summary>
        /// The running model's speed in large figures: the measured one once the monitor has it, the plan's
        /// estimate before that.
        /// </summary>
        void LiveSpeed(Control parent, Dictionary<string, object> model)
        {
            var estimate = State.Num(model, "speed");
            var gpus = Stages(model).Count(s => s.TryGetValue("first", out var f) && f != null);
            var row = Speed(estimate, gpus, estimate: true);
            var number = (Label)row.Controls[0];
            var unit = (Label)row.Controls[1];
            parent.Controls.Add(row);
            Action update = () =>
            {
                if (row.IsDisposed) return;
                var real = MeasuredSpeed();
                var n = real.HasValue ? SpeedText(real.Value) : $"{estimate:0.0}";
                var u = SpeedUnit(gpus, real.HasValue ? T("実測", "measured") : T("見込み", "estimate"));
                if (number.Text != n) number.Text = n;
                if (unit.Text != u) unit.Text = u;
            };
            update();
            live.Add(update);
        }

        /// <summary>The speed the monitor measured on the running model's answers, if it has one.</summary>
        double? MeasuredSpeed()
        {
            var m = monitor.Last;
            if (m == null || !State.Flag(m, "running") || !m.TryGetValue("speed", out var v) || v == null) return null;
            return State.Num(m, "speed");
        }

        /// <summary>The model's state now, its estimate beside a measured speed, and the part that takes the longest, from the monitor.</summary>
        (string, bool) MonitorText()
        {
            var m = monitor.Last;
            if (m == null) return (T("実際の速さを調べています…", "Reading the real speed…"), false);
            if (!State.Flag(m, "running")) return (T("モニタがこのモデルを見つけられません", "The monitor does not see this model"), false);
            string text;
            var now = State.Flag(m, "busy") ? T("回答中", "answering") : T("待機中", "idle");
            if (MeasuredSpeed().HasValue)
                text = T($"いま {now} ・ 見込みは {State.Num(m, "estimate"):0.0} トークン/秒", $"Now {now} · the estimate was {State.Num(m, "estimate"):0.0} tokens/s");
            else if (m.ContainsKey("ready") && !State.Flag(m, "ready"))
                text = T("ウォームアップ中です", "Warming up");
            else if (State.Flag(m, "busy"))
                text = T("最初の質問に回答中です。実際の速さは回答のあとに出ます", "Answering its first question; the real speed shows after the answer");
            else
                text = T("実際の速さは最初の回答のあとに出ます", "The real speed shows after the first answer");
            var slowest = State.Str(m, "slowest");
            if (slowest.Length > 0)
            {
                text += T($"\nいちばん時間がかかる部分: {(slowest == "CPU" ? "CPU" : ShortGpu(slowest))}",
                          $"\nSlowest part: {(slowest == "CPU" ? "CPU" : ShortGpu(slowest))}");
                if (State.Flag(m, "slowest_cpu_experts"))
                    text += T("（その層のエキスパートを CPU が計算する時間を含む）", " (with the time the CPU takes for its layers' experts)");
            }
            var slow = State.Num(m, "speed") > 0 && State.Num(m, "estimate") > 0 && State.Num(m, "speed") < 0.8 * State.Num(m, "estimate");
            return (text, slow);
        }

        /// <summary>A measured speed: slow models (a few tokens/s) need the second decimal to tell runs apart.</summary>
        static string SpeedText(double speed) => speed.ToString(speed < 10 ? "0.00" : "0.0");

        /// <summary>A card's memory, heat, clock and load now, from the monitor; red when the driver slowed it down.</summary>
        (string, bool) GpuNow(int index)
        {
            var g = monitor.Gpu(index);
            if (g == null) return (T("いまの状態を調べています…", "Reading it…"), false);
            // Two lines of a fixed shape, so that the cards keep the same height whatever the numbers.
            var text = T($"いま {State.Num(g, "used_gib"):0.0} / {State.Num(g, "total_gib"):0.0} GB ・ {State.Num(g, "temp_c"):0} ℃\n{State.Num(g, "clock_mhz"):0} MHz ・ 負荷 {State.Num(g, "load"):0}%",
                         $"Now {State.Num(g, "used_gib"):0.0} / {State.Num(g, "total_gib"):0.0} GB · {State.Num(g, "temp_c"):0} °C\n{State.Num(g, "clock_mhz"):0} MHz · load {State.Num(g, "load"):0}%");
            var down = State.Flag(g, "clock_down");
            if (down) text += T("\nクロック低下（省電力）", "\nClocks lowered (power saving)");
            if (State.Flag(g, "hot")) text += T("\n温度が高めです", "\nRunning hot");
            return (text, down || State.Flag(g, "hot"));
        }

        /// <summary>nvidia-smi's number of a card of this PC, from the name the engine gives it ("GPU1 NVIDIA ..."), else -1.</summary>
        static int GpuIndex(Dictionary<string, object> g)
        {
            var name = State.Str(g, "name");
            if (!State.Flag(g, "here") || !name.StartsWith("GPU", StringComparison.Ordinal)) return -1;
            var end = name.IndexOf(' ');
            return end > 3 && int.TryParse(name.Substring(3, end - 3), out var i) ? i : -1;
        }

        /// <summary>One GPU as a card: where it is, its name, and what the model will use of its memory.</summary>
        Card GpuCard(Dictionary<string, object> g, Color color, double? used, int cardW, bool showPc)
        {
            var c = Card(band: color);
            c.MinimumSize = new Size(cardW, 0);
            c.MaximumSize = new Size(cardW, 0);
            if (showPc)
                c.Controls.Add(Label(State.Flag(g, "here") ? T("この PC", "This PC") : T($"別の PC（{State.Str(g, "pc")}）", $"Other PC ({State.Str(g, "pc")})"), Small, Muted));
            c.Controls.Add(Label(ShortGpu(State.Str(g, "name")), H2));
            var vram = State.Num(g, "vram_gib");
            if (used.HasValue && !worker.Running)
            {
                c.Controls.Add(new Meter(vram > 0 ? used.Value / vram : 0, color, color, cardW - Px(46), 6));
                c.Controls.Add(Label(showPc ? T($"{used.Value:0.#} / {Gib(vram)} 使用（予定）", $"{used.Value:0.#} / {Gib(vram)} planned")
                    : $"{used.Value:0.#} / {Gib(vram)}", Small, Muted));
            }
            else
                c.Controls.Add(Label(T($"メモリ {Gib(vram)}", $"{Gib(vram)} of memory"), Small, Muted));
            var index = GpuIndex(g);
            if (server.State == RunState.Ready && index >= 0)
                LiveText(c, () => GpuNow(index), cardW - Px(46), Small, Ink);
            return c;
        }

        /// <summary>The PCs that hold a stretch of layers, in order, with the first and last layer of each stretch.</summary>
        static List<(string pc, double first, double last, double gib)> PcRuns(GpuMap map, List<Dictionary<string, object>> stages, List<int> parts)
        {
            var runs = new List<(string pc, double first, double last, double gib)>();
            foreach (var k in parts)
            {
                var pc = map.Pc(map.Stage[k]);
                var first = State.Num(stages[k], "first");
                var last = State.Num(stages[k], "last");
                var gib = State.Num(stages[k], "used_gib");
                if (runs.Count > 0 && runs[runs.Count - 1].pc == pc)
                {
                    var r = runs[runs.Count - 1];
                    runs[runs.Count - 1] = (pc, r.first, last, r.gib + gib);
                }
                else
                    runs.Add((pc, first, last, gib));
            }
            return runs;
        }

        /// <summary>PCs whose rows on the home page are open to show their GPUs.</summary>
        readonly HashSet<string> openPcs = new HashSet<string>();

        /// <summary>Opens a PC's row on the home page (<c>--open-pc NAME</c>, for <c>--capture</c>).</summary>
        public void OpenPc(string pc) => openPcs.Add(pc);

        /// <summary>
        /// Many GPUs: one row per PC (its color, where it is, its GPUs, their memory side by side), in the order
        /// of the model's split; a row opens to show that PC's GPUs as cards.
        /// </summary>
        void PcList(Control col, GpuMap map, Dictionary<int, double> useOf, int w)
        {
            var gpuList = state!.Gpus.ToList();
            var head = Row(Label(T("使える GPU", "GPUs"), H2),
                Label(T($"PC {map.Pcs.Count} 台 ・ GPU {gpuList.Count} 枚", $"{map.Pcs.Count} PCs · {gpuList.Count} GPUs"), Body, Muted));
            head.WrapContents = false;
            head.Margin = new Padding(0, Px(14), 0, Px(10));
            head.Controls[1].Margin = new Padding(Px(12), Px(5), 0, 0);
            col.Controls.Add(head);

            var list = Card();
            list.Padding = new Padding(0, Px(6), 0, Px(6));
            list.MinimumSize = new Size(w, 0);
            list.MaximumSize = new Size(w, 0);
            int nameW = Px(220), gpuW = Px(200), meterW = Math.Max(Px(120), w - nameW - gpuW - Px(130));
            foreach (var (pc, cards) in map.Pcs)
            {
                var color = map.PcColor(pc);
                var open = openPcs.Contains(pc);
                var here = cards.Any(i => State.Flag(gpuList[i], "here"));
                var row = new PcRow(color, open) { Width = w - Px(2), Height = Px(62) };
                row.Click += (s, e) => { if (!openPcs.Remove(pc)) openPcs.Add(pc); Render(); };

                var name = Label(PcName(pc), Bold);
                name.Location = new Point(Px(22), Px(12));
                name.MaximumSize = new Size(nameW - Px(10), Px(22));
                var count = T($"GPU {cards.Count} 枚", $"{cards.Count} GPU{(cards.Count > 1 ? "s" : "")}");
                var where = Label(here ? count : T("別の PC ・ ", "Other PC · ") + count, Small, Muted);
                where.Location = new Point(Px(22), Px(34));
                var kinds = cards.GroupBy(i => ShortGpu(State.Str(gpuList[i], "name")))
                    .Select(gr => gr.Count() > 1 ? $"{gr.Key} ×{gr.Count()}" : gr.Key);
                var gpusText = Label(string.Join(" ・ ", kinds), Small, Ink);
                gpusText.MaximumSize = new Size(gpuW - Px(10), Px(40));
                gpusText.Location = new Point(nameW, Px(22));
                var mem = cards.Select(i => (State.Num(gpuList[i], "vram_gib"), useOf.TryGetValue(i, out var u) ? u : 0.0)).ToArray();
                var meter = new MultiMeter(mem, color, meterW) { Location = new Point(nameW + gpuW, Px(16)) };
                var used = mem.Sum(m => m.Item2);
                var total = mem.Sum(m => m.Item1);
                var memText = Label(used > 0 ? T($"{used:0} / {Gib(total)} 使用（予定）", $"{used:0} / {Gib(total)} planned") : T($"メモリ {Gib(total)}", $"{Gib(total)} of memory"), Small, Muted);
                memText.Location = new Point(nameW + gpuW, Px(32));
                var arrow = Label(open ? "▾" : "▸", Body, Muted);
                arrow.Location = new Point(w - Px(40), Px(20));
                foreach (var c in new Control[] { name, where, gpusText, meter, memText, arrow })
                {
                    c.Click += (s, e) => row.PerformClick();
                    c.Cursor = Cursors.Hand;
                    row.Controls.Add(c);
                }
                list.Controls.Add(row);

                if (open)
                {
                    var detail = new FlowLayoutPanel
                    {
                        AutoSize = true, AutoSizeMode = AutoSizeMode.GrowAndShrink, BackColor = Color.Transparent,
                        MaximumSize = new Size(w - Px(20), 0), Margin = new Padding(Px(20), Px(6), 0, Px(6)),
                    };
                    var cardW = Math.Max(Px(180), (w - Px(40)) / 4 - Px(14));
                    foreach (var i in cards)
                        detail.Controls.Add(GpuCard(gpuList[i], color, useOf.TryGetValue(i, out var u) ? u : (double?)null, cardW, showPc: false));
                    list.Controls.Add(detail);
                }
            }
            col.Controls.Add(list);
        }

        /// <summary>"13.0 tokens/s · 3 GPUs": the speed in large figures.</summary>
        static FlowLayoutPanel Speed(double speed, int gpus, bool estimate = false)
        {
            var row = Row(
                new Label { Text = $"{speed:0.0}", Font = Huge, ForeColor = Ink, AutoSize = true, Margin = new Padding(0, 0, Px(6), 0), UseMnemonic = false },
                new Label
                {
                    Text = SpeedUnit(gpus, estimate ? T("見込み", "estimate") : null),
                    Font = Body, ForeColor = Muted, AutoSize = true, Margin = new Padding(0, Px(22), 0, 0), UseMnemonic = false,
                });
            row.WrapContents = false;
            row.Margin = new Padding(0, 0, 0, Px(4));
            return row;
        }

        /// <summary>"tokens/s (estimate) · on 3 GPUs": what stands beside the large figures.</summary>
        static string SpeedUnit(int gpus, string? kind) =>
            (kind != null ? T($"トークン/秒（{kind}）", $"tokens/s ({kind})") : T("トークン/秒", "tokens/s")) +
            (gpus > 0 ? T($" ・ GPU {gpus} 枚で動作", $" · on {gpus} GPU{(gpus > 1 ? "s" : "")}") : "");

        /// <summary>Where models are downloaded, with the free space there, and a button to choose another folder.</summary>
        void ModelsFolder(Control col, int w)
        {
            var card = Card();
            card.MinimumSize = new Size(w, 0);
            card.MaximumSize = new Size(w, 0);
            card.Margin = new Padding(0, Px(10), 0, 0);
            var folder = state!.ModelsDir;
            var chosen = Settings.ModelsDir != null;
            card.Controls.Add(Label(T("モデルの保存先", "Where models are saved"), Bold));
            card.Controls.Add(Wrap(folder, w - Px(48), Body, Ink));
            var free = FreeSpace(folder);
            card.Controls.Add(Label((free != null ? T($"空き {Gib(free.Value)}", $"{Gib(free.Value)} free") : T("フォルダがまだありません（最初のダウンロードで作ります）", "The folder does not exist yet (the first download makes it)"))
                + (chosen ? "" : T("　・ 既定の場所（このアプリのフォルダの data\\models）", " · the default (data\\models in this app's folder)")), Small, Muted));
            var busy = server.Running || worker.Running || (pull != null && pull.Running);
            var change = Button(T("変更…", "Change…"), false, (s, e) => ChooseModelsFolder());
            change.Enabled = !busy;
            var row = Row(change);
            if (chosen)
            {
                var reset = Button(T("既定に戻す", "Use the default"), false, (s, e) => { Settings.ModelsDir = null; ReadState(); });
                reset.Enabled = !busy;
                row.Controls.Add(reset);
            }
            card.Controls.Add(row);
            card.Controls.Add(Wrap(busy ? T("モデルの起動中やダウンロード中は変えられません。", "It cannot change while a model runs or downloads.")
                : T("変えても、ダウンロード済みのモデルは移動しません（前の保存先に残ります）。", "Models already downloaded are not moved; they stay in the old folder."), w - Px(48), Small, Muted));
            col.Controls.Add(card);
        }

        static double? FreeSpace(string folder)
        {
            try
            {
                if (!System.IO.Directory.Exists(folder)) return null;
                var root = System.IO.Path.GetPathRoot(System.IO.Path.GetFullPath(folder));
                return new System.IO.DriveInfo(root).AvailableFreeSpace / (1024.0 * 1024 * 1024);
            }
            catch (Exception) { return null; }
        }

        void ChooseModelsFolder()
        {
            using (var dialog = new FolderBrowserDialog
            {
                Description = T("モデルを保存するフォルダを選んでください。大きいモデルは 60 GB ほどあります。英数字だけのパスがおすすめです。",
                                "Choose the folder to save models in. Large models take about 60 GB. A path of plain letters and digits is best."),
                ShowNewFolderButton = true,
                SelectedPath = state?.ModelsDir ?? "",
            })
            {
                if (dialog.ShowDialog(this) != DialogResult.OK || string.IsNullOrEmpty(dialog.SelectedPath)) return;
                var path = dialog.SelectedPath;
                if (path.Any(c => c > 127) || path.Contains(' '))
                {
                    var go = MessageBox.Show(this, T("このパスには日本語や空白が入っています。llama.cpp が読めないことがあるため、英数字だけのフォルダをおすすめします。このまま使いますか？",
                                                    "This path has non-ASCII characters or spaces, which llama.cpp may fail to read. A plain path is recommended. Use it anyway?"),
                        "GPUEnsemble", MessageBoxButtons.YesNo, MessageBoxIcon.Warning);
                    if (go != DialogResult.Yes) return;
                }
                Settings.ModelsDir = path;
                AppendLog("[app] models folder: " + path);
                ReadState();
            }
        }

        void ModelsPage(FlowLayoutPanel col)
        {
            Heading(col, T("モデル", "Models"));
            var w = Width0;
            col.Controls.Add(Wrap(T("この PC と、つながっている PC の GPU で動かせるモデルです。上ほどおすすめです。速さは見込みです。",
                                    "Models for the GPUs of this PC and of the PCs it is linked with, best first. Speeds are estimates."), w, Body, Muted));
            Reading(col);
            if (state == null) return;
            ModelsFolder(col, w);
            var update = Row(Button(T("リストを新しくする", "Update the list"), false, (s, e) => UpdateCatalog()),
                             Button(T("モデルを追加…", "Add a model..."), false, (s, e) => AddModel()));
            update.Controls[1].Enabled = !addingModel;
            if (updatingCatalog)
            {
                update.Controls[0].Enabled = false;
                var busyNote = Label(CatalogNoteText(), Small, Muted);
                catalogNote = busyNote;
                busyNote.Margin = new Padding(Px(10), Px(13), 0, 0);
                update.Controls.Add(busyNote);
            }
            if (addingModel)
            {
                var addNote = Label(T("追加するモデルのファイルを読んでいます…", "Reading the file of the model to add..."), Small, Muted);
                addNote.Margin = new Padding(Px(10), Px(13), 0, 0);
                update.Controls.Add(addNote);
            }
            col.Controls.Add(update);
            var list = Card();
            list.MinimumSize = new Size(w, 0);
            list.MaximumSize = new Size(w, 0);
            list.Margin = new Padding(0, Px(10), 0, Px(10));
            var table = new TableLayoutPanel
            {
                ColumnCount = 4,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                BackColor = Surface,
                Width = w - Px(46),
                MinimumSize = new Size(w - Px(46), 0),
                Margin = new Padding(0),
            };
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, Px(130)));
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, Px(90)));
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, Px(260)));
            table.CellPaint += (s, e) =>
            {
                if (e.Row > 0) using (var p = new Pen(Line)) e.Graphics.DrawLine(p, e.CellBounds.Left, e.CellBounds.Top, e.CellBounds.Right, e.CellBounds.Top);
            };
            void Head(string t) => table.Controls.Add(Label(t, Small, Muted));
            Head(T("モデル", "Model")); Head(T("速さ（見込み）", "Speed (est.)")); Head(T("大きさ", "Size")); Head("");
            var busy = server.Running || worker.Running;
            // "The higher, the better": the models that run here first, each part in the list's order.
            foreach (var m in state.Models.OrderBy(x => State.Flag(x, "fits") ? 0 : 1))
            {
                var id = State.Str(m, "id");
                var fits = State.Flag(m, "fits");
                var name = Column(0);
                name.Margin = new Padding(0, Px(12), 0, Px(12));
                name.Controls.Add(Label(State.Str(m, "name"), Bold, fits ? Ink : Muted));
                if (id == state.Best) name.Controls.Add(Label(T("おすすめ", "Recommended"), Small, Purple));
                if (State.Flag(m, "new"))
                    name.Controls.Add(Label(T("新着・未検証（Hugging Face で人気の新しいモデル）", "New, not checked (popular on Hugging Face)"), Small, Gold));
                if (State.Flag(m, "mine"))
                {
                    var repo = State.Str(m, "hf_repo");
                    name.Controls.Add(Label(T("自分で追加・未検証", "Added by you, not checked"), Small, DeepBlue));
                    name.Controls.Add(Label(repo, Small, Muted));
                    var off = new LinkLabel
                    {
                        Text = T("一覧から外す", "Take off the list"),
                        Font = Small,
                        AutoSize = true,
                        LinkColor = Muted,
                        ActiveLinkColor = Ink,
                        BackColor = Color.Transparent,
                        Margin = new Padding(0, 0, 0, Px(4)),
                        Enabled = !(pull != null && pull.Id == id && pull.Running),
                    };
                    var shown = State.Str(m, "name");
                    off.LinkClicked += (s, e) => RemoveModel(id, shown);
                    name.Controls.Add(off);
                }
                name.Controls.Add(Wrap(fits ? Placement(m) : T("この PC には入りません", "Does not fit on this PC"), w - Px(560), Small, Muted));
                table.Controls.Add(name);
                var sp = Label(fits ? T($"{State.Num(m, "speed"):0.#} トークン/秒", $"{State.Num(m, "speed"):0.#} tokens/s") : "-", Body, fits ? Ink : Muted);
                sp.Margin = new Padding(0, Px(14), 0, 0);
                table.Controls.Add(sp);
                var size = Label(Gib(State.Num(m, "file_gib")), Body, fits ? Ink : Muted);
                size.Margin = new Padding(0, Px(14), 0, 0);
                table.Controls.Add(size);
                var act = Row();
                act.WrapContents = false;
                act.Margin = new Padding(0, Px(8), 0, Px(8));
                if (pull != null && pull.Id == id && pull.Running)
                {
                    var box = Column(0);
                    box.Margin = new Padding(0, 0, Px(10), 0);
                    var p = pull!;
                    LiveProgress(box, () => p.Total > 0 ? p.Done / p.Total : 0,
                        () => p.Total > 0 ? T($"{p.Done / p.Total:0%}　{TimeLeft(p.MinutesLeft)}", $"{p.Done / p.Total:0%}, {TimeLeft(p.MinutesLeft)}") : T("準備しています", "Preparing"), Px(130));
                    act.Controls.Add(box);
                    act.Controls.Add(Button(T("止める", "Stop"), false, (s, e) => pull?.Stop()));
                }
                else if (fits && State.Flag(m, "downloaded"))
                {
                    var done = Label(T("ダウンロード済み", "Downloaded"), Small, Good);
                    done.Margin = new Padding(0, Px(13), Px(10), 0);
                    act.Controls.Add(done);
                    var b = Button(T("起動", "Start"), true, (s, e) => StartServer(id));
                    b.Enabled = !busy;
                    act.Controls.Add(b);
                }
                else if (fits)
                {
                    var b = Button(T("ダウンロード", "Download"), false, (s, e) => StartPull(id));
                    b.Enabled = pull == null || !pull.Running;
                    act.Controls.Add(b);
                }
                table.Controls.Add(act);
            }
            list.Controls.Add(table);
            col.Controls.Add(list);
            col.Controls.Add(Label(T($"保存先: {state.ModelsDir}", $"Saved in {state.ModelsDir}"), Small, Muted));
        }

        /// <summary>How the model is placed, in plain words: "GPU 3 枚 ・ 一部の層は CPU".</summary>
        static string Placement(Dictionary<string, object> m)
        {
            if (!(m.TryGetValue("layout", out var v) && v is Dictionary<string, object> l)) return State.Str(m, "placement");
            var parts = new List<string>();
            var gpus = (int)State.Num(l, "gpus");
            parts.Add(T($"GPU {gpus} 枚", $"{gpus} GPU{(gpus > 1 ? "s" : "")}"));
            var cpu = (int)State.Num(l, "cpu_layers");
            if (cpu > 0) parts.Add(T($"{cpu} 層は CPU", $"{cpu} layers on the CPU"));
            if (l.TryGetValue("experts_on_gpu", out var e) && e != null)
            {
                var f = State.Num(l, "experts_on_gpu");
                parts.Add(f >= 0.995 ? T("エキスパートはすべて GPU", "all experts on the GPUs") : T($"エキスパートの {f:0%} を GPU、残りは CPU", $"{f:0%} of the experts on the GPUs, the rest on the CPU"));
            }
            if (State.Str(l, "kv") != "f16" && State.Str(l, "kv").Length > 0) parts.Add(T("会話の記憶を圧縮", "compressed chat memory"));
            if (State.Flag(l, "there")) parts.Add(T("別の PC で動かします", "runs on the other PC"));
            else if (l.TryGetValue("faster_there", out var t) && t != null)
                parts.Add(T($"別の PC だけで動かすと約 {State.Num(l, "faster_there"):0.#} トークン/秒", $"about {State.Num(l, "faster_there"):0.#} tokens/s on the other PC alone"));
            return string.Join(T(" ・ ", " · "), parts);
        }

        void LinkPage(FlowLayoutPanel col)
        {
            Heading(col, T("つながる PC", "Other PCs"));
            var w = Width0;
            var inner = w - Px(48);

            var role = Card();
            role.MinimumSize = new Size(w, 0);
            role.MaximumSize = new Size(w, 0);
            role.Controls.Add(Label(T("この PC の役割", "What this PC does"), H2));
            role.Controls.Add(Wrap(T("メイン: モデルを動かし、ほかの PC の GPU も使います。GPU を貸す: この PC の GPU をメインの PC に使わせます。",
                                     "Main: runs the model and uses the GPUs of other PCs too. Lend GPUs: lets a main PC use this PC's GPUs."), inner, Body, Muted));
            var main = Button(T("メイン", "Main"), !worker.Running, (s, e) => { worker.Stop(); ReadState(); });
            var lend = Button(T("GPU を貸す", "Lend GPUs"), worker.Running, (s, e) => { if (!server.Running) worker.Start(); });
            lend.Enabled = !server.Running || worker.Running;
            role.Controls.Add(Row(main, lend));
            if (server.Running && !worker.Running)
                role.Controls.Add(Label(T("モデルを動かしている間は貸せません。", "Stop the model first to lend the GPUs."), Small, Muted));
            if (worker.Running)
                role.Controls.Add(Status(worker.State == RunState.Ready ? T("貸しています。メインの PC から見つけられます。", "Lending: main PCs on the network can find this PC.") : T("準備しています…", "Getting ready…"), Good));
            else if (worker.State == RunState.Failed && worker.LastError.Length > 0)
                role.Controls.Add(Wrap(worker.LastError, inner, Small, Bad));
            col.Controls.Add(role);

            foreach (var a in worker.Asking)
            {
                var ask = Card(Notice, NoticeLine);
                ask.MinimumSize = new Size(w, 0);
                ask.MaximumSize = new Size(w, 0);
                ask.Controls.Add(Label(T("ほかの PC がこの PC の GPU を使おうとしています", "Another PC asks to use this PC's GPUs"), Bold));
                ask.Controls.Add(Label($"{a.Key}  ({a.Value})", Body, Muted));
                var address = a.Value;
                ask.Controls.Add(Row(
                    Button(T("許可する", "Allow"), true, (s, e) => worker.Allow(address)),
                    Button(T("断る", "Refuse"), false, (s, e) => worker.Forget(address))));
                ask.Controls.Add(Label(T("許可した PC は覚えておき、次からは聞きません。", "An allowed PC is remembered and not asked about again."), Small, Muted));
                col.Controls.Add(ask);
            }

            if (worker.Running) return;
            Section(col, T("GPU を貸している PC", "PCs that lend their GPUs"));
            Reading(col);
            if (state == null) return;
            var list = new FlowLayoutPanel { AutoSize = true, AutoSizeMode = AutoSizeMode.GrowAndShrink, MaximumSize = new Size(w + Px(14), 0), Margin = new Padding(0), BackColor = Color.Transparent };
            var cardW = Px(310);
            var id = CurrentId();
            var map = new GpuMap(state, Stages(id != null ? state.Model(id) : null));
            foreach (var wk in state.Workers)
            {
                var c = Card();
                c.MinimumSize = new Size(cardW, 0);
                c.MaximumSize = new Size(cardW, 0);
                c.Controls.Add(Label(State.Str(wk, "name"), H2));
                c.Controls.Add(Label(State.Str(wk, "endpoint"), Small, Muted));
                // This worker's cards in the GPU list (same endpoint), in order: their colors.
                var endpoint = State.Str(wk, "endpoint");
                var mine = state.Gpus.Select((g, i) => (g, i)).Where(x => State.Str(x.g, "endpoint") == endpoint).Select(x => x.i).ToList();
                var k = 0;
                foreach (var g in (wk.TryGetValue("gpus", out var gv) && gv is object[] arr ? arr : new object[0]).OfType<Dictionary<string, object>>())
                {
                    var color = k < mine.Count ? map.Color(mine[k]) : Idle;
                    k++;
                    var line = Row(new Dot(color) { Margin = new Padding(0, Px(5), Px(6), 0) }, Label($"{ShortGpu(State.Str(g, "name"))}  {Gib(State.Num(g, "vram_gib"))}", Body));
                    line.WrapContents = false;
                    line.Margin = new Padding(0, Px(2), 0, 0);
                    c.Controls.Add(line);
                }
                if (State.Flag(wk, "this_pc"))
                    c.Controls.Add(Label(T("この PC 自身（直接使います）", "This PC itself (used directly)"), Small, Muted));
                else if (State.Flag(wk, "allowed"))
                    c.Controls.Add(Status(T("つながっています", "Linked"), Good));
                else
                    c.Controls.Add(Wrap(T("まだ許可されていません。その PC のアプリで「許可する」を押してから、もう一度さがしてください。",
                                          "Not allowed yet: press Allow on that PC, then look again."), cardW - Px(46), Small, Bad));
                list.Controls.Add(c);
            }
            var add = Card(border: Line2);
            add.MinimumSize = new Size(cardW, 0);
            add.MaximumSize = new Size(cardW, 0);
            add.Controls.Add(Label(T("ほかの PC を足す", "Add another PC"), H2));
            add.Controls.Add(Wrap(T("その PC にもこのアプリを入れて「GPU を貸す」を押すと、ここに出てきます。別の場所の PC は、両方を Tailscale の同じアカウントにつなぎます。",
                                    "Install this app there too and press Lend GPUs; it shows up here. For a PC at another place, sign both PCs in to the same Tailscale account."), cardW - Px(46), Small, Muted));
            add.Controls.Add(Row(Button(T("もう一度さがす", "Look again"), false, (s, e) => ReadState())));
            list.Controls.Add(add);
            col.Controls.Add(list);
        }

        // --- window and tray ---------------------------------------------------------------------

        public void BringBack() => ShowWindow();

        void ShowWindow()
        {
            Show();
            if (WindowState == FormWindowState.Minimized) WindowState = FormWindowState.Normal;
            Activate();
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            if (!exiting && e.CloseReason == CloseReason.UserClosing && (server.Running || worker.Running || (pull?.Running ?? false)))
            {
                // Closing the window keeps the model (or the lending) running; the icon's menu ends it.
                e.Cancel = true;
                Hide();
                if (!told)
                {
                    told = true;
                    tray.ShowBalloonTip(5000, "GPUEnsemble",
                        T("動いたままです。止めるときは、通知領域のアイコンから「終了」を選びます。",
                          "Still running. To stop it, choose Exit from the icon in the notification area."), ToolTipIcon.Info);
                }
                return;
            }
            exiting = true;
            server.Stop();
            worker.Stop();
            pull?.Stop();
            tray.Visible = false;
            base.OnFormClosing(e);
        }

        void ExitApp()
        {
            exiting = true;
            Close();
        }
    }

    sealed class DoubleBufferedPanel : Panel
    {
        public DoubleBufferedPanel() { DoubleBuffered = true; }
    }

    /// <summary>A menu entry: a soft rounded box with a gradient dot when selected.</summary>
    sealed class NavItem : Control
    {
        bool selected, hover;

        public NavItem(string text)
        {
            Text = text;
            Height = Px(44);
            Margin = new Padding(0, 0, 0, Px(4));
            Cursor = Cursors.Hand;
            TabStop = true;
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint | ControlStyles.Selectable, true);
            AccessibleRole = AccessibleRole.PageTab;
            AccessibleName = text;
        }

        public bool Selected { get => selected; set { selected = value; Invalidate(); } }

        protected override void OnMouseEnter(EventArgs e) { hover = true; Invalidate(); base.OnMouseEnter(e); }
        protected override void OnMouseLeave(EventArgs e) { hover = false; Invalidate(); base.OnMouseLeave(e); }
        protected override void OnGotFocus(EventArgs e) { Invalidate(); base.OnGotFocus(e); }
        protected override void OnLostFocus(EventArgs e) { Invalidate(); base.OnLostFocus(e); }

        protected override void OnKeyDown(KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter || e.KeyCode == Keys.Space) { OnClick(EventArgs.Empty); e.Handled = true; }
            base.OnKeyDown(e);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            g.Clear(Surface);
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            if (selected || hover)
                using (var path = Round(new RectangleF(0, 0, Width - 1, Height - 1), Px(10)))
                using (var b = new SolidBrush(selected ? Soft : Hex("#F6F6FB"))) g.FillPath(b, path);
            var d = Px(8);
            var dot = new RectangleF(Px(14), (Height - d) / 2f, d, d);
            if (selected)
                using (var b = new SolidBrush(Purple)) g.FillEllipse(b, dot);
            else
                using (var p = new Pen(Muted)) g.DrawEllipse(p, dot);
            var r = new Rectangle(Px(34), 0, Width - Px(34), Height);
            TextRenderer.DrawText(g, Text, selected ? Bold : Body, r, Ink, TextFormatFlags.VerticalCenter | TextFormatFlags.Left | TextFormatFlags.NoPrefix);
            if (Focused && ShowFocusCues)
                using (var p = new Pen(Gold, Px(2)))
                using (var path = Round(new RectangleF(1, 1, Width - 3, Height - 3), Px(10))) g.DrawPath(p, path);
        }
    }
}
