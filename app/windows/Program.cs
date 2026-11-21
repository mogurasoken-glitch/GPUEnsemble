using System;
using System.Threading;
using System.Windows.Forms;

namespace GPUEnsembleApp
{
    static class Program
    {
        [STAThread]
        static int Main(string[] args)
        {
            // One window per folder: a second start brings the first one forward instead.
            var name = "GPUEnsemble-" + Engine.BaseDir.ToLowerInvariant().GetHashCode().ToString("x8");
            if (Array.IndexOf(args, "--state") >= 0) name += "-state";
            using (var one = new Mutex(true, name, out var first))
            using (var shown = new EventWaitHandle(false, EventResetMode.AutoReset, name + "-show"))
            {
                if (!first)
                {
                    shown.Set();
                    return 0;
                }
                // A fault in the window is written to data\app-errors.log, not shown as .NET's dialog.
                Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);
                Application.ThreadException += (_, e) => Report(e.Exception);
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                // The language chosen in the window, or --lang ja|en for this start only (pictures of both).
                var lang = Settings.Language;
                var at = Array.IndexOf(args, "--lang");
                if (at >= 0 && at + 1 < args.Length) lang = args[at + 1];
                if (lang == "ja" || lang == "en") Ui.SetLanguage(lang == "ja");
                // The colors: "theme" in app-settings.json, or --theme charcoal|light for this start only.
                var theme = Settings.Theme;
                at = Array.IndexOf(args, "--theme");
                if (at >= 0 && at + 1 < args.Length) theme = args[at + 1];
                Ui.UseTheme(theme);
                var form = new MainForm();
                // Windows' start at sign-in (Settings.OpenAtSignIn): only the icon in the notification area.
                if (Array.IndexOf(args, "--tray") >= 0) form.StartHidden = true;
                at = Array.IndexOf(args, "--capture");
                if (at >= 0 && at + 1 < args.Length) form.CaptureTo = args[at + 1];
                at = Array.IndexOf(args, "--open-pc");
                if (at >= 0 && at + 1 < args.Length) form.OpenPc(args[at + 1]);
                at = Array.IndexOf(args, "--start-model");
                if (at >= 0 && at + 1 < args.Length) form.StartModel = args[at + 1];
                at = Array.IndexOf(args, "--state");
                if (at >= 0 && at + 1 < args.Length) Engine.StateFile = args[at + 1];
                var wait = ThreadPool.RegisterWaitForSingleObject(shown, (_, __) =>
                {
                    try { form.BeginInvoke(new Action(form.BringBack)); } catch (InvalidOperationException) { }
                }, null, Timeout.Infinite, false);
                Application.Run(form);
                wait.Unregister(null);
                return 0;
            }
        }

        static void Report(Exception e)
        {
            try
            {
                System.IO.Directory.CreateDirectory(Engine.Home);
                System.IO.File.AppendAllText(System.IO.Path.Combine(Engine.Home, "app-errors.log"),
                    $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} {e}{Environment.NewLine}{Environment.NewLine}");
            }
            catch (Exception) { }
        }
    }
}
