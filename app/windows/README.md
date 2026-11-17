# GPUEnsemble.exe (Windows app)

A window over the planner: it shows this PC's GPUs and those other PCs lend, the models they can run,
and starts, stops and links them. The Windows zip carries it next to the launchers
(`packaging/windows/build_zip.py` builds it when the .NET SDK is installed, or takes `--app-exe`).

- C# WinForms on .NET Framework 4.8, which Windows 10 (1903+) and 11 already have: no runtime to install.
- Builds on Windows, Linux or macOS with the .NET SDK: `dotnet build -c Release` (the reference
  assemblies come from NuGet; the Windows Desktop SDK is not needed).
- It runs `python\python.exe -m gpuensemble_planner` from the zip: `state --remote auto` for what it
  shows (see `planner/src/gpuensemble_planner/appstate.py`, the contract between the two), and
  `up`, `pull` and `worker` as child processes, which end with the app (a job object).
- Japanese when Windows is Japanese, English otherwise; "English" / "日本語" below the menu switches it and
  keeps the choice in `data\app-settings.json` (`language`). `--lang ja|en` sets it for one start (pictures of both).
- `GPUEnsemble.exe --capture FOLDER` saves a picture of each page and exits (for checking the look).
  With `--start-model ID` it first starts that model, as the Start button does, asks the ready model one short
  question (the monitor has no speed before the first answer), and takes the pictures once the monitor has read
  the speed (`--start-model` alone just starts the model).
- Errors in the window are written to `data\app-errors.log`.

| File | What it is |
|---|---|
| `Program.cs` | one window per folder, error log |
| `Engine.cs` | finds the zip's Python, runs the engine, parses `state` and the console lines the app reacts to |
| `Services.cs` | the long-running commands (`up`, `pull`, `worker`) and their state; `monitor --json --every 2` while a model runs (real speed, each card's memory, heat, clocks, load) |
| `MainForm.cs` | the pages (Home, Models, Other PCs, Log) and the notification-area icon |
| `Ui.cs` | colors, fonts and the drawn controls (cards, buttons, bars) |
