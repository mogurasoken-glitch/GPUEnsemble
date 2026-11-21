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
- The Other PCs page sets this PC's role, kept in `data\app-settings.json` (`role`): the app asks for it on its first start (Home page) and lends nothing before that. Auto runs
  `worker --lan --cache --serve --trust-lan` while no model runs here, stops it to start a model here (not while
  another PC uses the GPUs: the gate's "is using these GPUs" lines), and starts it again when the model ends;
  Main never lends; Lend GPUs only lends. "Link PCs on the home LAN without asking" (`trust_lan`, on by default)
  drops `--trust-lan`. "Open this app when I sign in" writes `GPUEnsemble.exe --tray` under the user's
  `HKCU\...\CurrentVersion\Run` key; `--tray` starts with the window hidden. The page also shows the PCs let in
  without asking (Refuse = `deny ADDRESS`) and the worker's notes on why other PCs cannot reach it.
- Home has a Boost switch under Start (`boost` in `data\app-settings.json`, off by default): when on, `up` gets
  `--boost`, which locks this PC's cards' clocks while the model runs (Windows asks for administrator approval at
  each start). Home shows "Boost: on" once the engine says `boost: on`, or the engine's `note: --boost:` reason
  when the clocks were not locked.
- Colors come from `Palette.cs`: `charcoal` (warm near-black, smoky gold buttons; the default, with Windows' dark
  title bar) or `light` (the earlier near-white look), picked by `theme` in `data\app-settings.json` or
  `--theme charcoal|light` for one start (pictures of both).
- Errors in the window are written to `data\app-errors.log`.

| File | What it is |
|---|---|
| `Program.cs` | one window per folder, error log |
| `Engine.cs` | finds the zip's Python, runs the engine, parses `state` and the console lines the app reacts to |
| `Services.cs` | the long-running commands (`up`, `pull`, `worker`) and their state; `monitor --json --every 2` while a model runs (real speed, each card's memory, heat, clocks, load) |
| `MainForm.cs` | the pages (Home, Models, Other PCs, Log) and the notification-area icon |
| `Palette.cs` | the two color themes (charcoal, light) |
| `Ui.cs` | fonts, the theme in use and the drawn controls (cards, buttons, bars) |
