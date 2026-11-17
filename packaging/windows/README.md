# Windows zip

`build_zip.py` builds two zips that differ only in the window app:
`dist/GPUEnsemble-<version>-windows-x64-app.zip` (with `GPUEnsemble.exe`, and a README.txt that starts
with it) and `dist/GPUEnsemble-<version>-windows-x64-cli.zip` (launchers and command line only). Without
the .NET SDK only the `-cli` zip is made (`--require-app` makes that an error, as releases want).

| Path in the zip | What it is |
|---|---|
| `GPUEnsemble.exe` | `-app` zip only. The Windows app (see [app/windows](../../app/windows/README.md)): GPUs, models, start and stop, other PCs, in a window with an icon in the notification area. Built with the .NET SDK when it is installed (`--require-app` makes that a must, `--app-exe` gives a built one) |
| `Start GPUEnsemble.cmd` | Double-click: opens a "GPUEnsemble" window running `up --open --remote auto` (pick, download and start the best model for this PC's GPUs and those that workers on the LAN lend it, then open the chat page) through `gpuensemble_planner.launch`, which waits for Enter at the end so errors stay readable. It uses `start`, so no batch file runs in that window and Ctrl+C does not get cmd.exe's "Terminate batch job (Y/N)?" |
| `Start GPUEnsemble worker.cmd` | Double-click on a PC that lends its GPUs: opens a "GPUEnsemble worker" window running `worker --lan --cache --serve` through `gpuensemble_planner.launch worker` (with `--serve` it runs a model that fits on its own GPUs itself when a main PC asks; the received weights stay in `data\rpc-cache`, so later loads send only their hashes; for a main PC at another site it downloads its share of the weights from Hugging Face itself the first time). Main PCs on the LAN find it; typing `y` there lets one in. The launcher reads the typed lines in one thread and hands them to the closing prompt when the worker stops, so a single Enter closes the window |
| `gpuensemble.cmd` | Command line: `gpuensemble.cmd recommend`, `gpuensemble.cmd up --model qwen3-8b`, ... |
| `python\` | python.org's embeddable Python 3.12.10 (64-bit), checked against its SHA-256 |
| `app\gpuensemble_planner\` | the planner package, on Python's search path through `python312._pth` |
| `examples\` | rig files and model profiles |
| `data\` | created on first use: models and llama.cpp (`GPUENSEMBLE_HOME` moves it; `GPUENSEMBLE_MODELS`, or the app's Models page, moves only the models) |

Nothing is installed or written outside the folder, so removing the folder removes everything.

```
python packaging/windows/build_zip.py                 # downloads the embeddable Python once
python packaging/windows/build_zip.py --python-zip python-3.12.10-embeddable-amd64.zip
python -m unittest discover -s packaging/windows -p "test_*.py"
```

The [windows-zip workflow](../../.github/workflows/windows-zip.yml) runs the tests, builds the zip on
a clean Windows runner, starts `--help`, `models` and `recommend` from the unpacked zip, and keeps the
zip as the run's artifact. It does not run on pull requests or pushes; it is started by hand
(workflow_dispatch) when the zip has to be checked, for example before a release, and by the release
workflow for a tag. Everyday tests, including these build script tests, run with `python -m unittest`.
