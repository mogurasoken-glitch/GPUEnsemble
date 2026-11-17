# GPUEnsemble user guide

*[日本語](GUIDE.md)*

**Pool the GPUs you have and run a large AI model (LLM) that no single card can hold, on your own PC.**

GPUEnsemble adds up the VRAM of the GPUs in your PC so that a language model too large for any one of them still runs. The cards may differ in model and VRAM size. GPUs of another PC can be added too.

- **Starts with a double-click.** It looks at the PC's GPUs, RAM and CPU, picks the most capable model that runs well on it, downloads it, starts it and opens a chat page in the browser. Nothing else (such as Python) needs to be installed.
- **Decides the placement for you.** Which part of the model goes on which GPU, and what goes to RAM when it does not fit, is worked out from the real model file and the speed of each GPU. No hunting for llama.cpp arguments by hand.
- **Uses the GPUs of another PC too (experimental).** With a second PC that has a GPU, a double-click there adds its GPU as well. That PC can be on the same home LAN or at another place; a PC at another place is linked over a VPN.
- **Stays on your PC.** The model runs on your own PC and conversations are not sent to any outside service. Other apps can use it through an OpenAI-compatible API.

> **Status: pre-alpha.** The computation is done by llama.cpp; an execution engine of GPUEnsemble's own comes later. It has been checked on Windows with NVIDIA RTX 20-series cards.

## How fast it runs

Measured on the development PC (steady speed while generating). As a rough guide, about 5 tokens per second is the speed at which people read.

| Setup | Model | Speed |
| --- | --- | --- |
| RTX 2080 Ti + RTX 2080 SUPER (19 GB of VRAM together), 64 GB RAM | gpt-oss-120b (59 GiB) | about 6.5 tokens/s |
| The PC above plus an RTX 4090 PC at another place, over a VPN (Tailscale) | gpt-oss-120b | about 10 to 10.7 tokens/s |
| One RTX 2080 Ti | Qwen3-8B | about 80 tokens/s |

A 59 GiB model that fits on neither card runs because the cards at hand are put together. The measurements are in [benchmarks/](../benchmarks/README.md).

## What you need

| Item | Details |
| --- | --- |
| OS | Windows 10 / 11 (64-bit) |
| GPU | An NVIDIA GPU; checked with the RTX 20 series. With a driver that supports CUDA 12.4 (551.61 or later) the CUDA build is used, with an older one the Vulkan build |
| RAM | Larger models use more. The development PC, which picks gpt-oss-120b, has 64 GB |
| Disk | Free space for the model (about 5 to 59 GiB for the usual picks) and about 1 GB for llama.cpp |
| Network | To download the model and llama.cpp the first time |

Which model is picked depends on the PC. On a smaller PC it picks the best model that fits.

## Getting started (one PC)

1. Download one of the two zips from the GitHub Releases page. They differ only in whether the app (`GPUEnsemble.exe`) is included.
   - `GPUEnsemble-<version>-windows-x64-app.zip`: with the window app. Double-click `GPUEnsemble.exe` to see the GPUs, models and linked PCs, and to start and stop.
   - `GPUEnsemble-<version>-windows-x64-cli.zip`: no app; the `.cmd` files and commands below only.
2. Unpack it to a folder **with only letters and digits in its path** on a drive with plenty of free space (for example `D:\GPUEnsemble`). Avoid paths with spaces or non-ASCII characters.
3. Double-click `Start GPUEnsemble.cmd` (or `GPUEnsemble.exe` with the app zip, see "Using the app" below).
4. A console window opens, looks at the PC and decides on a model. Before downloading it asks about the size, such as "download gpt-oss-120b MXFP4 (59.0 GiB) from Hugging Face? [Y/n]"; press Enter to go ahead.
5. When the download and loading are done, the chat page (`http://127.0.0.1:8080/`) opens in the browser.

**The first start takes a while.** Large models take time to download, and every start reads the model into RAM: for gpt-oss-120b the loading alone takes about 5 minutes. If a download stops halfway, double-click again and it continues where it stopped. Downloaded files are checked against their published hashes.

**To stop**, press Ctrl+C in the window. The window stays open so that the last messages can be read; press Enter to close it. Closing the window directly also ends the llama.cpp that ran behind it.

**From then on**, the same double-click is all it takes; a model already downloaded is used as it is.

## Using the app

Double-click `GPUEnsemble.exe` in the folder of the `-app.zip` and a window opens.

- **Home:** the GPUs that can be used, the recommended model, Start and Stop, and a button that opens the chat. While a model runs it shows the real speed and each card's memory and temperature.
- **Models:** the models to choose from with their estimated speeds. Download, start, change the folder for models, and "Add a model..." (put a GGUF repository on Hugging Face on this PC's list).
- **Other PCs:** use the GPUs of other PCs (main), or lend this PC's GPUs.
- **Log:** the output of the commands that run behind the window.

After the window is closed, the icon in the notification area opens it again. The window is in Japanese or English by the language of Windows; "English" / "日本語" below the menu on the left switches it, and the choice is kept for the next start.

## Using it from other apps

While it runs, `http://127.0.0.1:8080/v1` is an OpenAI-compatible API. Point an OpenAI library or tool at that address. As the model name, give the model's id from `gpuensemble.cmd models` (for example `gpt-oss-120b`).

Qwen3 and gpt-oss think before they answer (they write reasoning tokens first), so with a small `max_tokens` the answer is cut off while it is still thinking and comes back empty. Allow plenty.

## Choosing the model yourself

Use `gpuensemble.cmd` in the unpacked folder from a Command Prompt or PowerShell.

```
gpuensemble.cmd recommend                     the models that run on this PC, with estimated speeds
gpuensemble.cmd models                        the models to choose from, and whether they are downloaded
gpuensemble.cmd up --model qwen3-8b           start a given model
gpuensemble.cmd up --model D:\models\x.gguf   use a GGUF file you have
gpuensemble.cmd --version                     the version
```

The models to choose from are 49 checked ones (from Qwen3.5-4B at 2.5 GiB up to models of several hundred GiB) and new models found on Hugging Face. `gpuensemble.cmd models` lists them and `gpuensemble.cmd catalog --update` brings the list up to date (the app does this by itself each time it starts). Which ones run depends on the PC's GPUs and RAM; `recommend` lists those that run, with estimated speeds. A GGUF model that is not on the list can be put on this PC's list with `gpuensemble.cmd catalog --add <repository on Hugging Face>`.

The commands are described in detail in [planner/README.md](../planner/README.md) (Japanese).

## Using the GPUs of another PC (experimental)

With a second PC that has a GPU, its GPU can be used as well.

### On the same home LAN

1. Unpack the same zip on the PC that lends its GPU, and double-click `Start GPUEnsemble worker.cmd`.
2. On the main PC, double-click `Start GPUEnsemble.cmd`. The main PC finds the lending PC on the LAN by itself.
3. The first time, the lending PC's window shows "PC '(main PC's name)' at (address) asks to use these GPUs". Type `y` and press Enter to lend the GPUs to that PC. Once allowed, it does not ask again.

The main PC waits up to 2 minutes for this. If Windows asks whether Python may communicate, allow **private networks only**. If the main PC does not find the other one, look at the lending PC's window: it says with `note:` why it cannot be reached, for example when the network is set to Public.

### At another place

A PC at another place is linked over a VPN first. Once both PCs are on the same VPN, it works much as on a LAN.

1. On the PC that lends its GPU, double-click `Start GPUEnsemble worker.cmd`.
2. On the main PC, start with the lending PC's VPN address (for example `gpuensemble.cmd up --remote auto:10.0.0.2`).
3. The first time, type `y` and press Enter in the lending PC's window to allow it.

The PC at another place downloads its part of the model from Hugging Face itself, so a slow upload from the main PC does not hold it up.

With Tailscale as the VPN it is simpler still: when both PCs are signed in with the same account, a double-click on `Start GPUEnsemble.cmd` on the main PC finds the lending PC and links to it without the `y`.

### Good to know

- The lending PC keeps the model parts it received, so later starts are faster.
- When talking to the other PC takes too long to gain speed, a placement without that GPU is chosen.
- When the whole model fits on the lending PC's GPU alone and runs faster there, the lending PC downloads and runs the model itself, and the main PC passes the chat page and the API on to it. Nothing changes in how it is used.
- On the development PCs, adding an RTX 4090 at another place over a VPN (Tailscale) took gpt-oss-120b from about 6.5 to about 10 tokens/s. The first start then took about 8 minutes ([measurement](../benchmarks/phase2b-two-sites-tailscale.md)).

## Security

The mechanism that lends GPUs (llama.cpp's RPC) has no password and no encryption. GPUEnsemble therefore puts a gate in front of it that lets in only the addresses of the PCs you allowed.

- Use it **only on networks you trust**: a home LAN, a cable between the two PCs, or a VPN.
- Do not expose the lending PC's ports (50052 to 50054) to the internet.
- The chat page and the API (`127.0.0.1:8080`) can be used only from within that PC.

See [SECURITY.md](../SECURITY.md) for details.

## When something goes wrong

- **The download stopped:** double-click again and it continues where it stopped. If there is not enough free space, it stops before it starts.
- **llama.cpp crashes right after it starts:** this happens with an old Visual C++ runtime. GPUEnsemble finds a newer one from other apps and fixes it by itself; if none is found, follow its message and install the Visual C++ Redistributable.
- **The other PC is not found:** see the notes on the firewall and the network type under "Using the GPUs of another PC".

## Uninstalling

The models and llama.cpp all go into `data` inside the unpacked folder. Deleting the folder leaves nothing behind.

To keep the models on another drive, press "Change…" under "Where models are saved" on the app's Models page and choose a folder (on the command line, set the environment variable `GPUENSEMBLE_MODELS` to the folder). Models already downloaded are not moved.

## License

GPUEnsemble is under the [MIT License](../LICENSE). Python in the zip, and llama.cpp and the models downloaded when it runs, are under their own licenses ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).
