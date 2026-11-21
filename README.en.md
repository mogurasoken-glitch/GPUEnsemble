# GPUEnsemble

**Pools the VRAM of several GPUs, in one PC or across PCs, so that LLMs too large for any single card run locally.**

**[日本語の README](README.md)**

> **Status: pre-alpha.** Inference runs on llama.cpp's layer split (a pipeline); GPUEnsemble is the placement planner around it: it picks the model that suits this PC, downloads it, decides which layers go where, runs llama.cpp that way, and shows the real speed and each card's state while it runs. GPUs of another PC can be added (experimental). An executor of our own comes after the MVP. The "MVP is done when" condition below is met, v0.1.0 was the first public version; v0.1.1 adds PCs without a GPU and AMD or Intel GPUs (through Vulkan).

## Try it (Windows, NVIDIA GPUs)

A step-by-step user guide is in [docs/GUIDE.en.md](docs/GUIDE.en.md).

**From the zip:** there are two zips, `GPUEnsemble-<version>-windows-x64-app.zip` with the window app (`GPUEnsemble.exe`) and `GPUEnsemble-<version>-windows-x64-cli.zip` without it (command line only). Unpack one into a folder with a plain ASCII path on a drive with plenty of free space (for example `D:\GPUEnsemble`) and double-click `GPUEnsemble.exe` (app zip) or `Start GPUEnsemble.cmd`. Python is included; nothing else has to be installed. The app shows Japanese or English; "English" / "日本語" below its menu switches between them. If you have another PC with a GPU, unpack the zip there too and double-click `Start GPUEnsemble worker.cmd`: its GPUs are used as well (experimental; the first time, type `y` in the worker's window to let the main PC in). A PC at another place links the same way when both PCs are signed in to Tailscale with the same account, and then no `y` is needed ([measured between two sites](benchmarks/phase2b-two-sites-tailscale.md)). The model and llama.cpp go into the `data` folder next to the launchers. The zip is built by the GitHub Actions workflow [windows-zip](.github/workflows/windows-zip.yml), for a release or when started by hand (locally: `python packaging/windows/build_zip.py`).

**With Python:** 3.10 or later, no extra libraries.

```powershell
pip install .\planner
gpuensemble up
```

`up` does the following, in order:

1. reads the GPUs (VRAM and PCIe link), the RAM and the number of CPU threads
2. places each model of the catalog on this PC, estimates its speed, and picks the smartest one that still runs at reading speed (5 tokens/s) or faster
3. downloads that model from Hugging Face (resumed after an interruption, checked against its SHA-256)
4. downloads the llama.cpp build that suits this PC (CUDA 12.4 / Vulkan / CPU) and checks that it sees the GPUs
5. reads the actual file, assigns layers and MoE experts to the GPUs, and starts llama-server

Once it is up, `http://127.0.0.1:8080/` is a chat page and `http://127.0.0.1:8080/v1` is an OpenAI-compatible API. `gpuensemble recommend` lists the candidates with their estimated speeds, and `gpuensemble up --model qwen3-8b` names a model. While it runs, `gpuensemble monitor --every 2` shows the real generation speed, each card's memory, temperature, clocks and load, and which part takes the longest (the app shows the same on its home page). The details are in [planner/README.md](planner/README.md) (Japanese).

On the development PC (RTX 2080 Ti + RTX 2080 SUPER, 64 GB of RAM) it picks gpt-oss-120b, which runs at about 6.5 tokens/s (steady state). With an RTX 4090 at another site added through Tailscale, the same model ran at 10.7 tokens/s.

## The idea

- Split the model's layers into one contiguous range per card (a pipeline split), so the VRAM adds up without duplicating weights or the KV cache
- Only activations cross between cards (tens of kB per token), so neither NVLink nor P2P is needed; PCIe x4 and x1 slots are enough
- Small 8 GB cards are used too. Fast and slow cards, big and small VRAM: the planner finds the best split from measured numbers
- MoE experts that do not fit stay in RAM and are streamed to the GPU as they are needed
- The kernels come from [ggml](https://github.com/ggml-org/ggml) (what llama.cpp is built on), models are ordinary GGUF files, and the API is OpenAI-compatible

See [docs/DESIGN.md](docs/DESIGN.md) (Japanese).

## Target hardware (MVP)

| | |
| --- | --- |
| GPU | Several NVIDIA Turing cards (RTX 20 series). The development PC has an RTX 2080 Ti and two RTX 2080 SUPER |
| Backend | CUDA 12.x (not CUDA 13, so that Pascal and other old generations can be added later) |
| OS | Windows 10 / 11 |
| CPU | CPUs without AVX2 (such as Xeon E5 v1) are supported |

## Roadmap

| Phase | |
| --- | --- |
| 0 | Baselines with stock llama.cpp ([docs/PHASE0_BENCHMARK.md](docs/PHASE0_BENCHMARK.md)) |
| 1 | MVP: one PC with several GPUs, automatic placement planner, execution by llama.cpp's layer split (llama-server's OpenAI-compatible API), monitor (`monitor` and the app's home page) |
| 1c | A pipeline executor of our own (`engine/`), once there is a way for it to beat llama.cpp's layer split |
| 1b | MoE experts kept in RAM and streamed to the GPU |
| 2 | Older AMD cards through Vulkan next to NVIDIA, KV cache quantization, Pascal and other old generations |
| 3 | Several PCs (stages linked over TCP), micro-batching. As a first form, layers go to another PC's GPUs through llama.cpp's RPC (`worker` and `--remote`; `worker --lan` / `--remote auto` find each other on the LAN and through Tailscale; `Start GPUEnsemble worker.cmd` in the zip; a model that fits on one other PC runs there from end to end, with its API passed on, through `worker --serve`. Experimental. [Measured inside one PC](benchmarks/phase2a-rpc-loopback.md), [between two sites](benchmarks/phase2b-two-sites-tailscale.md)) |

**The MVP is done when** a 70B-class Q4 model that does not fit on one card runs without a hand-written split, at a speed within 10% of the llama.cpp layer split measured in phase 0. Checked on the development rig on 2026-11-17 (Llama 3.3 70B Q4_K_M: the planner's placement 1.22 tokens/s, the split tuned by hand 1.23; [measurement](benchmarks/mvp-70b-e5.md)).

## Repository layout

| Folder | |
| --- | --- |
| `engine/` | Room for a pipeline executor of our own (after the MVP; empty for now) |
| `planner/` | Python: device detection and the layer placement planner |
| `server/` | Room for an API server of our own (empty: the API is llama-server's, the monitor is the planner's `monitor`) |
| `benchmarks/` | Summaries of measurements |
| `docs/` | Design and procedures |
| `third_party/` | External code such as ggml, pinned to a version |

## Security

Lending GPUs to another PC uses llama.cpp's RPC server, which has neither a password nor encryption. GPUEnsemble therefore puts a small gate in front of it that lets in only the addresses you approve. Use it only on networks you trust: a home LAN, a cable between two PCs, or a VPN such as Tailscale. See [SECURITY.md](SECURITY.md).

## Thanks

The layer split and the MoE expert offload follow the ideas of [Strata](https://github.com/Niko1221/Strata).

## License

[MIT](LICENSE). The Python in the zip, and the llama.cpp builds and models downloaded at run time, come under their own licenses ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
