# Install and go on the E5 rig (2026-11-15)

`gpuensemble pull`, `setup` and `up` on the development rig (2080 Ti + 2080 SUPER,
64 GB, driver 560.94, Windows 10), with an empty GPUEnsemble home
(`GPUENSEMBLE_HOME=C:\llm\gpue-home`) and no rig file. The code was planner/moe-refit,
which also holds the refitted MoE estimates.

## pull

All 9 catalog sources resolve to one file each with a published SHA-256, and none
needs a login.

| Model | Repository | File | GiB |
|---|---|---|---:|
| gpt-oss-120b | ggml-org/gpt-oss-120b-GGUF | gpt-oss-120b-MXFP4.gguf | 59.03 |
| Qwen3-32B | Qwen/Qwen3-32B-GGUF | Qwen3-32B-Q4_K_M.gguf | 18.40 |
| Llama-3.3-70B | bartowski/Llama-3.3-70B-Instruct-GGUF | Llama-3.3-70B-Instruct-Q4_K_M.gguf | 39.60 |
| Qwen3-30B-A3B | Qwen/Qwen3-30B-A3B-GGUF | Qwen3-30B-A3B-Q4_K_M.gguf | 17.28 |
| gpt-oss-20b | ggml-org/gpt-oss-20b-GGUF | gpt-oss-20b-MXFP4.gguf | 11.28 |
| Mistral-Small-3.2-24B | unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF | Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf | 13.35 |
| Qwen3-14B | Qwen/Qwen3-14B-GGUF | Qwen3-14B-Q4_K_M.gguf | 8.38 |
| Qwen3-8B | Qwen/Qwen3-8B-GGUF | Qwen3-8B-Q4_K_M.gguf | 4.68 |
| Llama-3.1-8B | bartowski/Meta-Llama-3.1-8B-Instruct-GGUF | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | 4.58 |

Qwen3-8B (4.68 GiB):

1. A pull killed with `Stop-Process -Force` after 28 s left a 1.51 GiB `.part` file.
2. The next pull resumed from there and finished at 65-71 MB/s, and the hash matched.
3. A third pull answered "already downloaded" in 0.25 s.

## setup

The first run took 41 s:

- It chose CUDA 12.4 for driver 560.94 and downloaded b12375 (319 MB) and the CUDA runtime (391 MB). Both SHA-256 values matched.
- llama.cpp then crashed at start (0xC0000005) on the system Visual C++ runtime 14.28.
- setup copied the runtime DLLs from Edge (`EdgeCore\157.0.4341.46`) next to llama.cpp.
- `--list-devices` then showed both cards.

A second run answered "already installed" in 0.5 s.

## up

`gpuensemble up --model qwen3-8b --yes`:

- **Plan:** the whole model on the 2080 Ti, `-ngl 999 -dev CUDA0 -c 8192`.
- **Startup:** ready in 55 s. The web UI and `/v1/models` (listing `qwen3-8b`) both answered.
- **Generation:** 80.4 and 81.2 tokens/s, against an estimate of 89 with an empty context.
- **Stop:** a hard kill of the launcher ended llama-server within 1 s and freed the VRAM.

`gpuensemble up --yes` with no model named. gpt-oss-120b was already in the home through a hard link, so nothing was downloaded:

- **Choice:** it recommended gpt-oss-120b and found it downloaded.
- **Plan:** made on the real file: `-dev CUDA1,CUDA0 -ts 6,31 -t 32`, experts of 27 layers on the CPU, `--no-op-offload -lm none --no-host`.
- **Startup:** ready in 323 s (the file is read into RAM). Peak VRAM was 10515 / 6946 MiB.
- **Generation:** against an estimate of 7.8 / 7.7 / 7.7 tokens/s (empty / half / full context):

  | Request | Prompt tokens | Prompt tok/s | Generated | tok/s |
  |---|---:|---:|---:|---:|
  | "Say hello in Japanese" | 76 | 21.2 | 54 | 7.46 |
  | 1159-word text, 3-sentence summary | 1387 | 43.7 | 200 | 6.23 |
  | first request again | 76 | | | 6.03 |

- **Stop:** a hard kill freed the VRAM, and llama-server was gone 7 s later, after releasing 43 GiB of RAM.

### What a first-time user will notice

- **Slow first requests (now handled at startup).** The CUDA 12.4 build has no ready kernels for Turing, so the driver compiles them on first use, for each new batch shape. For Qwen3-8B, a 17-token prompt took 30 s at first. Once the shapes were compiled, new prompts took 24-26 ms. The compiled kernels are kept (`CUDA_CACHE_MAXSIZE` 4 GB), so later starts skip most of this. `up` now warms up before it says ready (see below).
- **Reasoning tokens.** Qwen3 and gpt-oss reason before they answer. With a small `max_tokens`, the reply can be cut off while it is still reasoning, so `content` comes back empty. Allow more tokens, or for Qwen3 send `chat_template_kwargs: {"enable_thinking": false}`.
- **pip on embeddable Python.** The E5 uses the embeddable Python, which has neither pip nor venv, so `pip install` could not be tried there. The commands were run as `python -m gpuensemble_planner`.

## Warm-up before "ready"

`up` sends eight completions of 1 to 512 prompt tokens after loading. Each run below used
Qwen3-8B with an empty CUDA kernel cache (a new `CUDA_CACHE_PATH`), and then three chat
requests:

| Start | Loaded | Warm-up | Ready | 1st request (14 tokens) | 32 tokens | 401 tokens |
|---|---:|---:|---:|---:|---:|---:|
| empty cache, warm-up | 98 s | 99 s | 197 s | 25 ms | 36 ms | 196 ms |
| empty cache, `--no-warmup` | 81 s | | 82 s | **60,891 ms** | 28 ms | 158 ms |
| warmed cache, warm-up | 9 s | 1 s | 10 s | 23 ms | 27 ms | 156 ms |

Without the warm-up, the first chat on a new PC waits about a minute. In the warm-up
itself, the 4- and 12-token prompts took 38 s and 61 s, and every other length took
under 0.2 s. A warmed cache also cuts loading from 98 s to 9 s.

For gpt-oss-120b on the normal cache, loading took 293 s and the warm-up 24 s. That time
is CPU prompt processing (46 tokens/s at 512 tokens), not compilation. Afterwards, a short
chat generated at 9.27 tokens/s, the best so far for this model in `up`, and a
1081-token prompt at 6.73.

The first version wrote its "warmed up" and "ready" lines into the middle of
llama-server's log lines, which share the terminal. llama-server's output now passes
through the launcher a line at a time.
