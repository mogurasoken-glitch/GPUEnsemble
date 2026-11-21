# A main PC without a GPU (2026-11-17)

PR #69 lets GPUEnsemble run on a main PC that has no GPU: the plan uses the cards of other
PCs, and with none of those it runs the whole model on the CPU. This was checked on the E5 rig
made to look like a PC without a GPU.

Setup:

- Main PC: the E5 rig (2x Xeon E5-2680, 64 GB DDR3, Windows 10), with a separate GPUEnsemble
  home that holds llama.cpp b12375's **CPU build** (`setup --backend cpu`; it has `ggml-rpc.dll`,
  no CUDA), and a rig file that is `rig-e5-turing.json` with `"gpus": []`. The E5's own cards were
  not used by these runs.
- Lending PC: the RTX 4090 at another site, the Windows zip's worker (`--lan --cache`), over
  Tailscale. `up` measured a round trip of 7.2 ms.
- Model: Qwen3.8-27B Q4_K_M (16.2 GiB), context 8192.

## Every layer on the 4090, started from the main PC

`up --rig rig-nogpu.json --remote auto --model qwen3.8-27b --run here` (`--run here` keeps
llama-server on the main PC; without it, `up` asks the 4090 PC to run the model itself, see below).

- Command: `llama-server --rpc <4090 PC>:50052 -ngl 999 -sm layer -dev RPC0 -c 8192 --cache-ram 0 --metrics`.
- The 4090 PC downloaded the 1.2 GiB of weights it lacked itself (71 s); loaded in 282 s.
- Estimate 25.0 tok/s. Generation over 8 requests (max_tokens 16-256): 22.6-22.9 on the first
  request and the short ones, 11.0-14.2 on others. Whether something else used the 4090 at the
  time could not be seen from the main PC.
- Every prompt costs a fixed 3-5 s, whatever its length (61 tokens: 5.3-5.6 s; during the
  warm-up 12 tokens 3.8 s, 512 tokens 3.0 s), and 8-16 s when part of the cached conversation is
  reused. A guess, not checked: the model's 48 recurrent layers copy their state (3.1 MiB each)
  over the network.

The same model run on the 4090 PC by itself gives 45.8-46.0 tok/s ([phase 2c](phase2c-three-cards.md)),
and `recommend --remote auto` from the main PC without a GPU said so ("runs on the other PC",
45.4). So a model that fits on one card of another PC is best run on that PC (the default); the
path above is what a model larger than any one PC's cards uses, with each stage costing a round trip.

## CPU only

`up --rig rig-nogpu.json --model qwen3.8-27b` with no `--remote`: `llama-server -ngl 0 -dev none -c 8192 -t 32 --no-op-offload`.

- Loaded in 57 s, warmed up in 128 s.
- Prompt 61 tokens at 4.87 tok/s; generation 1.70 tok/s. The estimate was 1.2 / 1.1 (empty /
  half-full context), with the CPU rate of 19.6 GB/s that PR #72 has since raised to 25 GB/s
  ([cpu-rates-held.md](cpu-rates-held.md)), which brings the estimate to about 1.5.

## Detection

- Vulkan build (`setup --backend vulkan`, plan only): the cards were listed as `Vulkan0 NVIDIA GeForce RTX 2080 Ti`
  (11241 MiB, 10472 free) and `Vulkan1 NVIDIA GeForce RTX 2080 SUPER` (8201 MiB, 7432 free) and planned.
  This is the path AMD and Intel cards take; no AMD or Intel card was available to try.
- CUDA build (the normal home): detection through nvidia-smi as before.
- Found here: on Japanese Windows the display adapter list (PowerShell) has a Japanese name in it,
  and reading it with the console's code page failed and hid every card. It is now read as UTF-8
  (in PR #69 before the merge).
