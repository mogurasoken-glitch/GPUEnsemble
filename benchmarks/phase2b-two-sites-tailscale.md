# Phase 2b: two PCs at two sites over Tailscale (2026-11-15)

The first test with a real network between the PCs. The main PC and the PC that
lends its GPU are in different buildings, linked by Tailscale over the internet.

Setup:

- Main PC: the E5 rig (RTX 2080 Ti + RTX 2080 SUPER, 64 GB DDR3, Windows 10), running
  `up --model gpt-oss-120b --remote auto:<the other PC's Tailscale address>` from the branch of PR #27
  (it measures the round trip and plans with it).
- Lending PC: RTX 4090 24 GB, i9-9980XE, 32 GB RAM, Windows 11, running the Windows zip's
  `worker --host <its Tailscale address> --allow <the main PC's Tailscale address> --cache`.
- Tailscale found a direct path (no relay); `tailscale ping` gave 7 ms, `up` measured 7.2-7.7 ms.
- No Windows firewall or network setting was changed on either PC: Tailscale's own inbound rule lets
  traffic to the Tailscale address in, and the main PC only connects out.
- Same request for every run: "Explain how a CPU cache works in about 300 words.", max_tokens 400,
  temperature 0, context 8192.

## Plan

`up` placed gpt-oss-120b (MXFP4, 59 GiB) like this:

| Card | Layers | Weights | Estimated time per token |
|---|---|---|---|
| 2080 SUPER (local) | 0-3 | 6.4 GiB | 1.9 ms |
| RTX 4090 (other site) | 4-16 | 20.9 GiB | 31.6 ms |
| 2080 Ti (local) | 17-35 | 8.4 GiB, 26% of the experts; the rest run on the CPU | 66.8 ms |

22.7 GiB of weights stay in RAM. The estimate is 10.0 tok/s (9.8 with the 7.7 ms round trip).

## Results

| Run | Ready after | Generation | Prompt (79 tokens) |
|---|---|---|---|
| Main PC alone (no `--remote`) | 316 s | 6.45 tok/s (estimate 6.4) | 27.9 tok/s |
| + RTX 4090, first load | 804 s | 5.35, 6.18, 8.47 tok/s | 37.1 tok/s |
| + RTX 4090, second load (weights cached there) | 349 s | 10.68 tok/s (estimate 9.8-10.0) | 34.3 tok/s |

- **Speed.** Once the lending PC has the weights, the 4090 at another site makes gpt-oss-120b
  1.65 times as fast as the main PC alone, and the planner's estimate was right.
- **First load.** About 21.6 GiB went up through the Tailscale adapter at a steady 45-48 MB/s
  (370-400 Mb/s), against the 20.9 GiB the plan puts on the 4090. The whole load took 804 s,
  close to the 900 s that `up` waited for llama-server at the time. A slower link would have
  made it give up, so `up` now allows extra time for the weights that go to other PCs, counted at
  10 MB/s, and says every minute that it is still loading.
- **Second load.** With `--cache`, the lending PC kept what it received; the second load sent
  only hashes and small tensors (one 30 s sample at 3.9 MB/s). Most of its 349 s is the main PC
  reading its own 42 GiB share from disk.
- **The first session was slower and uneven** (llama.cpp's 3 s readings went from 2 to 12 tok/s).
  The cause is not known; the lending PC may still have been writing the 21 GiB it had just received
  to its cache. The second session did not show it.
- **Clean stop.** Ctrl+C stopped `up` all three times (exit code 130), with no llama-server left behind.

## The next night: found by itself, and the lending PC downloads its weights (2026-11-16)

Same two PCs, with main as of PR #33. Both were started the way the zip's launchers start them:

- Lending PC: `worker --lan --cache` (what `Start GPUEnsemble worker.cmd` runs), with nobody at it.
- Main PC: `up --model gpt-oss-120b --remote auto` (what `Start GPUEnsemble.cmd` runs, with the model
  named so that nothing else is downloaded). No address was given.

| Run | Ready after | Generation |
|---|---|---|
| Weights already cached on the lending PC | 373 s | 10.08, 9.76, 9.77 tok/s (estimate 9.8) |
| First load, cache emptied: the lending PC downloads its share itself | 502 s (133 s download + load) | 9.86, 9.79, 9.78 tok/s |
| For comparison, the first load the night before (the main PC sent the weights) | 804 s | 5.35, 6.18, 8.47 tok/s |

- **Found and let in without a hand.** `find` listed the lending PC at its Tailscale address after
  3.8 s. Both PCs are signed in to Tailscale with the same account, so the worker let the main PC in
  without anyone typing `y`.
- **The download.** The main PC told the worker which byte ranges of the model's file on Hugging Face
  its layers need (65 tensors, 20.8 GiB). The worker fetched them in 133 s, 168 MB/s on average,
  against the 46 MB/s at which the main PC had sent them. While llama-server then loaded, the main
  PC's upload stayed at zero but for one 30 s sample at 3.9 MB/s: the worker had every tensor under
  the name llama.cpp asks for. The 65 files it downloaded have the same names (the hashes of their
  bytes) and sizes as the 65 that the main PC had sent the night before.
- **Steady from the first request.** The uneven first session of the night before did not come back,
  so it went with the main PC sending the weights, not with a first load as such.
