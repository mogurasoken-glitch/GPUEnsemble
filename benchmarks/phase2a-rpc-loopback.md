# Phase 2a: llama.cpp RPC over loopback on the E5 rig (2026-11-15)

Before linking a second PC, this measures what llama.cpp's RPC backend costs on
its own, with no real network in the way. The RTX 2080 SUPER is served by
`ggml-rpc-server` on the same PC, and llama.cpp reaches it over 127.0.0.1.
These numbers are the best case that a LAN can approach.

Setup:

- E5 rig: RTX 2080 Ti on PCIe Gen2 x8 and RTX 2080 SUPER on Gen2 x4, driver 560.94, Windows 10.
- llama.cpp b12375, CUDA 12.4 build, `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- RPC server: `ggml-rpc-server.exe -H 127.0.0.1 -p 50052 -d CUDA1`, which serves the SUPER only.
- The benchmark tool runs with `CUDA_VISIBLE_DEVICES=0` when the SUPER is reached through RPC, so it sees only the Ti locally.

## What the build has

The b12375 Windows CUDA build ships its RPC server as `ggml-rpc-server.exe`, not
`rpc-server.exe`, together with `ggml-rpc.dll`. All of its tools load the RPC
backend from that DLL.

`ggml-rpc-server` takes these options:

- `-H` sets the address to listen on (default 127.0.0.1).
- `-p` sets the port (default 50052).
- `-d` lists the devices to serve, separated by commas.
- `-t` sets the threads for the CPU device.
- `-c` keeps a local cache of the tensors it receives.

On the client side, llama-server takes `--rpc host:port,...` and llama-bench takes
`-rpc`/`--rpc`. Remote devices are named `RPC0`, `RPC1` and so on:

```
> llama-server --rpc 127.0.0.1:50052 --list-devices
CUDA0: NVIDIA GeForce RTX 2080 Ti (11263 MiB, 10107 MiB free)
CUDA1: NVIDIA GeForce RTX 2080 SUPER (8191 MiB, 7105 MiB free)
RPC0: 127.0.0.1:50052 (8191 MiB, 7105 MiB free)
```

`--rpc` must come before `--list-devices`. With the order reversed, the program
lists the devices and exits before it reads `--rpc`, so no RPC device appears.

The server's description of a device is its address, not the GPU's name. The
main PC therefore cannot tell the card's model, and so its bandwidth, from llama.cpp
alone. `gpuensemble worker` prints the card as a rig entry for this reason.

When idle with no client, the server holds 116 MiB on the SUPER, which is its CUDA
context. After the runs it held 146-162 MiB. It logged only connect and disconnect
lines, with no warnings.

## Speed

All runs use `llama-bench -fa 1 -lm none --no-host 1 -r 3 -p 512 -n 128`. pp512 and
tg128 are in tokens/s.

| Model | Placement | Arguments | pp512 | tg128 | Load and overhead, s |
|---|---|---|---:|---:|---:|
| Qwen3-14B Q4_K_M | Ti alone | `-ngl 99 -dev CUDA0` | 1582 | 53.91 | 37.5 (first read from disk) |
| Qwen3-14B Q4_K_M | Ti + SUPER through RPC | `--rpc 127.0.0.1:50052 -ngl 99 -sm layer -dev CUDA0/RPC0 -ts 2/1` | 1376 | 46.41 | 10.8 |
| Qwen3-14B Q4_K_M | Ti + SUPER, both local | `-ngl 99 -sm layer -dev CUDA0/CUDA1 -ts 2/1` | 1445 | 48.10 | 5.7 |
| gpt-oss-20b MXFP4 | SUPER through RPC + Ti | `--rpc 127.0.0.1:50052 -ngl 99 -sm layer -dev RPC0/CUDA0 -ts 3/22` | 3134 | 92.09 | 10.3 |
| gpt-oss-20b MXFP4 | SUPER + Ti, both local | `-ngl 99 -sm layer -dev CUDA1/CUDA0 -ts 3/22` | 3548 | 99.46 | 7.4 |

### Readings

- **Generation.** One stage behind RPC adds about the same time to every token
  for both models: 20.79 → 21.55 ms (+0.76) for Qwen3-14B, and 10.05 → 10.86 ms
  (+0.80) for gpt-oss-20b. The cost is fixed per token, not proportional to the
  stage's work.
  - It comes from the round trips each token makes to the server: the graph is
    sent, the inputs are set and the outputs are read back.
  - The planner charges `rpc_overhead` once per remote stage that holds layers:
    0.8 ms on a loopback address.
  - On a LAN every round trip adds the network's latency. The default there is
    2 ms, a guess until two PCs are measured.
- **Prompts.** RPC costs about 18-19 ms per 512-token batch: 354 → 372 ms for
  Qwen3-14B and 144 → 163 ms for gpt-oss-20b. Over loopback this is mostly fixed
  cost.
  - Over a network, the hidden states of the whole batch also cross the link:
    512 × n_embd × 4 bytes, about 10 MB for Qwen3-14B. That is about 90 ms per
    batch on 1 Gb/s Ethernet, and 9 ms on 10 Gb/s.
- **Loading.** Uploading the RPC card's share adds 3-5 s here: about 1.5 GB of
  weights in these runs. The same upload takes about 13 s on 1 Gb/s Ethernet and
  1.5 s on 10 Gb/s. The planner adds a note with this estimate for every remote
  stage.
- **Splitting a model that fits on one card costs speed even locally.**
  Qwen3-14B runs at 53.9 on the Ti alone and 48.1 split 2/1. The planner already
  keeps such models on one card.
- **The "Ti alone" gpt-oss-20b run is not a valid reference.** Its model buffer
  was 10.9 GiB, which fills the 11 GB card.
  - The Windows driver then very likely spilled into shared system memory. The
    run gave 12 tg and 113 pp instead of about 100 and 3500, with no error.
  - llama.cpp does not fail when this happens; it just runs about 8x slower.
  - The planner's VRAM caps for the Ti (9.8 GiB) keep plans out of this range.

## Planner check

`rig-e5-rpc-loopback.json` is the E5 rig file with the SUPER reached through RPC.
For gpt-oss-20b the planner picks the same split as the benchmark, `--rpc
127.0.0.1:50052 -dev RPC0,CUDA0 -ts 3,22`. It estimates 84.8 tokens/s with an empty
context. llama-bench measured 92.09, 8% above the estimate. The local estimate is
off by about the same amount: 91.0 against 99.46.

## `gpuensemble worker` and `up` through RPC

This test ran on branch planner/rpc-gpus.

- Window A ran `worker --device CUDA1`, which served the SUPER on 127.0.0.1:50052
  only.
- Window B ran `up` with the loopback rig and `CUDA_VISIBLE_DEVICES=0`.
- Each measurement is one `/completion` request generating 256 tokens with
  `cache_prompt` false, read from `timings.predicted_per_second` in tokens/s.

| Model | Run | llama-server placement | Loaded / warmed up | Generation, tokens/s | Estimate, empty context |
|---|---|---|---|---|---:|
| gpt-oss-20b | Through RPC | `--rpc 127.0.0.1:50052 -dev RPC0,CUDA0 -ts 3,22` | 15 s / 2 s | 60.8, 81.9, 75.8 | 84.8 |
| gpt-oss-20b | Through RPC, second start | same | 19 s / 2 s | 90.0, 77.8 | 84.8 |
| gpt-oss-20b | Local (`rig-e5-turing.json`) | `-dev CUDA1,CUDA0 -ts 3,22` | 15 s / 2 s | 83.5, 78.5, 77.0 | 91.0 |
| Qwen3-32B Q4_K_M | Through RPC | `--rpc 127.0.0.1:50052 -ngl 56 -dev CUDA0,RPC0 -ts 34,22 -ctk q8_0 -ctv q8_0 -t 32` | 101 s / 13 s | 4.60, 4.59, 4.59 | 5.2 |

### Readings

- **gpt-oss-20b, through RPC and local.** The two are within the request-to-request
  noise of the server. Both sit 7-10% under the estimate for an empty context.
- **Qwen3-32B.** It ran 12% under the estimate. Most of that gap is not RPC:
  - The RPC stage adds 0.8 ms to a 192 ms token.
  - With CPU layers in front, llama-server warned `resolve_fused_ops: layer 0 is
    assigned to device CPU but Flash Attention over K/V rows is assigned to device
    CUDA0 ... set to disabled`. llama-bench `-fa 1` did not print this, and measured
    4.90-5.00 for the same placement locally.
  - Loading took 101 s, with 5.8 GiB going to the SUPER over loopback.
- **Stopping.**
  - Ctrl+C on `up` ended llama-server in 1.6-2.2 s.
  - Ctrl+C on `worker` ended ggml-rpc-server in 0.6 s. Port 50052 closed and the
    SUPER went back to 0 MiB.
  - While the worker runs with no client, the SUPER holds 116-164 MiB.
- **No worker running.** llama-server exits after about 6 s with `Failed to connect
  to 127.0.0.1:50052` and `error while handling argument "-dev": invalid device:
  RPC0`, followed by its usage text, which reads like a wrong command. `up` and
  `run` now try each `--rpc` server first. If one does not answer, they say so and
  suggest starting `worker`.
- **`worker --cache`.**
  - ggml-rpc-server reported `local cache : C:\llm\gpue-home\rpc-cache\rpc`.
  - The first load saved 15 tensors, 1,282 MiB.
  - The second load saved nothing new, so it used the cache.
  - Nothing was written under `%LOCALAPPDATA%\llama.cpp`.
  - Over loopback the load times barely change, 23 s and then 19 s. The cache is
    for real networks.

## The gate in front of the RPC server (`worker --allow`)

With `--allow`, ggml-rpc-server listens on 127.0.0.1 only. A Python gate takes
the worker's address, lets in only the allowed addresses, and copies bytes both
ways unchanged. Every RPC round trip then passes one more hop. The test used the
same llama-bench gpt-oss-20b command as above, with `-r 5`, run alternately.

| Worker | pp512 | tg128 | CPU while benchmarking |
|---|---:|---:|---|
| Direct, 3 runs | 3107-3152 | 89.5-90.5 | ggml-rpc-server 11-14% of a core |
| Gate, 3 runs | 2707-2960 | 80.8-84.5 | gate 11-13%, ggml-rpc-server 30-31% |

- **The gate costs about 1.1 ms per token**, 7-10% at this speed, and 1-2 s more
  to load. The planner adds `RPC_GATE_OVERHEAD` for entries the worker marks with
  `"gate": true`.
- **Raising the gate threads' priority made no difference.** It was verified to
  take effect, at Highest and at TimeCritical, and the cost stayed the same. The
  cost comes from the extra hop itself, so the priority code was removed.
- **Served through `up`, the gate is also less steady.** Over 8 starts, the second
  and third requests averaged 64-71 through the gate against 79-80 direct, and
  single requests dropped to 49-58.
- **A connection from an address that is not allowed is closed at once.**
  llama-bench failed after 0.2 s with `RPC handshake failed`, and the worker logged
  `gate: refused a connection from 127.0.0.1`.
- **For a model at 5 tokens/s, 1.1 ms is 0.5%.** On a real LAN the network's own
  round trips cost several times more than the gate. When nothing else can reach
  the network, as with two PCs joined by one cable, `--allow-any` skips the gate.
