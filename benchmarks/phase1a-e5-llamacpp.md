# Phase 1a: planner-driven llama.cpp on the E5 rig (2026-11-14)

`gpuensemble-plan args` turns the plan into llama.cpp arguments. This checks that
llama.cpp follows them, and how fast the result is compared with the hand-tuned
phase 0 best. CUDA 12.4 build b12375, driver 560.94, `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`llama-bench -p 512 -n 128 -r 3 -fa 1`. Load maps from `llama-completion -v -c 2048`.

## Does llama.cpp follow the arguments?

| Question | Answer |
|---|---|
| With PCI_BUS_ID ordering, is CUDA0 the 2080 Ti? | Yes: CUDA0 = 2080 Ti (bus 02), CUDA1 = 2080 SUPER (bus 81) |
| Does `-dev CUDA1,CUDA0` give the first offloaded layers to CUDA1? | Yes. llama-bench takes `-dev CUDA1/CUDA0` |
| Are `-ts` values used as exact layer counts? | Yes, but the output head counts as one more offloaded layer on the last device |
| Does the `-ot` regex hit only the planned expert layers? | Yes: 168 tensors = 28 layers x 6 (gate/up/down weights and biases), nothing else |

The first version of the launcher did not count the output head, so every range came out
one layer off (Qwen3-32B: CPU 0-6 instead of 0-5). With `-ngl` and the last `-ts` entry
raised by one, the load log matches the plan layer for layer. The launcher now does this.

## Speed

| Model | Arguments | pp512 | tg128 | Peak VRAM Ti / SUPER (MiB) |
|---|---|---:|---:|---|
| Qwen3-32B Q4_K_M | phase 0 hand-tuned best (`-ngl 58 -ts 11/8`) | 247 | 5.66 | |
| Qwen3-32B Q4_K_M | planner, first version (`-ngl 58 -dev CUDA1/CUDA0 -ts 25/33`) | 158 | 6.69 | 10303 / 7264 |
| Qwen3-32B Q4_K_M | **planner, corrected** (`-ngl 59 -dev CUDA1/CUDA0 -ts 25/34`) | 184 | **7.67** | 10599 / 7264 |
| gpt-oss-120b | phase 0 hand-tuned best (`-ncmoe 28 -ts 32/4 -t 32`) | 20.7 | 5.14 | |
| gpt-oss-120b | planner, first version (`-ts 5/31`, `-ot` 28 expert layers) | 15.8 | 6.89 | 8753 / 7406 |
| gpt-oss-120b | **planner, corrected** (`-ts 5/32`, `-ot` 28 expert layers, `-t 32`) | 15.6 | **7.05** | 8792 / 7376 |

Text generation is **36-37% faster** than the best found by hand, with no manual tuning.
The gain comes from the order and the head: the slower 2080 SUPER takes the first
layers and the 2080 Ti, which has the larger and faster memory, takes the rest and the
output head. Neither run ran out of memory.

Prompt processing is 20-25% slower than the hand-tuned runs. With the SUPER first, the
prompt batch crosses the slow Gen2 x4 link of the second socket first; the planner only
models generation so far. Worth a look when the planner learns prompt cost.

## Real GGUF files

The rig now has Python 3.12 (embeddable), MinGit and gh under `C:\llm\tools`, and a
clone of this repo. `gpuensemble-plan describe` read all four files in about a second each:

| File | Layers | Kind | Weights per planner | File size |
|---|---:|---|---:|---:|
| Qwen3-14B Q4_K_M | 40 | dense | 8.378 GiB | 8.384 GiB |
| Qwen3-32B Q4_K_M | 64 | dense | 18.399 GiB | 18.405 GiB |
| gpt-oss-20b MXFP4 | 24 | MoE 32/4 | 11.266 GiB | 11.278 GiB |
| gpt-oss-120b MXFP4 | 36 | MoE 128/4 | 59.022 GiB | 59.034 GiB |

The 6-12 MiB gap is the header and tokenizer; the weights match llama-bench's size column.
Planner arguments from the real files, then benched (all 26 unit tests also pass on the rig):

| Model | Planner arguments | pp512 | tg128 |
|---|---|---:|---:|
| Qwen3-14B | `-dev CUDA0` (fits on the Ti alone) | 1046 | 51.0 |
| gpt-oss-20b | `-dev CUDA1/CUDA0 -ts 3/22` | 1707 | 90.1 |
| Qwen3-32B | `-ngl 59 -dev CUDA1/CUDA0 -ts 25/34` (same as above) | 180 | 7.67 |
| gpt-oss-120b | `-ts 5/32`, `-ot` 27 expert layers, `-t 32` | 16.4 | 7.33 |

## Planner refit

With the output head on the GPU, the phase 0 fit was too pessimistic whenever layers ran on
the CPU (Qwen3-32B: 5.8 estimated vs 7.67 measured), and too optimistic for gpt-oss-20b
(175 vs 90.1, MoE kernels cost more than their bytes). The rig file now has
expert rate 17 GB/s, CPU overhead 30 ms, and a fixed GPU time per layer (0.03 ms dense,
0.22 ms MoE). Estimates are now 8.0 / 7.6 / 93 against 7.67 / 7.33 / 90.1 measured.

## Prompt processing and the order of the GPUs

Prompt processing (pp) was 20-25% slower with the planner's arguments. Cause, confirmed on
the rig: for each prompt batch, llama.cpp's op offload copies the weights that live on the
CPU to the **first** device in `-dev`, over that device's PCIe link. Under WDDM
`nvidia-smi dmon` reads 0, so the link was identified from each GPU's copy-engine load:
only the first device's copy engine is busy after load.

| Model | Arguments | pp512 | tg128 |
|---|---|---:|---:|
| Qwen3-32B | SUPER first (Gen2 x4), `-ts 25/34` | 199 | 7.62 |
| Qwen3-32B | **Ti first (Gen2 x8)**, `-ts 35/24` | **278** | 7.54 |
| Qwen3-32B | SUPER first, `-nopo 1` | 52.7 | 7.79 |
| gpt-oss-120b | SUPER first, `-ts 5/32`, 27 expert layers on CPU | 16.3 | 7.15 |
| gpt-oss-120b | same, **`-nopo 1`** | **42.2** | **8.08** |

- Dense model with a few CPU layers: keep op offload, and put the GPU with the fastest link
  first (`-ts 34/25` ran out of memory on the SUPER once it also held the head; `35/24` fits).
- MoE with experts on the CPU: turn op offload off. Copying ~42 GiB of experts per batch
  over Gen2 x4 loses to the CPU, and generation gets faster too.
- `-ub` above 512 does not fit in the remaining VRAM for either model.

The planner now does both: among plans within 3% of the fastest generation, a dense model
with CPU layers gets the fastest-link GPU first, and MoE models with CPU experts get
`--no-op-offload` (`-nopo 1` for llama-bench). The last GPU also keeps 0.3 GiB more free
for the output head's compute buffer.

### End-to-end check of the planner's commands (real GGUFs)

| Model | Planner arguments | pp512 | tg128 |
|---|---|---:|---:|
| Qwen3-14B | `-ngl 999 -dev CUDA0` | 1047 | 50.6 |
| gpt-oss-20b | `-dev CUDA1/CUDA0 -ts 4/21` | 1689 | 89.3 |
| gpt-oss-120b | `-ts 5/32`, 27 expert layers on CPU, `-nopo 1`, `-t 32` | 42.3 | 7.80 |
| Qwen3-32B | Ti first, but 7 CPU layers (`-ngl 58 -ts 36/22`) | 209 | 6.62 |

The 0.3 GiB reserve for the last GPU cost Qwen3-32B a layer (the SUPER peaked at 7038 of
7566 MiB). It is back to 0, which gives exactly the verified `-ngl 59 -ts 35/24`
(pp 278 / tg 7.54). gpt-oss-120b needs a warm page cache: the first run after reading
40 GB of other models gave 28.6 / 7.07.

llama-server with the planner's server arguments (gpt-oss-120b, ctx 8192) loads in 22 s and
answers `/v1/chat/completions` correctly. Without `-t`, llama.cpp used 16 of the 32 threads
and generated 2.7 tokens/s; with `-t 32`, 4.3-5.0. The rig file now sets `threads: 32` and
the launcher passes `-t` whenever work runs on the CPU. The apparent gap to llama-bench
turned out to be a first-request penalty, see below.

### llama-server: memory-mapping a model bigger than the RAM budget

gpt-oss-120b (59 GiB) on 64 GB of RAM. With llama.cpp's default mmap, the whole file stays
mapped, including the ~14.5 GB the loader copies to the GPUs, so the working set sits at
55-56 GB and the OS keeps evicting expert pages. Measured with five *different* prompts per
new server process (generation tokens/s, server timings):

| Load mode | Requests 1-5 | Prompt speed | Load time |
|---|---|---|---|
| default mmap | 6.65 / 7.81 / 7.75 / 7.46 / 7.54 | falls to 1-6 when pages were evicted | ~20 s |
| `-lm none --no-host` (read into RAM) | **7.94 / 7.93 / 8.04 / 7.74 / 7.82** | 25-37 | ~280 s |

Repeating the same prompt hid the problem (later requests touch the same experts), and an
extra throwaway warm-up request did not help. Slots, context size, flash attention, batch
size, `--poll` and `--cpu-strict` were all within noise. Plain `-lm none` runs out of GPU
memory at load; `--no-host` is needed with it.

The launcher now adds `-lm none --no-host` to llama-server when the model's weights are
larger than the rig's RAM budget for weights (`ram_for_weights_gib`), and `run` stops
llama-server on any exit, including Ctrl+C while loading.

### Quantized KV cache for long contexts (Qwen3-32B, ctx 8192)

At ctx 8192 the f16 KV cache of Qwen3-32B takes 2 GiB of VRAM, which pushes layers back to
the CPU. With `--kv auto` (the default) the planner now also tries a q8_0 cache and takes it
when the estimate is more than 2% faster. Qwen3-32B at 8192 gets q8_0 and keeps 10 layers
on the CPU instead of 13 (with the rig file: `-ngl 56` instead of `-ngl 53`, plus
`-ctk q8_0 -ctv q8_0`). At 32768 it gets q8_0 with `-ngl 48`.

llama-bench `-p 512 -n 128 -r 3 -fa 1 -d 0,4096,7552` with the placements from `args`
without a rig file (q8_0: `-ngl 55 -ts 33/22`, Ti first; f16: `-ngl 52 -ts 22/30`,
SUPER first). Depth 7552 + 512 + 128 is exactly 8192.

| Depth (tokens already in the cache) | tg128 f16 | tg128 q8_0 | pp512 f16 | pp512 q8_0 |
|---:|---:|---:|---:|---:|
| 0 | 4.50 | 4.30 | 109 | 199 |
| 4096 | 2.76 | **3.64** | 104 | **183** |
| 7552 | 2.03 | **3.22** | 97 | **175** |

Peak VRAM (Ti / SUPER): f16 10389 / 7118 MiB, q8_0 10314 / 7360 MiB. Nothing ran out of
memory. The f16 run had the SUPER first, because without a rig file the planner does not
know the link speeds, so part of its prompt-speed gap is the slower link. The server
numbers below use the rig file for both.

llama-server with `--rig rig-e5-turing.json`, ctx 8192:

| Test | f16 (`-ngl 53`) | q8_0 (`-ngl 56`) |
|---|---|---|
| 5 different short prompts, generation tokens/s | 4.30 | **4.63** (first request cold: 3.97) |
| one 5996-token prompt, prompt / generation tokens/s | 148 / 2.70 | **196 / 3.74** |
| peak VRAM Ti / SUPER (MiB) | | 10577 / 7234 of 11264 / 8192 |
| answer to a factual question at temperature 0 | | identical to f16, character for character |

With an empty cache the two are about even. q8_0 pays off once the context fills: 1.3-1.6x
faster generation at depth 4k-7.5k, and faster prompts. gpt-oss-120b stays f16 (q8_0
would be 0.4% faster at 8192 and 1.6% at 32768, under the 2% threshold), so its command
does not change.

The planner picks the right cache type, but its estimates at 8192 are 25-30% optimistic
(q8_0 6.6 estimated against 4.6 measured, f16 5.7 against 4.3): it does not yet count the
KV cache reads per token, which grow as the context fills.

### How generation slows as the context fills (Qwen3-32B, KV depth sweep)

llama-bench `-fa 1 -sm layer -dev CUDA0/CUDA1 -p 512 -n 128 -r 2 -d 0,2048,4096,7552`, CUDA
12.4 b12375, after the forced Windows Update restart (same driver 560.94, same links).
tg128 / pp512 in tokens/s.

| Run | Placement | KV | Threads | depth 0 | 2048 | 4096 | 7552 |
|---|---|---|---:|---|---|---|---|
| R6 | 6 CPU layers, `-ngl 59 -ts 35/24` | f16 | 32 | 7.06 / 269 | | | |
| X2 | 9 CPU layers, `-ngl 56 -ts 34/22` | f16 | 32 | 5.03 / 224 | | | |
| R3 | 9 CPU layers, `-ngl 56 -ts 34/22` | q8_0 | 32 | 4.65 / 214 | 4.23 / 208 | 3.84 / 203 | 3.42 / 186 |
| R1 | 12 CPU layers, `-ngl 53 -ts 32/21` | f16 | 32 | 4.36 / 162 | 3.61 / 175 | 3.14 / 167 | 2.80 / 159 |
| R2 | 12 CPU layers, `-ngl 53 -ts 32/21` | q8_0 | 32 | 4.48 / 180 | 3.75 / 180 | 3.37 / 172 | 2.99 / 163 |
| R4 | 12 CPU layers | f16 | 16 | 4.56 / 166 | | 2.82 / 154 | |
| R5 | 12 CPU layers | q8_0 | 16 | 4.31 / 166 | | 3.32 / 157 | |

No run ran out of memory. Peak VRAM (Ti / SUPER, MiB): R1 10468 / 7342, R3 10810 / 7360,
R6 10479 / 7566. A rerun of R1 at depth 0 after the restart gave 4.42.

- **Time per token grows linearly with depth, in proportion to the CPU layers.** +16.9 ms
  per 1k tokens of context for R1, +14.7 for R2, +10.2 for R3: about 1.4 ms per 1k tokens
  per CPU layer with an f16 cache and 1.1-1.2 with q8_0. Attention on the GPUs adds well
  under 1 ms per 1k tokens. In bytes, the E5 runs attention over its KV cache at about
  3.0 GB/s (f16) and 1.9 GB/s (q8_0), against ~21 GB/s for weights.
- **The cache type alone changes little.** At the same placement, q8_0 equals f16 at depth
  0 and is 7% faster at 7552. Its real gain is the 3 layers it moves back to the GPUs.
- **Threads.** 16 and 32 are the same at depth 0. Deep into the context, f16 needs 32
  (2.82 vs 3.14 at 4096); q8_0 does not care. The launcher keeps `-t 32`.
- **A CPU layer costs ~14-19 ms per token**, not ~8.6 as phase 0 suggested: 127 / 142 /
  199 / 226 ms per token at 5 / 6 / 9 / 12 CPU layers. The first layers of Q4_K_M are the
  larger kind (see below), and the cost is steeper from 6 to 9 than from 9 to 12, so a
  straight-line fit is within about 5-12% across the range.

### Where the VRAM runs out

| Placement (Qwen3-32B, f16 cache, `-t 32`) | Context | Result | Peak VRAM Ti / SUPER (MiB) |
|---|---:|---|---|
| `-ngl 59 -ts 35/24` (SUPER: 23 layers + head) | 640 | fits, 7.06 / 269 | 10479 / 7566 |
| `-ngl 60 -ts 36/24` (Ti: 36 layers) | 640 | **fits, 7.89 / 295** | 10782 / 7566 |
| `-ngl 60 -ts 36/24` | 1152 | out of memory on the SUPER | 10854 / 7602 before the crash |
| `-ngl 59 -ts 35/24` | 1152 | out of memory on the SUPER | 10537 / 7598 before the crash |
| `-ngl 58 -ts 35/23` (SUPER: 22 layers + head) | 1152 | fits, 5.57 tg at depth 512 | 10525 / 7346 |

So the Ti held one layer more than the rig file allowed (its reserve was 1.3 GiB, now
1.15), and the SUPER with the output head sits right at its edge: 6.95 GiB of planned
weights and cache fit, 7.0 do not (`last_stage_reserve_gib` 0 → 0.03). At ctx 640 the plan
is now `-ngl 60 -ts 36/24`, 12% faster than before; at ctx 1152 it drops to the
`-ts 36/23` that fits.

`describe` on the real GGUF replaced the uniform estimate in
`examples/qwen3-32b-q4_k_m.json`. Q4_K_M keeps `ffn_down` and `attn_v` in Q6_K for the
first and last eighth of the layers and every third layer between, so 32 layers are 295 MiB
and 32 are 262 MiB.

Running each planned command with a full cache (`-d` = context - 640):

| Context | Command | Result | tg128 at full depth | Peak VRAM Ti / SUPER (MiB) |
|---:|---|---|---:|---|
| 2048 | `-ngl 58 -ts 35/23`, f16 | fits | 5.23 (est. 5.4) | 10651 / 7430 |
| 4096 | `-ngl 58 -ts 35/23`, q8_0 | fits | 4.66 (est. 5.1) | 10687 / 7444 |
| 32768 | `-ngl 49 -dev CUDA1/CUDA0 -ts 21/28`, q8_0 | out of memory on the SUPER | | 10573 / 7276 before the crash |

At 32768 the context could not be created: llama.cpp asked for a 316 MiB prompt compute
buffer on the SUPER, which already held 7.0 GiB of planned weights and cache. That buffer
grows with the context (the attention mask, and for a quantized cache one layer's K and V
converted to f16), which the fixed 1 GiB reserve did not cover. Every GPU now also keeps
context x (one layer's f16 KV bytes per token + 1 KiB) free: 160 MiB for Qwen3-32B at
32768, 40 MiB at 8192, 3 MiB at 640. The plans from 640 to 8192 do not change. Rechecked
with a full cache:

| Context | Command | Result | tg128 at full depth | pp512 | Peak VRAM Ti / SUPER (MiB) |
|---:|---|---|---:|---:|---|
| 16384 | `-ngl 53 -ts 32/21`, q8_0 | fits | 2.14 (est. 2.2) | 130 | 10692 / 7380 |
| 32768 | `-ngl 48 -dev CUDA0/CUDA1 -ts 29/19`, q8_0 | fits | 1.07 (est. 1.1) | 100 | 10834 / 7400 |

Both commands are identical with and without the rig file.

### Planner model and refit

The planner now prices each layer's KV cache reads at the current depth: on the GPU at its
bandwidth, on the CPU at `host.cpu_kv_gbs` for the cache type. It picks the placement
that is fastest at half the context and also prints the speed with an empty and a full
context. The rig file is refitted with the real layer sizes: dense CPU layers
`cpu_bandwidth_gbs` 34 → 19.6 with `cpu_overhead_ms` 30 → 10, MXFP4 experts
`expert_bandwidth_gbs` 17 → 14 (phase 0 measured ~13), `cpu_kv_gbs` f16 3.0 / q8_0 1.9.

| Measured vs estimated (tokens/s) | Measured | Estimate before | Estimate now |
|---|---:|---:|---:|
| Qwen3-32B ctx 640, `-ngl 60`, depth 0 | 7.89 | | 7.5 |
| Qwen3-32B ctx 8192, q8_0, `-ngl 56`, depth 0 | 4.65 (f16 at the same placement: 5.03) | 6.6 | 5.2 |
| same, depth 4096 | 3.84 | 6.6 | 4.25 |
| same, depth 7552 (full) | 3.42 | 6.6 | 3.6 |
| gpt-oss-120b, `-ts 5/32`, 27 expert layers on the CPU | 7.80 | 7.6 | 7.8 |

### Without a rig file

`gpuensemble-plan` without `--rig` now reads the RAM (64 GiB → 51 GiB for weights), the
thread count (32) and each card's PCIe link from nvidia-smi (Gen2 x8 / Gen2 x4 → the Ti
goes first). gpt-oss-120b plans instead of failing with "does not fit", and gets
`-lm none --no-host -t 32`. The detected cards use 75% of their published bandwidth. An
idle card is planned at its total VRAM minus 1 GiB, like the rig file, and a card that
drives the display at its free VRAM minus 0.55 GiB, which on the E5 is just under the rig
file's measured room. The CPU defaults are the E5's fitted values. The first version came
out one layer different from the rig file (8 CPU layers at ctx 8192, and q8_0 with the
SUPER first at ctx 640). Two fixes followed: the VRAM rule above, and `--kv auto` now tries
q8_0 only when it frees at least one layer's worth of VRAM. A smaller cache that only
squeezes in a layer at the edge of the estimate is not worth losing the fast-link order.
With the free memory nvidia-smi reported on the E5 (10620 / 7987 MiB), the commands with
and without the rig file are identical for Qwen3-32B and gpt-oss-120b at every context
from 640 to 32768.

### Hard-killing the launcher

`gpuensemble-plan run` with llama-server (Qwen3-14B, ctx 4096, 9.5 GiB on the Ti), then
`taskkill /F` on the Python process only:

| Launcher | llama-server after the kill | Port | GPU0 memory |
|---|---|---|---|
| before (main) | still running at +10 s | still answering | stays at ~9550 MiB |
| job object (`childproc.spawn`) | gone within 1 s, both runs | closed | back to ~400 MiB |

### Choosing a model (`recommend`)

`gpuensemble-plan recommend --ctx 8192` on the E5, catalog of 9 models. Speeds are the
estimate with an empty / half-full / full context. With and without the rig file the
picks are the same and the speeds differ by at most 0.2.

| Model | Size GiB | Estimate | Placement |
|---|---:|---|---|
| **gpt-oss-120b MXFP4** (recommended) | 60.7 | 7.7 / 7.7 / 7.6 | 2 GPUs, experts 23% on GPU |
| Qwen3-32B Q4_K_M | 18.4 | 5.4 / 4.4 / 3.7 | 9 layers on CPU, KV q8_0 |
| Llama-3.3-70B Q4_K_M | 39.5 | 0.7 / 0.6 / 0.5 | 49 layers on CPU |
| Qwen3-30B-A3B Q4_K_M (`--min-speed 20`) | 17.3 | 32.6 / 31.6 / 30.6 | experts 93% on GPU |
| gpt-oss-20b MXFP4 | 11.4 | 93.0 / 89.2 / 85.7 | 2 GPUs |
| Mistral-Small-3.2-24B Q4_K_M | 13.3 | 29.5 / 28.2 / 27.0 | 2 GPUs |
| Qwen3-14B Q4_K_M | 8.4 | 50.8 / 47.3 / 44.3 | 1 GPU |
| Qwen3-8B Q4_K_M | 4.7 | 89.4 / 80.0 / 72.5 | 1 GPU |
| Llama-3.1-8B Q4_K_M | 4.6 | 91.4 / 82.6 / 75.4 | 1 GPU |

Catalog sizes against `describe` of the real files: Qwen3-14B, Qwen3-32B and
Qwen3-30B-A3B match (within 0.04 GiB). The gpt-oss entries are too large (120b: 60.73
vs 59.03 GiB, 20b: 11.38 vs 11.28); their head and embedding are 0.573 GiB each and the
dense part of a layer is 0.027 GiB.

Qwen3-30B-A3B Q4_K_M, which had never run on the rig, was downloaded for the check
(official Qwen repository, 18.6 GB in about 6 minutes, SHA-256 matches). The planner's
command on the real file: `-ngl 999 -dev CUDA1/CUDA0 -ts 19/30 -t 32
-ot 'blk\.(0|19|20|21|22)\.ffn_.*_exps\.=CPU' -nopo 1` (2080 SUPER first, 5 layers of
experts in RAM, 1.3 GiB). Peak VRAM 10530 / 6908 MiB of 11264 / 8192.

| Depth | Estimate | tg128 (`-r 2`) | tg128 (`-r 5`) | pp512 |
|---|---:|---:|---:|---:|
| 0 | 33.1 | 55.1 ± 5.1 | 42.2 ± 5.1 | 412.8 |
| 4096 | 32.1 | 55.8 ± 2.9 | 50.7 ± 4.7 | 409.5 |
| 7552 | 31.1 | | 57.8 ± 0.1 | |

The estimate is 30-45% low. The planner adds its fixed CPU cost (10 ms, fitted on a
dense model with layers on the CPU) and 0.22 ms per MoE layer, which together are
two thirds of its 31 ms; the real token takes 17-24 ms. The spread between runs is
large and does not fall with depth, so it is probably the CPU side (`-t 32` on two
sockets) rather than the GPUs.

### MoE refit: exact profiles and the cost of experts in RAM

The gpt-oss examples used to be estimates whose head, embedding and dense part were
about twice the real size. With `describe` of the real files, and Qwen3-30B-A3B added,
the old cost model put gpt-oss-120b 4% too high and Qwen3-30B-A3B 22% too low: a fixed
10 ms "CPU used" cost, fitted on a dense model with whole layers on the CPU, does not fit
a model whose only CPU work is the experts of a few layers. Now a CPU prefix keeps the
10 ms, and each layer whose experts stay in RAM costs 0.5 ms for the hand-over to the
CPU and back. Under llama.cpp the experts move a whole layer at a time (`-ot`), and the
plan now prices exactly that, so the plan and the command agree.

Old and new placements, interleaved (A B C A B C) to cancel drift, llama-bench b12375,
`-fa 1 -p 0 -n 128 -nopo 1 -t 32`, after one discarded warm-up run:

| Model | Placement | Runs | Mean | Estimate |
|---|---|---|---:|---:|
| Qwen3-30B-A3B | A old: `-ts 19/30`, experts of 0, 19-22 on the CPU | 41.9, 39.4, 38.9 | 40.1 | |
| Qwen3-30B-A3B | B new (ctx 8192): `-ts 23/26`, experts of 0-3 on the CPU | 44.7, 44.8, 45.2 | **44.9** | 41.9 |
| gpt-oss-120b | A old: `-ts 5/32`, experts of 0, 5-30 | 5.95, 6.18 | 6.07 | |
| gpt-oss-120b | B new (ctx 640): `-ts 4/33`, experts of 4-30 | 6.18, 6.18 | 6.18 | 7.78 |
| gpt-oss-120b | C new (ctx 8192): `-ts 6/31`, experts of 0-1, 6-30 | 5.83, 5.94 | 5.89 | 7.7 |

The new Qwen3-30B-A3B placement is 12% faster and won every round. The gpt-oss-120b
placements are equal within the noise (±0.9-1.4 per run). `-t 16` gave the same speed
as `-t 32` with less spread (A 6.09, B 6.27).

gpt-oss-120b measures about 6.1 here against 7.80 in the runs above. llama-bench memory-maps
the 59 GB file, and with 52.5 GiB of free RAM the page cache cannot hold it, which is the
mmap slowdown already seen with llama-server. Served by `up` (read into RAM,
`-lm none --no-host`), it gave 6.0-7.5. The estimate of 7.8 is the top of what this rig
reaches today, and the rig file stays fitted to the 7.80 runs until the spread is
understood (NUMA placement of the experts in RAM is the main suspect). Update: refitted
to runs with the weights read into RAM, see the next section.

### Calibrating with the weights read into RAM

Why gpt-oss-120b's estimate (7.8) sat above what `up` serves (6.5-6.6 steady), on the
same build (b12375 CUDA 12.4, `-fa 1 -p 0 -nopo 1`), 2026-11-15.

**Not the quant type.** One GPU (the Ti, `-dev CUDA0 -ngl 99`), experts on the CPU,
llama-bench with its default memory-mapped file, tg128, `-t 32`:

| Experts on the CPU | gpt-oss-20b (MXFP4) | Qwen3-30B-A3B (Q4_K_M) |
|---|---:|---:|
| all (`-ot exps=CPU`), 1.27 / 1.10 GB per token | 9.22, 9.66 | 10.44, 10.69 |
| about half (`-ncmoe 12` / `-ncmoe 24`) | 20.79 | 20.62 |

Per byte both run at about 12 GB/s with 32 threads. With 16 threads Q4_K gains 18% and
MXFP4 loses 5%. The CPU backend is `ggml-cpu-sandybridge` (AVX, no AVX2 or FMA).

**Not the clock.** Power plan High performance. Under full load `% Processor Performance`
holds 114% (all-core turbo, about 3.08 GHz) for a whole 50 s run, with 32 or 16 threads.

**The memory-mapped file skews short runs.** Qwen3-30B-A3B, all experts on the CPU, one
GPU, `-t 32`:

| -n | default (mmap) | `-lm none` (read into RAM) |
|---:|---:|---:|
| 16 | 6.90 | 10.41 |
| 64 | 9.54 | 11.72 |
| 256 | 11.22 | 11.79 |
| 512 | 11.69 | 11.77 |

With the file mapped, the first touch of each 4 KB page of experts is a page fault, even
with the file in the page cache, and with 8 of 128 experts per token it takes hundreds of
tokens to touch most of them. The earlier MoE calibration runs (tg128, mmap) were skewed by
this; `up` reads the weights into RAM, so the calibration now does too.

**Recalibration**, `-lm none --no-host 1 -n 256`, the planner's own placements for ctx
8192 (A, B, E, F) and forced ones on the Ti alone (C, D):

| Run | Placement | -t 32 | -t 16 |
|---|---|---:|---:|
| A gpt-oss-120b | `-ts 6/31`, experts of 0-1 and 6-30 on the CPU (43 GiB in RAM) | 6.55 | 6.64 |
| A, -n 16 | same | 6.31 | 6.61 |
| B Qwen3-30B-A3B | `-ts 22/27`, experts of 0-3 on the CPU | 43.6 (41.0-47.8) | 37.3 |
| Ci Qwen3-30B-A3B, Ti | `-ncmoe 24` | 22.42 | 22.51 |
| Cii Qwen3-30B-A3B, Ti | `-ot exps=CPU` | 11.76 | 13.30 |
| Di gpt-oss-20b, Ti | `-ncmoe 12` | 21.51 | 21.79 |
| Dii gpt-oss-20b, Ti | `-ot exps=CPU` | 9.44 | 10.32 |
| E gpt-oss-20b | `-ts 3/22`, all on the GPUs | | 100.3 (93-108) |
| F Qwen3-32B | 9 CPU layers, KV q8_0 | 4.90 | 5.00 |

- **A matches `up`** (6.5-6.6), so llama-bench with `-lm none` is the right reference. It
  does not show `up`'s fast first tokens (about 11 tok/s for the first 16-48 tokens after a
  start), so those come from llama-server itself. Still open.
- **The CPU's rate for experts falls as more of them sit in RAM.** gpt-oss-20b and -120b do
  the same expert work per layer (4 experts, 53 MB in all, per layer and token). After subtracting
  the GPU's part, a layer's experts cost about 3.1 ms with 4.7 GiB of experts in RAM (Di),
  4.1 ms with 9.5 GiB (Dii) and 5.1 ms with 43 GiB (A). Qwen3-30B-A3B shows the same: twice
  the experts on the CPU take 2.3 times as long. A cost per byte plus a cost per layer
  cannot fit this (the best such fit puts A 26% high and Ci 19% low). Why is not known;
  remote NUMA reads and page-table walks over a larger set are candidates. Earlier,
  llama-bench with the file mapped and fully in the page cache gave 7.80 for the 120b,
  faster than read into RAM, so where the pages live may matter.
- **New model:** the CPU runs experts at `expert_bandwidth / (1 + 0.016 x GiB of experts in
  RAM)`, 17 GB/s with few of them, and no extra cost per layer. Since the amount in RAM
  depends on the plan, the planner plans again at the rate for its own amount until it
  settles (one or two rounds).

  | Run | Old estimate | New estimate | Measured (-t 32) |
  |---|---:|---:|---:|
  | A gpt-oss-120b | 7.78 (+19%) | 6.47 (-1%) | 6.55 |
  | B Qwen3-30B-A3B | 41.9 (-4%) | 48.1 (+10%) | 43.6 |
  | Ci | 15.5 (-31%) | 20.0 (-11%) | 22.42 |
  | Cii | 8.7 (-26%) | 10.7 (-9%) | 11.76 |
  | Di | 16.4 (-24%) | 20.1 (-7%) | 21.51 |
  | Dii | 9.0 (-5%) | 10.6 (+12%) | 9.44 |
  | E gpt-oss-20b (no experts in RAM) | 91.0 (-9%) | 91.0 (-9%) | 100.3 |

  The root mean square error falls from 19% to 9%.
- **Threads stay at 32.** 16 threads are as fast or faster wherever many experts run on
  the CPU (Cii +13%, Dii +9%, A +1%) and much steadier, but 14% slower for B, which has only
  4 layers of experts on the CPU, and B is the placement the planner picks for that model.
