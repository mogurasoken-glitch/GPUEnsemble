# Phase 2c: three cards at two sites, other models (2026-11-16)

Phase 2b lent an RTX 4090 at another site to the E5 rig for gpt-oss-120b. This night
tried other models on the same three cards, and then looked at where the time per token
goes when a stage is on another PC.

Setup, as in [phase 2b](phase2b-two-sites-tailscale.md):

- Main PC: the E5 rig (RTX 2080 Ti + RTX 2080 SUPER, 64 GB DDR3, Windows 10), `up --model <id> --remote auto`.
- Lending PC: RTX 4090 24 GB at another site, the Windows zip's worker started the double-click way
  (`--lan --cache`), reached over Tailscale on a direct path. `up` measured a round trip of 7.6-8.2 ms.
- Same request for every run: "Explain how a CPU cache works in about 300 words.", max_tokens 400,
  temperature 0, context 8192. llama.cpp b12375.

## Three models

Generation speed in tok/s.

| Model | Main PC alone | Three cards | Placement on three cards |
|---|---|---|---|
| Qwen3-32B Q4_K_M (18.4 GiB) | 4.70-4.85 | 25.8-25.95 | all 64 layers and the head on the 4090 |
| Llama 3.3 70B Q4_K_M (39.6 GiB) | 1.21-1.24 | 4.13-4.21 | CPU 0-4, SUPER 5-17, 4090 18-62, Ti 63-79 and the head (KV q8_0) |
| | | 4.64 (one traced request) | CPU 0-3, Ti 4-22, SUPER 23-36, 4090 37-79 and the head (the planner's choice after this night's changes) |
| gpt-oss-120b MXFP4 (59 GiB) | 6.45 (phase 2b) | 8.4-8.9 | Ti 0-18, SUPER 19-22, 4090 23-35 and the head |
| | | 9.2-9.9 | SUPER 0-3, 4090 4-16, Ti 17-35 and the head (the planner's choice after this night's changes) |

- A model that fits on the 4090 alone runs there entirely, 5.4 times as fast as on the main PC.
  The two local cards stay empty: every layer moved to them made it slower (see below).
- The 70B model needs the CPU for 5 layers on the main PC and still runs 3.4 times as fast.
- gpt-oss-120b was slower than in phase 2b (9.8-10.7 tok/s). That night the 4090 was missing from the
  planner's table and was planned at the guess for an unknown card (225 GB/s, less than half the
  2080 Ti), which happened to put it in the middle. With its real speed in the table the planner put
  it last, with the output head, and that placement is slower. The rest of this page is why. With
  what was found, the planner chooses the placement of phase 2b again, now knowing the card's speed.

## What a stage on another PC costs

`GGML_RPC_DEBUG=3` with `-lv 5 --log-timestamps` makes llama-server log every command it sends to the
other PC and how long it took. Qwen3-32B, placed by hand (`-dev`, `-ts`), averages over tokens 11-50:

| Placement | Commands per token | The one that waits | This PC's own share | Per token |
|---|---|---|---|---|
| X: 4090 0-63 and the head | 7 SET_TENSOR, GRAPH_RECOMPUTE, GET_TENSOR | 38.2 ms (593.5 KiB of logits) | 0.8 ms | 39.8 ms |
| Z: 4090 0-31, Ti 32-49, SUPER 50-63 and the head | 6 SET_TENSOR, GRAPH_RECOMPUTE, GET_TENSOR | 20.1 ms (20 KiB of hidden state) | 29.1 ms | 50.3 ms |
| Z2: Ti 0-17, SUPER 18-31, 4090 32-63 and the head | 7 SET_TENSOR, GRAPH_RECOMPUTE, GET_TENSOR | 26.3 ms (593.5 KiB of logits) | 29.0 ms | 56.0 ms |

- **One round trip per token.** The inputs and the order to compute go out without waiting (0.4 ms
  together). Only the read of the result waits for the other PC. The planner had guessed three round trips.
- **The 4090 is as fast as the table says.** Z's 20.1 ms are the 7.6 ms round trip plus 12.5 ms for
  8.7 GiB of weights: about 750 GB/s, against 756 in the table (1008 GB/s published, times 0.75).
- **The head on the other PC costs about 5 ms per token.** The same 32 layers answer in 26.3 ms with the
  head and in 20.1 ms without. 0.8 ms of that is the head's own work; the rest is 0.6 MB of logits
  (4 bytes for each of the 151,936 tokens of the vocabulary) crossing the network at about 115 MB/s.
  gpt-oss has 201,088 tokens: 0.8 MB.

The same placements with 400 tokens, untraced, plus Y (4090 0-62, Ti 63 and the head):

| Placement | tok/s | ms per token | Tailscale down per token |
|---|---|---|---|
| X | 25.21, 25.51, 25.40 | 39.4 | 0.91 MB |
| Y | 24.85, 23.66, 25.17 | 40.7 | 0.30 MB |
| Z | 14.80, 14.57, 14.63 | 68.2 | 0.17 MB |
| Z2 | 12.14, 12.71, 12.99 | 79.2 | 0.78 MB |

Z and Z2 are much slower over 400 tokens than over the first 50. The next two sections explain the
two things in this table that the traces above do not.

## The same model on the lending PC itself

Qwen3-32B fits on the 4090 alone, so the main PC adds nothing but the network. To see how much that
is, the model was started on the 4090 PC itself the next morning (i9-9980XE, Windows 11, the same
llama.cpp b12375, `up --model qwen3-32b` from the Windows zip, the same request). The lending worker
stayed running beside it, idle, holding about 0.9 GiB of the card.

| Where Qwen3-32B runs | tok/s | ms per token |
|---|---|---|
| Started on the main PC, all layers on the 4090 over Tailscale (X above) | 25.2-25.95 | 38.5-39.7 |
| Started on the 4090 PC, three answers of 400 tokens | 41.39, 41.47, 41.54 | 24.1 |
| Started on the 4090 PC, llama-bench (`-p 512 -n 128 -r 3`) | 42.55 (prompt: 2900) | 23.5 |
| Started from the main PC, run by the 4090 PC's worker, API passed on (later that morning, see below) | 41.44, 41.59, 41.67 | 24.1 |

- **1.6 times as fast on the PC that holds the model.** The network costs 14.4 to 15.6 ms per token
  for a model that needs 24.1 ms by itself. The planner counts 14.7 ms for it: 9.5 ms for the wait
  (the 7.6 ms round trip, 0.8 ms as on loopback, 1.1 ms for the gate) and 5.2 ms for the logits.
- The card ran at 97% utilisation, 2805 MHz and 341 W, and read its 18.0 GiB per token at about
  820 GB/s. The planner on that PC estimated 36.4 tok/s from the table's 756 GB/s (published x 0.75).
- A model that fits on one other PC is better started on that PC. When that is at least 10%
  faster, `recommend` now shows that speed beside the model ("N if started on the other PC"), and
  `plan` and `up` say so in a note.
- **`up` on the main PC now starts such a model on that PC.** A worker started with `--serve` (the
  double-click worker is) runs a whole catalog model itself when the main PC asks, and the main PC
  passes its API on. `up --model qwen3-32b --remote auto` on the main PC, with the 4090 PC's worker
  on this build: 41.44, 41.59 and 41.67 tok/s at the main PC's 127.0.0.1:8080 (three answers of 300
  tokens), the speed of the 4090 PC itself. It answered 22 s after `up` was started (19 s the second
  time; the model was already on the 4090 PC), the first piece of a streamed answer came after
  0.17 s and the rest evenly, and Ctrl+C on the main PC stopped llama-server on the 4090 PC: a
  second start right after loaded on the same card. The main PC downloaded nothing and started no
  llama-server; its own two cards stayed idle.

## The local cards clock down while they wait

Z again, traced for 400 tokens, with `nvidia-smi` sampled every 0.5 s:

| Tokens | Per token | The read that waits | This PC's own share |
|---|---|---|---|
| 11-50 | 50.1 ms | 20.1 ms | 29.0 ms |
| 101-140 | 65.5 ms | 19.9 ms | 44.7 ms |
| 151-190 | 73.6 ms | 20.2 ms | 52.7 ms |
| 251-290 | 75.4 ms | 20.1 ms | 54.5 ms |

The network part does not change. The two local cards start each request at full clocks (P2, 1965 MHz)
and about 3 s later sit in P3 with the SM clock at 510-615 MHz and the memory clock at 5000 MHz
instead of 6800/7500. Utilization is about 33% and nvidia-smi's throttle reason is "GPU idle": each
card works for a third of every token and waits for the 4090 the rest, and the driver (560.94, default
power settings) takes that for light load. X uses no local card and stays flat.

gpt-oss-120b on the planner's new placement (SUPER 0-3, 4090 4-16, Ti 17-35 and the head, most of
the Ti's experts on the CPU) shows it more strongly. The estimate is 12.1 tok/s:

| Request | First 100 tokens | The rest | Whole request |
|---|---|---|---|
| 1 (cards at full clocks when it starts) | 11.52 tok/s | 9.46 | 9.90 |
| 2 | 9.11 | 9.19 | 9.17 |
| 3 | 9.35 | 9.23 | 9.26 |

In the first request the 2080 Ti is in P2 for 1.5 s, then in P3, and from 8 s on in P5 (SM clock
about 430 MHz, memory clock 810 MHz); in the 2nd and 3rd it is in P5 throughout, at about 32%
utilization. The 2080 SUPER, with 4 layers, sits in P8 at idle clocks (300 MHz, memory 405 MHz) at
about 20%. The throttle reason is "GPU idle" in every sample.

So a placement split between this PC's cards and another PC's runs slower than the sum of its parts:
Qwen3-32B's 32 local layers take 54 ms instead of 29, and gpt-oss-120b runs at the estimated speed only
until the cards clock down. Nothing was changed on the PC to avoid it (pinning the clocks or NVIDIA's
"prefer maximum performance" setting are untested), and the planner does not model it: its estimates
for such placements are 20-30% high.

## Llama 3.3 70B: the placement before and after, traced

After the changes the planner puts the 4090 last, with the head, and one layer fewer on the CPU. Both
placements were started by hand with the trace on, one request of 120 tokens each, the same night:

| | Before: CPU 0-4, SUPER 5-17, 4090 18-62, Ti 63-79 and the head | After: CPU 0-3, Ti 4-22, SUPER 23-36, 4090 37-79 and the head |
|---|---|---|
| The request | 4.41 tok/s | 4.64 tok/s |
| Per token, tokens 11-50 | 222.5 ms | 204.3 ms |
| Per token, tokens 61-110 | 237.4 ms | 230.1 ms |
| The read that waits | 35.8 ms (32 KiB of hidden state) | 42.0 ms (501 KiB of logits) |
| This PC's own share, tokens 11-50 | 186.3 ms | 162.7 ms |
| This PC's own share, tokens 61-110 | 200.6 ms | 187.3 ms |

- **The read is close to what the planner now computes.** Before: 7.6 ms of round trip plus
  21.6 GiB at 756 GB/s is 38.3 ms, measured 35.8. After: the round trip, 21.3 GiB with the head, and
  0.5 MB of logits at 117 MB/s are 42.2 ms, measured 42.0.
- **The head on the other PC costs 6 ms here and a layer off the CPU gains more.** The estimate for
  the new placement is 4.8 tok/s (4.4 with 4,096 tokens in the context); for the old one it was 3.7.
- The local cards are in P3 in both (SM clock 390-435 MHz, 22-28% busy): the clock-down is the same.

One run is not explained. The first start of the new placement through `up` (untraced, right after
the lending PC had downloaded 9.0 GiB of layers it lacked in 68 s, and with the model file read from a
cold disk cache: loaded in 272 s against 101 s) gave 1.73, 2.30 and 2.45 tok/s, swinging between 1.6
and 3.3 within a request. The traced run above used the same command line twenty minutes later. A
repeat through `up` was cut short when the main PC went offline, so what slowed that run is not known.
(Found the morning after: the main PC was installing a Windows update at that time and restarted for
it; the repeat is in "The morning after" below.)

## The server's cache of finished chats reads the KV cache back

The extra download per token in the table above (0.30 MB in Y against 0.02 MB of hidden state) is not
per token. Before the 2nd and 3rd request the server saves the finished chat to its RAM cache
(`--cache-ram`, 8 GiB by default), and for that it reads the chat's KV cache from the cards:

| Run | Layers on the 4090 | Wait before the next chat starts |
|---|---|---|
| X | 64 | 2.03 s |
| Y | 63 | 2.07 s, 1.99 s |
| Z, Z2 | 32 | 1.07-1.12 s |

422 tokens × 4 KiB × 64 layers are 108 MB, at about 54 MB/s. With `--cache-ram 0` the wait is 0.2 ms,
the download per request falls from 366 MB to 251 MB (the logits alone), and the speed is the same
(24.89-25.03 tok/s). `up` now passes `--cache-ram 0` when layers are on another PC. With gpt-oss-120b
started that way, the 2nd and 3rd chat began 0.4 and 0.5 ms after they arrived.

## The lending PC's own download: two faults

Both are fixed in PR #37 and were checked on the two PCs:

| Case | Before | After |
|---|---|---|
| gpt-oss-120b; the lending PC holds layers 23-35, which the main PC had sent it | it took the layers 4-16 it had downloaded earlier for them (same sizes), downloaded 0.6 GiB, and the main PC sent 23.2 GB: ready in 706 s | "has its share of the weights already (66 tensors, 21.4 GiB)", told by their first and last 64 KiB; 0.14 GB sent; ready in 412 s |
| Qwen3-32B; the main PC's file is another repository's build of the model, 1,024 bytes larger than the one at the catalog's address | downloaded 17.5 GiB that nothing used, ending with HTTP 416 | "... on Hugging Face has 19762149024 bytes and the main PC's has 19762150048: they are not the same file"; nothing downloaded; ready in 148 s |

When the lending PC could download its share itself, it was quick: Llama 3.3 70B's 20.6 GiB in 136 s.

## The morning after: the main PC's cards kept at full clocks

The owner set the NVIDIA Control Panel's power management mode to "Prefer maximum performance" on
the main PC, and the same placements were measured again through `up` (three answers of 400 tokens
each, `nvidia-smi` sampled every 0.5 s).

| Model and placement | The night before | With the setting | Estimate |
|---|---|---|---|
| Llama 3.3 70B: CPU 0-3, Ti 4-22, SUPER 23-36, 4090 37-79 and the head | 4.42-4.44 | 5.38-5.40 | 4.8 |
| gpt-oss-120b: SUPER 0-3, 4090 4-16, Ti 17-35 and the head | 9.2-9.9 | 12.98-13.67 | 12.1 |
| Qwen3-32B, X: 4090 0-63 and the head | 25.21-25.51 | 24.91-25.09 | 23.9 |
| Qwen3-32B, Y: 4090 0-62, Ti 63 and the head | 23.66-25.17 | 27.81-27.94 | 26.8 |

- **The 2080 Ti now stays at P2 and 1350 MHz** (memory 6800 MHz) through every run, where it sat
  at P5 or P8 and 420 MHz the night before. The speed no longer drops after the first 100 tokens.
- **The 2080 SUPER still clocks down** (P3 at 390 MHz with Llama 3.3 70B, P8 with gpt-oss-120b):
  the setting reached only one of the two cards. It holds few layers in these placements.
- **With the Ti awake, the head is better on this PC.** Y keeps the last layer and the head on the
  2080 Ti, so 20 KiB of hidden state cross the network with each token instead of 0.6 MB of logits:
  11% faster than X, and 4% above the estimate. The night before, Y was the slower of the two.
- The planner priced Y below X all along, but its search never offered it: in every order that
  ends on a card of the main PC, the cheapest split on paper was the head alone on that card
  behind the 4090's 64 layers, which llama.cpp cannot run, and with those dropped only X was left.
  The search now keeps the head with a layer (see the table below), and `up` starts Y.
- The slow run of Llama 3.3 70B the night before (1.7-2.5 tok/s, one `up` at 04:03-04:10) fell
  inside the installation of a Windows update on the main PC (04:01-04:33, then a restart). It did
  not come back in these runs.

## What changed in the planner

| Change | Before | After |
|---|---|---|
| RTX 30/40/50 cards in the bandwidth table (PR #35, #36) | the 4090 planned at 225 GB/s | 756 GB/s |
| Round trips per token for a stage on another PC | 3 (a guess) | 1 (traced) |
| Logits coming back from a head on another PC | not counted | vocabulary × 4 bytes at the network's speed (1 Gb/s unless the rig file says) |
| Finished chats kept in RAM | llama-server's default (8 GiB) | off when layers are on another PC |
| The head in the search for a placement (the morning after) | allowed alone on the last card, and such plans dropped afterwards | always with a layer, so every order gives a plan that runs as priced |

Estimates against measurements (tok/s, empty context, `up` on the main PC without a rig file):

| Model and placement | Estimate before | Estimate now | Measured |
|---|---|---|---|
| Qwen3-32B, all on the 4090 | 18.6-19.1 | 23.4 | 25.2-25.95 |
| gpt-oss-120b, 4090 last with the head | 10.4 | no longer chosen | 8.4-8.9 |
| gpt-oss-120b, 4090 in the middle, head on the 2080 Ti | 9.8-10.0 (phase 2b: card unknown, 3 round trips) | 12.1 | 9.2-9.9 (11.5 until the local cards clock down); 9.8-10.7 in phase 2b |
| Llama 3.3 70B, 4090 in the middle, head on the 2080 Ti | 3.7 | no longer chosen | 4.13-4.21 (4.41 traced) |
| Llama 3.3 70B, 4090 last with the head | - | 4.8 | 4.64 traced (one more run at 1.7-2.5 during a Windows update); 5.38-5.40 the morning after |

With the head on this PC, 0.02 MB per token come down from the other PC instead of 0.86 MB.
The estimate of phase 2b was close because two errors cancelled what it did not know: the 4090 at
less than a third of its speed and three round trips, against local cards that slow down while they wait.

Still open:

- The clock-down of the local cards is not in the estimate (see above). It is the largest error left
  for a placement that mixes this PC's cards with another PC's.
- Without a rig file the CPU part is a rough guess: Llama 3.3 70B on the main PC alone was estimated
  at 0.6 tok/s and ran at 1.2.
- The speed of the network is assumed, not measured. A slow upload at the lending PC's site would make
  a head there cost more than the 5 ms measured here.

## Qwen3.8-27B, a hybrid model (the same afternoon)

Qwen3.8-27B (GGUF architecture `qwen35`, llama.cpp b12375) has 64 layers: 48 Gated DeltaNet layers that keep a
state of a fixed size, and 16 attention layers (3, 7, ... 63) with a KV cache. Its GGUF counts one more block
(`blk.64`, for predicting tokens further ahead) in `block_count`; llama.cpp counts it for `-ngl` and `-ts`
("offloaded 66/66 layers") but loads none of it. Started on the main PC with `up --remote auto`, ctx 8192,
three answers of 300 tokens each (all of them thinking: the model reasons before it answers by default):

| File | tok/s | Estimate | Placement |
|---|---|---|---|
| Q4_K_M (16.2 GiB) | 30.18, 30.04, 29.82 | 29.3 | all 64 layers on the 4090, the head on the Ti |
| Q8_0 (27.1 GiB) | 20.44, 20.45, 20.43 | 18.9 | 4090 0-53, Ti 54-63 and the head |

What llama.cpp reserved (Q8_0 run): 512 MiB of KV cache for the 16 attention layers (4096 bytes a token
each), and 598.5 MiB of recurrent state: 3,268,608 bytes for each of the 48 DeltaNet layers and each of
llama-server's four sequences, whatever the context. The compute buffer was 285 MiB on the 4090 and 195 MiB
on the Ti; Qwen3-32B on the main PC's own cards takes 250 and 178 MiB, so that is usual.

- **The planner read this model wrongly, and two errors cancelled.** It took `blk.64` for a real layer and
  gave all 65 a KV cache (four times too much), and did not know the recurrent state. The memory it counted
  for the 4090 came out about right (22.1 GiB counted, 1.33 GiB free during the Q8_0 run), but on the main PC
  alone the plan for Q4_K_M left a layer on the CPU that fits on the cards. The reader now counts `blk.64`
  as an empty layer, gives a KV cache to the attention layers only (from `attention.recurrent_layers`, or
  from `full_attention_interval` in a file without that list, as the Q8_0 file is) and counts the state.
- The 2080 SUPER held nothing in either run: the 4090 and the Ti were enough.
- Prompts were slow with the layers on the other PC: 13 to 15 tok/s for a 67-token prompt.

With the corrected reader, on the same files:

| Run | tok/s | Estimate | Placement | Free on the cards while it ran |
|---|---|---|---|---|
| Q4_K_M, main PC alone | 22.97, 23.45, 23.53 | 23.6 | SUPER 0-26, Ti 27-63 and the head, no layer on the CPU | SUPER 0.72 GiB, Ti 0.59 GiB |
| Q8_0, three cards, the 4090 filled as the main PC's own cards are | 10.64, 10.21, 10.57 | 20.2 | 4090 0-55, Ti 56-63 and the head | 4090 1.0 GiB as its PC reports it |
| Q8_0, three cards, 1 GiB more left free on the 4090 | 20.15, 20.18, 20.16 | 19.9 | 4090 0-53, Ti 54-63 and the head | 4090 1.32 GiB as its PC reports it |

- **On the main PC alone, Q4_K_M now runs on the two cards**, where the old reading left a layer on the
  CPU (14 tok/s estimated). llama.cpp's own count had the SUPER 195 MiB over its free memory and the Ti 64 MiB
  under; both ran at the estimated speed. A 67-token prompt was read at 90 tok/s.
- **A card on another PC must keep more room.** With the corrected numbers the planner filled the 4090 to
  what llama.cpp counted as its last free MiB (21,656 MiB of weights, 448 of KV cache, 524 of state, 285 of
  compute buffer against 22,914 free). It loaded without an error, took 180 s instead of 110 s, and ran at
  half the speed: Windows moves what does not fit into shared system memory. Two layers fewer (the Q8_0 run
  further up: 812 MiB free by the same count) ran at 20.4 tok/s. The planner now leaves 1 GiB more free on a
  card on another PC (`devices.REMOTE_FREE_MARGIN`), which gives the placement of that run (third row:
  ready in 105 s, the estimated speed).
- The same margin takes one layer of gpt-oss-120b off the 4090 (SUPER 0-4, 4090 5-17, Ti 18-35 and the head;
  12.6 tok/s estimated instead of 13.1 with 14 layers there). Thirteen layers on the 4090 is what the
  runs of gpt-oss-120b above had; the placement with fourteen was never measured. Qwen3-32B and Qwen3.8-27B
  Q4_K_M are placed as before: they leave the 4090 room to spare.
- A whole answer needs room: with max_tokens 2000 most of it went to the reasoning (5,119 characters, then
  1,006 characters of answer, cut off). 23.84 tok/s over those 2000 tokens.

### After the merge (main a959bb2, the same evening)

| run | tok/s | estimate | notes |
|---|---|---|---|
| gpt-oss-120b on three cards, `up --remote auto`: SUPER 0-3, 4090 4-16, Ti 17-35 and the head | 13.50 / 13.02 / 13.01 | 12.2 | the placement of the morning's runs (12.98-13.67): the margin took nothing off the 4090 here; ready in 351 s + 14 s |
| Qwen3.8-27B Q4_K_M, `up --model qwen3.8-27b --remote auto`: runs on the 4090 PC by itself | 45.80 / 46.02 / 46.02 | 45 | "fits on the other PC alone: starting it there … about 45 tokens/s instead of 30"; prompt 294 tok/s (67 tokens) |

- The 4090 PC's worker (build of main a959bb2) had no file of the model: it downloaded the 16.24 GiB itself
  in about 4.5 minutes, its progress shown on the main PC, then loaded in 13 s. Ready 296 s after `up` was
  started; the main PC downloaded nothing and its cards held nothing.
- While the model ran, the worker told 6.04 GiB free on the 4090; after Ctrl+C on the main PC, 23.36 GiB.
- With max_tokens 400 all 400 tokens were reasoning (1,781 characters) and the answer was empty.
