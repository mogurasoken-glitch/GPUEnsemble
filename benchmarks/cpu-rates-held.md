# The CPU's rates with the cards held at full clocks (2026-11-17)

E5 rig (2x Xeon E5-2680 v1, 64 GB DDR3-1600, RTX 2080 Ti + RTX 2080 SUPER), Windows 10, driver
560.94 with "Prefer maximum performance", llama.cpp b12375 CUDA. llama-bench with the weights read
into RAM as `up` runs them (`-lm none --no-host 1 -fa 1 -t 32 -p 0`). "Held" means a Vulkan device
open on both cards for the whole run (`cardhold.CardHold`, what `up` does since the clock-hold
change); "not held" leaves the clocks to the driver.

## Why

The planner's estimates were about 20% below what `up` served. gpt-oss-120b on three cards (SUPER
0-3, the 4090 at the other site 4-16, the Ti 17-35 with the experts of 17-30 on the CPU) was
estimated at 12.1-12.2 tok/s and ran at 15.3-15.5 ([clock-hold.md](clock-hold.md)). Of the
estimated 82.5 ms per token, 59 ms were the experts on the CPU.

The CPU's rates in the planner (`detect_host`, `rig-e5-turing.json`) were fitted in phase 1a
(2026-11-15), before the NVIDIA setting and before `up` held the cards. In those runs the cards
dropped their clocks while they waited for the CPU, and that wait is in every layer the CPU works
on: the hand-over to the CPU and back runs at the lowered clocks. So the fitted CPU rates came out
lower than what the CPU does next to a card that stays awake.

## Measurements

| Run | Placement | Held | Not held | Old estimate | New estimate |
|---|---|---:|---:|---:|---:|
| gpt-oss-120b | `-ts 8/29`, experts of 0-3 and 8-30 on the CPU (43 GiB in RAM), tg128 | **8.53** | 6.33 | 6.48 | 8.67 |
| Qwen3.8-27B Q4_K_M | `-ngl 57`, 8 layers (2.3 GiB) on the CPU | **7.05** | | 5.78 | 6.90 |
| Qwen3.8-27B Q4_K_M | `-ngl 41`, 24 layers (6.0 GiB) on the CPU | **3.48** | 3.06 | 2.75 | 3.42 |
| Qwen3.8-27B Q4_K_M | same, depth 4096 | **3.15** | | | |
| Qwen3.8-27B Q4_K_M | `-ngl 25`, 40 layers (9.4 GiB) on the CPU | **2.31** | | 1.86 | 2.34 |
| Qwen3.8-27B Q4_K_M | all on the GPUs | 25.10 | | 23.9 | 23.9 |
| Llama 3.3 70B Q4_K_M | `-ngl 32 -ts 14/18`, 48 layers (22.8 GiB) on the CPU | **1.37** | 1.17 | 0.77 | 0.97 |
| Llama 3.3 70B Q4_K_M | same, depth 4096 | **0.83** | | | 0.77 |
| Llama 3.3 70B Q4_K_M | `-ngl 16`, 64 layers (30.4 GiB) on the CPU | 0.99 | | 0.59 | 0.75 |
| Llama 3.3 70B Q4_K_M | `-ngl 0`, all on the CPU | 0.90 | | 0.47 | 0.59 |

Estimates for the exact placement at depth 0, from the planner's formulas (scripts were
`C:\llm\speedfit\predict.py` and `fit.py` on the E5, not in the repo).

- **Holding the cards is worth 14-35% wherever the CPU works** (Qwen3.8-27B +14%, Llama 3.3 70B +17%,
  gpt-oss-120b +35%), and about 5% with everything on the GPUs.
- **Dense layers on the CPU:** after subtracting the GPUs' part, Qwen3.8-27B's CPU layers ran at
  24-26 GB/s plus the fixed 10 ms, the same rate for 2 to 9 GiB on the CPU. Llama 3.3 70B's ran at
  33-37 GB/s: its wider layers (8192 against 5120) do more per byte on this AVX-only CPU. The rate is
  fitted to Qwen3.8-27B (25 GB/s), so Llama 3.3 70B with an empty context stays about 30% above its
  estimate; at depth 4096 the CPU's slower attention over its KV cache (about 1.7 GB/s measured, 3.0
  assumed) brings it to 7% below.
- **Experts on the CPU:** with the per-GiB slowdown kept at 1.6%, gpt-oss-120b's 43 GiB in RAM
  (above) and the three-card run's 22 GiB (15.4 tok/s, with the other stages as estimated) both
  give 23-24 GB/s with few experts in RAM.

## What changed

| Host value (E5 rig, and the default without a rig file) | Before | After |
|---|---:|---:|
| `cpu_bandwidth_gbs` | 19.6 | 25 |
| `expert_bandwidth_gbs` | 17 | 23.5 |
| `expert_slowdown_per_gib`, `cpu_overhead_ms`, `cpu_kv_gbs` | 0.016, 10, 3.0/1.9 | unchanged |

| Placement | Estimate before | Estimate now | Measured |
|---|---:|---:|---:|
| gpt-oss-120b on three cards (empty context) | 12.2 | about 15.1 | 15.3-15.5 |
| gpt-oss-120b on the E5's two cards | 6.5 | 8.7 | 8.53 |

Not measured again: Qwen3-32B and Qwen3-30B-A3B (not on the E5 any more). Their old runs were not
held, so the calibration tests now only check that the estimate is above them by less than 35%.
The estimates assume the cards are held: with `--no-hold-clocks`, or when the power management mode
is not "Prefer maximum performance", generation is slower than estimated by about the amounts above.
