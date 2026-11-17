# MVP check: Llama 3.3 70B on the E5 rig (2026-11-17)

The MVP is done when a 70B-class Q4 model that does not fit on one card runs without a
hand-written split, at a generation speed within 10% of the llama.cpp layer split. Phase 0
had no 70B run, so this night measured both sides on the same rig, one after the other.

Setup:

- The E5 rig alone: RTX 2080 Ti 11 GB + RTX 2080 SUPER 8 GB (19 GB of VRAM), dual Xeon E5-2680, 64 GB DDR3,
  Windows 10, driver 560.94. No card of another PC (`--remote` not used).
- Llama 3.3 70B Instruct Q4_K_M (39.6 GiB, 80 layers), downloaded with `gpuensemble pull --model llama-3.3-70b`.
- llama.cpp b12375, CUDA 12.4. GPUEnsemble main d51cd94, no rig file.
- All runs: `llama-bench -p 512 -n 128 -r 3`. Numbers are tokens/s.

## llama.cpp, split by hand

The phase 0 way: both cards, `-sm layer`, and `-ngl` and `-ts` tried until the cards are full.

| Config | pp512 | tg128 | Peak VRAM, Ti / SUPER (MiB) |
|---|---:|---:|---|
| `-ts 11/8 -ngl 32 -fa 1` (the best) | 35.00 | **1.23** | 10102 / 7434 |
| `-ts 11/8 -ngl 32 -fa 1 -t 32` | 34.48 | 1.22 | |
| `-ts 12/8 -ngl 34 -fa 1` | 20.38 | 1.08 | the Ti is full and spills |
| `-ts 11/8 -ngl 33 -fa 1` | | 1.14 | |
| `-ts 11/8`, `-ngl 34` and `36` | | | out of memory on the SUPER |
| `-ts 13/8 -ngl 34` | | | out of memory on the Ti |

From `-ngl 26` to `32` the generation speed rises from 1.05 to 1.25 (single runs of the scan),
then falls: 32 layers on the cards is the most this rig holds.

## GPUEnsemble

`gpuensemble args --model llama-3.3-70b --tool bench` printed the split; nothing was written by hand.
The plan: layers 0-48 on the CPU (23.9 GiB in RAM), 49-62 on the 2080 SUPER, 63-79 and the
output head on the 2080 Ti.

| Config | pp512 | tg128 | Peak VRAM, Ti / SUPER (MiB) |
|---|---:|---:|---|
| as printed: `-ngl 32 -sm layer -dev CUDA1/CUDA0 -ts 14/18 -t 32` | 29.16 | **1.22** | 10236 / 7210 |
| the same with `-fa 1` | 34.04 | 1.22 | |

**Generation: 1.22 against 1.23 tokens/s, a ratio of 0.99 (1.00 with the same `-t`). The condition
(within 10%) is met.** The planner found the same 32 layers by itself, and put the head on the
bigger card.

`gpuensemble up --model llama-3.3-70b` then loaded the model in 35 s, warmed up for about 70 s,
and answered two chat requests at 1.37 and 1.35 tokens/s (llama-server's own timings).

## What the run showed besides

- **`args --tool bench` lacked `-fa 1`.** llama-bench leaves flash attention off unless told
  (llama-server turns it on by itself), which cost 17% of the prompt speed (29.2 against 34.0).
  `args --tool bench` now adds it.
- **The estimate is about half the real speed without a rig file**: 0.6 tokens/s estimated, 1.22-1.37
  measured. With 49 of 80 layers on the CPU the speed is set by the RAM's bandwidth, and the
  default for an unmeasured CPU is on the careful side. Models mostly on the CPU look slower in
  `recommend` than they are. Not tuned here.
- **The monitor ran on real hardware for the first time**, and needed fixes, all in the same change as
  this file: the speed counted the warm-up's requests (1.5 shown, 1.37 real); one decimal is too
  coarse for a model at 1.3 tokens/s; the text said "generating" while the model was idle; a
  killed `up` left its run file behind; and the 2080 Ti was reported as clocked down at
  1350 of 2190 MHz in P2 although the CPU took 97% of each token's time (a card that waits for the
  CPU most of the time is not slowed by its clocks).

Raw logs are on the rig in `C:\llm\mvp70b\` (not in git).
