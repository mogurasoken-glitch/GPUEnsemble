# Keeping the main PC's cards at full clocks while they wait (2026-11-16)

E5 rig (RTX 2080 Ti on PCIe 2.0 x8, drives the display; RTX 2080 SUPER on a riser, PCIe 2.0 x4),
Windows 10, NVIDIA driver 560.94. `nvidia-smi` sampled every 250 ms.

## The question

On the morning of 2026-11-16 the owner set the NVIDIA Control Panel's power management mode to
"Prefer maximum performance", and the three-card runs got faster (gpt-oss-120b 9.2-9.9 → 13.0-13.7
tok/s, see [phase2c-three-cards.md](phase2c-three-cards.md)). The 2080 Ti stayed at P2 throughout;
the 2080 SUPER still fell to P3/P8 ("GPU idle") while it waited for the other stages.

## What the setting does

- The setting is in the driver's global profile (Base Profile, `PREFERRED_PSTATE` = 1). There is no
  profile for `python.exe` or `llama-server.exe`, so both get it. "CUDA - Force P2 state" is not set.
- **It does not act on CUDA alone.** A CUDA load of 30 ms bursts with 70 ms pauses on both cards
  (`cuMemsetD8` of 1 GiB in a loop) let both fall to P3 at 390-990 MHz and P8, reason "GPU idle", the
  2080 Ti included. So did llama-server (CUDA) with Qwen3-8B split over the two cards and 14 layers on
  the CPU, so that both cards wait for the CPU: P5 at 810 MHz memory most of the time, 5.4-5.6 tok/s.
- **It acts on a card that a graphics API has open.** With llama.cpp's Vulkan build loaded on the
  SUPER and idle, the SUPER sat at P0 (1650 MHz, memory 7750 MHz) at 0% load, reason none. A program
  that only creates a Vulkan instance and a logical device on the SUPER (7 MiB of VRAM, no work) did
  the same; with it running, the bursty CUDA load above kept the SUPER at P2 (7500 MHz memory) or P0
  in every sample, while the 2080 Ti fell to P3/P8. When the program ends the SUPER returns to P8.
- So the 2080 Ti's clocks that morning came from a program on the desktop drawing on it, not from the
  model. In the evening, with nothing drawing, the Ti clocked down like the SUPER.

Power while held and idle: about 50 W on the SUPER instead of 12 W.

## The engine's change

`up` now holds a Vulkan device open on each card of this PC that the plan uses, when the model is
split (over several cards, with CPU layers, or with another PC's card), from before llama-server
starts until it stops (`cardhold.py`). `--no-hold-clocks` turns it off. When the power management mode
is not "Prefer maximum performance", `up` says so: holding the cards then does not keep their clocks.

Check with the two E5 cards and 14 CPU layers (Qwen3-8B Q4_K_M, the case above), two answers of 400
tokens each, held by a separate program:

| | tok/s | SUPER | 2080 Ti |
|---|---|---|---|
| Not held | 5.6, 5.4 | P5/P8 in all but one sample | P5/P8 mostly |
| Both held | 10.0, 9.8 | P2 in every sample | P2/P3 about three quarters, P5/P8 the rest |

Another program (the owner's own `up` from the app) was loading at the time of both runs, so the
speeds are indicative. Why the 2080 Ti, which drives the display, still clocks down part of the time
while held is not known.

## Three cards: gpt-oss-120b through `up` with this branch

The same evening, nothing else running on the E5: `up --model gpt-oss-120b --remote auto` (SUPER 0-3,
4090 on the other site 4-16, Ti 17-35 and the head; the plan of phase2c), three answers of 400 tokens
each as in phase2c, then the same with `--no-hold-clocks`. `up` said
`note: keeping CUDA0, CUDA1 at full clocks while the model runs`.

| Run | tok/s | SUPER while busy | 2080 Ti while busy |
|---|---|---|---|
| Held (this branch) | 15.51, 15.36, 15.29 | P2 in every sample | mostly P3/P8 |
| `--no-hold-clocks` | 10.85, 10.10, 10.22 | P8 in nine samples of ten | P3-P8 mostly |
| The morning (phase2c), Ti held by the desktop | 12.98-13.67 | P8 | P2 |

The estimate for this placement is 12.6 tok/s. Holding the SUPER is worth about half again over
nothing held; the 2080 Ti still clocks down while held, so some speed is left on it.
