# Phase 0 baseline: E5 rig, RTX 2080 Ti + RTX 2080 SUPER (2026-11-14)

llama.cpp numbers that GPUEnsemble has to beat on the development rig.
Raw logs stay on the rig (`C:\llm\phase0\logs\`); this file is the summary.

## Hardware

| Part | Detail |
|---|---|
| CPU | 2x Xeon E5-2680 v1 (Sandy Bridge-EP, AVX only), 16C/32T |
| RAM | 64 GB DDR3-1600, 8 channels (theoretical ~102 GB/s) |
| GPU0 | RTX 2080 Ti 11 GB, CPU0 side, PCIe Gen2 x8, drives the display |
| GPU1 | RTX 2080 SUPER 8 GB, CPU1 side via riser, PCIe Gen2 x4 |
| Driver / OS | NVIDIA 528.49 (CUDA 12.0) for the first runs; Windows Update moved it to 560.94 (CUDA 12.6) the same day. WDDM, Windows 10 Pro 19044 |

Links stayed at Gen2 x8 / Gen2 x4 under load (sampled every second).

## llama.cpp builds

| Build | Result |
|---|---|
| b12375 win-cuda-12.4 | **Fails** on driver 528 (`invalid device function`: PTX newer than the driver can JIT) |
| b5309 win-cuda-cu11.7 | Works. Last release with a cu11.7 Windows build. ~10 min PTX JIT on first start. Too old for gpt-oss |
| b12375 win-vulkan-x64 | Works. Used for gpt-oss |
| b12375 win-cpu-x64 | Works (sandybridge backend) |

Every b12375 exe crashed at start (0xC0000005 in MSVCP140.dll 14.28, std::mutex ABI) until
newer VC++ runtime DLLs (14.51) were placed next to the exes. The CUDA JIT cache needed
`CUDA_CACHE_MAXSIZE=4GB`; the default 256 MB filled up and recompiled on every start.

All runs: `llama-bench -p 512 -n 128 -r 3 -fa 1`. Numbers are tokens/s, pp512 / tg128.

## Qwen3-14B Q4_K_M (8.38 GiB)

| Backend | Config | pp512 | tg128 |
|---|---|---:|---:|
| CUDA 11.7 | 2080 Ti only | 1490 | **46.0** |
| CUDA 11.7 | 2080 SUPER only, ngl 38 | 120 | 17.9 |
| CUDA 11.7 | both, layer split, ts 11/8 | 1322 | 40.4 |
| CUDA 11.7 | both, layer split, ts 1/1 | 1298 | 39.2 |
| CUDA 11.7 | both, row split, ts 11/8 | 157 | 6.4 |
| Vulkan | 2080 Ti only | 783 | 42.0 |
| Vulkan | both, layer split, ts 11/8 | 844 | 38.9 |
| CPU | 16 threads | 14.7 | 4.26 |
| CPU | 32 threads | 17.7 | 4.09 |

## Qwen3-32B Q4_K_M (18.40 GiB, 64 layers), both GPUs, layer split

| Backend | Config | pp512 | tg128 |
|---|---|---:|---:|
| CUDA 11.7 | ngl 58, ts 11/8 (most that fits) | 212 | **4.59** |
| CUDA 11.7 | ngl 54, ts 11/8 | 185 | 3.90 |
| CUDA 11.7 | ngl 50, ts 11/8 | 162 | 3.49 |
| Vulkan | ngl 58, ts 11/8 | 159 | 4.47 |

ngl 62 and above run out of memory for every ts.

## gpt-oss-20b MXFP4 (11.27 GiB), Vulkan

| Config | pp512 | tg128 |
|---|---:|---:|
| both GPUs | 729 | **71.6** |
| 2080 Ti only, -ncmoe 2 | 417 | 59.8 |
| 2080 Ti only, -ncmoe 4 | 281 | 42.8 |

## gpt-oss-120b MXFP4 (59.02 GiB, 36 layers), Vulkan, ngl 99

| Config | pp512 | tg128 |
|---|---:|---:|
| -ncmoe 36, 32 threads | 17.2 | 3.96 |
| -ncmoe 30, ts 33/3, 32 threads | 20.0 | 4.67 |
| -ncmoe 28, ts 32/4, 32 threads | 21.2 | **4.82** |
| -ncmoe 28, ts 32/4, numa distribute, 32 threads | 27.5 | 4.81 |

With the default ts, every -ncmoe below 36 ran out of memory on the 2080 SUPER: the
GPU-resident expert layers are the last layers, so they all landed on GPU1. A hand-skewed
`-ts` was needed. This is exactly the manual tuning the planner should remove.

## CUDA 12.4 build on driver 560.94 (current baseline)

Windows Update installed NVIDIA 560.94 during the runs above. With it the b12375
CUDA 12.4 build works and is the fastest backend for every model. These are the numbers
to beat. Same settings; the best earlier result is in brackets.

| Model | Config | pp512 | tg128 |
|---|---|---:|---:|
| Qwen3-14B | 2080 Ti only | 1452 | **51.5** (46.0) |
| Qwen3-14B | both, layer split, ts 11/8 | 1377 | 46.1 (40.4) |
| Qwen3-32B | ngl 58, ts 11/8 | 247 | **5.66** (4.59) |
| gpt-oss-20b | both GPUs | 1595 | **100.1** (71.6 Vulkan) |
| gpt-oss-120b | -ncmoe 28, ts 32/4, 32 threads | 20.7 | **5.14** (4.82 Vulkan) |
| Qwen3-14B | CUDA 11.7 b5309, 2080 Ti only (retest) | 1472 | 44.1 |

PCIe links stayed at Gen2 x8 / Gen2 x4 under load. Raw logs: `C:\llm\phase0\logs\drv560_*`.

## What this means for GPUEnsemble

1. **Require a recent driver and build for CUDA 12.** Driver 528 could only run CUDA 11.x
   builds; on 560 the CUDA 12.4 build is 12-40% faster than anything else. The engine targets
   CUDA 12.4 and asks for driver >= 551. The installer must bundle the VC++ runtime.
2. **A second card helps capacity, not speed.** Qwen3-14B is 51.5 on the 2080 Ti alone and 46.1
   on both. Row split collapses to 6 t/s (Gen2 links, two sockets, no P2P under WDDM), which
   confirms layer split as the only sensible mode here.
3. **The CPU is the wall.** Effective CPU bandwidth is ~36 GiB/s for dense layers and only
   ~9 GB/s (Vulkan) to ~13 GB/s (CUDA 12.4) for MXFP4 experts (compute-bound without AVX2). Anything over 19 GB of VRAM lands at
   3.5-5 t/s.
4. **Fixed cost of hybrid runs.** As soon as any layer runs on the CPU, tg drops by ~75-110 ms per
   token beyond what bandwidth explains (Qwen3-32B: 8 more layers on GPU only saved 68 ms).
5. **PCIe streaming of experts is not worth it on this rig.** At Gen2 x8 / x4 (~3.2 / 1.6 GB/s)
   streaming active experts is slower than even the slow CPU path, except on cards with a
   faster link.

## Planner calibration

`planner/examples/rig-e5-turing.json` now holds effective values fitted to these runs:
2080 Ti 463 GB/s, 2080 SUPER 362 GB/s (~75% of the published peak; fitted from the 14B
single-card and split runs on CUDA 12.4), CPU 34 GB/s dense, 13 GB/s experts, 75 ms CPU overhead.

| Model | Measured tg128 (CUDA 12.4) | Planner (ctx 640) | Planner placement |
|---|---:|---:|---|
| Qwen3-32B Q4_K_M | 5.66 | 5.9 | 6 layers on CPU (llama.cpp best: ngl 58 = 6-7 on CPU) |
| gpt-oss-120b | 5.14 | 5.2 | all dense parts on GPUs, ~27 layers of experts on CPU (llama.cpp best: -ncmoe 28) |

The planner lands within +20% and picks the same split that took manual trial and error.
`tests/test_plan.py::Phase0CalibrationTest` keeps it there.

Not yet done: host-to-device bandwidth (no `bandwidthTest` on the rig), and running
`gpuensemble-plan describe` on the real GGUF files (no Python on the rig yet).
