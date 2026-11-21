# GPU bandwidth table

The planner times each stage by how fast its GPU reads its own memory. This is the table it
uses for detected cards (`src/gpuensemble_planner/gpu_table.py`; `python -m gpuensemble_planner gpus`
prints it). "spec x 0.75" rows are predictions from the published bandwidth; "measured" rows were
measured with llama.cpp and are used as they are. A rig file's `bandwidth_gbs` overrides both.

AMD Radeon and Intel Arc cards are run through llama.cpp's Vulkan build. Their rows are the
published figures times 0.75, the share fitted on NVIDIA cards with CUDA; none has been measured
yet, so their estimates are less sure than the NVIDIA ones.

Cards not in the table are planned at 300 GB/s and their name gets "(bandwidth guessed)".
A notebook card ("Laptop GPU", "Max-Q") never takes a desktop card's figure. Where one name covers
several memory sizes, the detected size picks the row.

To add a measurement: run a model that fits on the card alone with `llama-bench -n 128`, divide
the bytes read per token (the model's weights) by the time per token, and set `measured_gbs` and
`measured_by` on the card's row, then regenerate this file with
`python -m gpuensemble_planner gpus` (a test checks that it matches).

| Card | Kind | Architecture | Memory | Published GB/s | Planner GB/s | Source |
|---|---|---|---|---|---|---|
| rtx 5090 d v2 | desktop | Blackwell | 24 GB | 1344 | 1008 | spec x 0.75 |
| rtx 5090 d | desktop | Blackwell | 32 GB | 1792 | 1344 | spec x 0.75 |
| rtx 5090 | desktop | Blackwell | 32 GB | 1792 | 1344 | spec x 0.75 |
| rtx 5080 | desktop | Blackwell | 16 GB | 960 | 720 | spec x 0.75 |
| rtx 5070 ti | desktop | Blackwell | 16 GB | 896 | 672 | spec x 0.75 |
| rtx 5070 | desktop | Blackwell | 12 GB | 672 | 504 | spec x 0.75 |
| rtx 5060 ti | desktop | Blackwell | 16 GB | 448 | 336 | spec x 0.75 |
| rtx 5060 ti | desktop | Blackwell | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 5060 | desktop | Blackwell | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 5050 | desktop | Blackwell | 8 GB | 320 | 240 | spec x 0.75 |
| rtx 4090 d | desktop | Ada | 24 GB | 1008 | 756 | spec x 0.75 |
| rtx 4090 | desktop | Ada | 24 GB | 1008 | 820 | measured: RTX 4090 PC (Windows 11, CUDA 12.4, llama.cpp b12375), tg with Qwen3-32B, 2026-11-16 |
| rtx 4080 super | desktop | Ada | 16 GB | 736 | 552 | spec x 0.75 |
| rtx 4080 | desktop | Ada | 16 GB | 717 | 538 | spec x 0.75 |
| rtx 4070 ti super | desktop | Ada | 16 GB | 672 | 504 | spec x 0.75 |
| rtx 4070 ti | desktop | Ada | 12 GB | 504 | 378 | spec x 0.75 |
| rtx 4070 super | desktop | Ada | 12 GB | 504 | 378 | spec x 0.75 |
| rtx 4070 | desktop | Ada | 12 GB | 504 | 378 | spec x 0.75 |
| rtx 4060 ti | desktop | Ada | 8 GB | 288 | 216 | spec x 0.75 |
| rtx 4060 ti | desktop | Ada | 16 GB | 288 | 216 | spec x 0.75 |
| rtx 4060 | desktop | Ada | 8 GB | 272 | 204 | spec x 0.75 |
| rtx 3090 ti | desktop | Ampere | 24 GB | 1008 | 756 | spec x 0.75 |
| rtx 3090 | desktop | Ampere | 24 GB | 936 | 702 | spec x 0.75 |
| rtx 3080 ti | desktop | Ampere | 12 GB | 912 | 684 | spec x 0.75 |
| rtx 3080 | desktop | Ampere | 10 GB | 760 | 570 | spec x 0.75 |
| rtx 3080 | desktop | Ampere | 12 GB | 912 | 684 | spec x 0.75 |
| rtx 3070 ti | desktop | Ampere | 8 GB | 608 | 456 | spec x 0.75 |
| rtx 3070 | desktop | Ampere | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 3060 ti | desktop | Ampere | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 3060 | desktop | Ampere | 12 GB | 360 | 270 | spec x 0.75 |
| rtx 3060 | desktop | Ampere | 8 GB | 240 | 180 | spec x 0.75 |
| rtx 3050 | desktop | Ampere | 8 GB | 224 | 168 | spec x 0.75 |
| rtx 3050 | desktop | Ampere | 6 GB | 168 | 126 | spec x 0.75 |
| titan rtx | desktop | Turing | 24 GB | 672 | 504 | spec x 0.75 |
| rtx 2080 ti | desktop | Turing | 11 GB | 616 | 463 | measured: E5 rig (Windows, CUDA 12.4, llama.cpp b12375), tg with Qwen3-14B, 2026-11-14 |
| rtx 2080 super | desktop | Turing | 8 GB | 496 | 362 | measured: E5 rig (Windows, CUDA 12.4, llama.cpp b12375), tg with Qwen3-14B, 2026-11-14 |
| rtx 2080 | desktop | Turing | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 2070 super | desktop | Turing | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 2070 | desktop | Turing | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 2060 super | desktop | Turing | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 2060 | desktop | Turing | 6 GB | 336 | 252 | spec x 0.75 |
| rtx 2060 | desktop | Turing | 12 GB | 336 | 252 | spec x 0.75 |
| gtx 1660 ti | desktop | Turing | 6 GB | 288 | 216 | spec x 0.75 |
| gtx 1660 super | desktop | Turing | 6 GB | 336 | 252 | spec x 0.75 |
| gtx 1660 | desktop | Turing | 6 GB | 192 | 144 | spec x 0.75 |
| gtx 1650 super | desktop | Turing | 4 GB | 192 | 144 | spec x 0.75 |
| gtx 1650 | desktop | Turing | 4 GB | 128 | 96 | spec x 0.75 |
| gtx 1630 | desktop | Turing | 4 GB | 96 | 72 | spec x 0.75 |
| titan v | desktop | Volta | 12 GB | 653 | 490 | spec x 0.75 |
| titan xp | desktop | Pascal | 12 GB | 548 | 411 | spec x 0.75 |
| titan x (pascal) | desktop | Pascal | 12 GB | 480 | 360 | spec x 0.75 |
| gtx 1080 ti | desktop | Pascal | 11 GB | 484 | 363 | spec x 0.75 |
| gtx 1080 | desktop | Pascal | 8 GB | 320 | 240 | spec x 0.75 |
| gtx 1070 ti | desktop | Pascal | 8 GB | 256 | 192 | spec x 0.75 |
| gtx 1070 | desktop | Pascal | 8 GB | 256 | 192 | spec x 0.75 |
| gtx 1060 | desktop | Pascal | 6 GB | 192 | 144 | spec x 0.75 |
| gtx 1060 | desktop | Pascal | 3 GB | 192 | 144 | spec x 0.75 |
| gtx 1060 | desktop | Pascal | 5 GB | 160 | 120 | spec x 0.75 |
| gtx 1050 ti | desktop | Pascal | 4 GB | 112 | 84 | spec x 0.75 |
| gtx 1050 | desktop | Pascal | 2 GB | 112 | 84 | spec x 0.75 |
| gtx 1050 | desktop | Pascal | 3 GB | 84 | 63 | spec x 0.75 |
| gt 1030 | desktop | Pascal | 2 GB | 48 | 36 | spec x 0.75 |
| gtx titan x | desktop | Maxwell | 12 GB | 336 | 252 | spec x 0.75 |
| gtx 980 ti | desktop | Maxwell | 6 GB | 336 | 252 | spec x 0.75 |
| gtx 980 | desktop | Maxwell | 4 GB | 224 | 168 | spec x 0.75 |
| gtx 970 | desktop | Maxwell | 4 GB | 224 | 168 | spec x 0.75 |
| gtx 960 | desktop | Maxwell | 2 GB | 112 | 84 | spec x 0.75 |
| gtx 960 | desktop | Maxwell | 4 GB | 112 | 84 | spec x 0.75 |
| rtx 5090 laptop | laptop | Blackwell | 24 GB | 896 | 672 | spec x 0.75 |
| rtx 5080 laptop | laptop | Blackwell | 16 GB | 896 | 672 | spec x 0.75 |
| rtx 5070 ti laptop | laptop | Blackwell | 12 GB | 672 | 504 | spec x 0.75 |
| rtx 5070 laptop | laptop | Blackwell | 8 GB | 384 | 288 | spec x 0.75 |
| rtx 5060 laptop | laptop | Blackwell | 8 GB | 384 | 288 | spec x 0.75 |
| rtx 5050 laptop | laptop | Blackwell | 8 GB | 384 | 288 | spec x 0.75 |
| rtx 4090 laptop | laptop | Ada | 16 GB | 576 | 432 | spec x 0.75 |
| rtx 4080 laptop | laptop | Ada | 12 GB | 432 | 324 | spec x 0.75 |
| rtx 4070 laptop | laptop | Ada | 8 GB | 256 | 192 | spec x 0.75 |
| rtx 4060 laptop | laptop | Ada | 8 GB | 256 | 192 | spec x 0.75 |
| rtx 4050 laptop | laptop | Ada | 6 GB | 192 | 144 | spec x 0.75 |
| rtx 3080 ti laptop | laptop | Ampere | 16 GB | 512 | 384 | spec x 0.75 |
| rtx 3080 laptop | laptop | Ampere | 16 GB | 448 | 336 | spec x 0.75 |
| rtx 3080 laptop | laptop | Ampere | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 3070 ti laptop | laptop | Ampere | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 3070 laptop | laptop | Ampere | 8 GB | 448 | 336 | spec x 0.75 |
| rtx 3060 laptop | laptop | Ampere | 6 GB | 336 | 252 | spec x 0.75 |
| rtx 3050 ti laptop | laptop | Ampere | 4 GB | 192 | 144 | spec x 0.75 |
| rtx 3050 laptop | laptop | Ampere | 4 GB | 192 | 144 | spec x 0.75 |
| rtx 3050 laptop | laptop | Ampere | 6 GB | 168 | 126 | spec x 0.75 |
| rtx 2050 | laptop | Ampere | 4 GB | 112 | 84 | spec x 0.75 |
| rtx pro 5000 blackwell laptop | laptop | Blackwell | 24 GB | 896 | 672 | spec x 0.75 |
| rtx pro 4000 blackwell laptop | laptop | Blackwell | 16 GB | 896 | 672 | spec x 0.75 |
| rtx pro 3000 blackwell laptop | laptop | Blackwell | 12 GB | 672 | 504 | spec x 0.75 |
| rtx pro 2000 blackwell laptop | laptop | Blackwell | 8 GB | 448 | 336 | spec x 0.75 |
| rtx pro 1000 blackwell laptop | laptop | Blackwell | 8 GB | 384 | 288 | spec x 0.75 |
| rtx pro 500 blackwell laptop | laptop | Blackwell | 6 GB | 288 | 216 | spec x 0.75 |
| rtx 5000 ada laptop | laptop | Ada | 16 GB | 576 | 432 | spec x 0.75 |
| rtx 4000 ada laptop | laptop | Ada | 12 GB | 432 | 324 | spec x 0.75 |
| rtx 3500 ada laptop | laptop | Ada | 12 GB | 432 | 324 | spec x 0.75 |
| rtx 3000 ada laptop | laptop | Ada | 8 GB | 256 | 192 | spec x 0.75 |
| rtx 2000 ada laptop | laptop | Ada | 8 GB | 256 | 192 | spec x 0.75 |
| rtx 1000 ada laptop | laptop | Ada | 6 GB | 192 | 144 | spec x 0.75 |
| rtx 500 ada laptop | laptop | Ada | 4 GB | 128 | 96 | spec x 0.75 |
| rtx a5500 laptop | laptop | Ampere | 16 GB | 512 | 384 | spec x 0.75 |
| rtx a5000 laptop | laptop | Ampere | 16 GB | 448 | 336 | spec x 0.75 |
| rtx a4000 laptop | laptop | Ampere | 8 GB | 384 | 288 | spec x 0.75 |
| rtx a3000 laptop | laptop | Ampere | 6 GB | 264 | 198 | spec x 0.75 |
| rtx a2000 laptop | laptop | Ampere | 4 GB | 224 | 168 | spec x 0.75 |
| rtx a2000 laptop | laptop | Ampere | 8 GB | 224 | 168 | spec x 0.75 |
| rtx a1000 laptop | laptop | Ampere | 4 GB | 224 | 168 | spec x 0.75 |
| rtx a500 laptop | laptop | Ampere | 4 GB | 112 | 84 | spec x 0.75 |
| rtx 2080 super with max-q | laptop | Turing | 8 GB | 352 | 264 | spec x 0.75 |
| rtx 2080 with max-q | laptop | Turing | 8 GB | 384 | 288 | spec x 0.75 |
| rtx 2070 super with max-q | laptop | Turing | 8 GB | 352 | 264 | spec x 0.75 |
| rtx 2070 with max-q | laptop | Turing | 8 GB | 384 | 288 | spec x 0.75 |
| rtx 2060 with max-q | laptop | Turing | 6 GB | 264 | 198 | spec x 0.75 |
| gtx 1660 ti with max-q | laptop | Turing | 6 GB | 288 | 216 | spec x 0.75 |
| gtx 1650 with max-q | laptop | Turing | 4 GB | 128 | 96 | spec x 0.75 |
| gtx 1080 with max-q | laptop | Pascal | 8 GB | 320 | 240 | spec x 0.75 |
| gtx 1070 with max-q | laptop | Pascal | 8 GB | 256 | 192 | spec x 0.75 |
| gtx 1060 with max-q | laptop | Pascal | 6 GB | 192 | 144 | spec x 0.75 |
| rtx pro 6000 blackwell | workstation | Blackwell | 96 GB | 1792 | 1344 | spec x 0.75 |
| rtx pro 5000 blackwell | workstation | Blackwell | 48 GB | 1344 | 1008 | spec x 0.75 |
| rtx pro 4500 blackwell | workstation | Blackwell | 32 GB | 896 | 672 | spec x 0.75 |
| rtx pro 4000 blackwell sff | workstation | Blackwell | 24 GB | 432 | 324 | spec x 0.75 |
| rtx pro 4000 blackwell | workstation | Blackwell | 24 GB | 672 | 504 | spec x 0.75 |
| rtx pro 2000 blackwell | workstation | Blackwell | 16 GB | 288 | 216 | spec x 0.75 |
| rtx 6000 ada | workstation | Ada | 48 GB | 960 | 720 | spec x 0.75 |
| rtx 5880 ada | workstation | Ada | 48 GB | 960 | 720 | spec x 0.75 |
| rtx 5000 ada | workstation | Ada | 32 GB | 576 | 432 | spec x 0.75 |
| rtx 4500 ada | workstation | Ada | 24 GB | 432 | 324 | spec x 0.75 |
| rtx 4000 ada | workstation | Ada | 20 GB | 360 | 270 | spec x 0.75 |
| rtx 4000 sff ada | workstation | Ada | 20 GB | 280 | 210 | spec x 0.75 |
| rtx 2000 ada | workstation | Ada | 16 GB | 224 | 168 | spec x 0.75 |
| rtx a6000 | workstation | Ampere | 48 GB | 768 | 576 | spec x 0.75 |
| rtx a5500 | workstation | Ampere | 24 GB | 768 | 576 | spec x 0.75 |
| rtx a5000 | workstation | Ampere | 24 GB | 768 | 576 | spec x 0.75 |
| rtx a4500 | workstation | Ampere | 20 GB | 640 | 480 | spec x 0.75 |
| rtx a4000 | workstation | Ampere | 16 GB | 448 | 336 | spec x 0.75 |
| rtx a2000 | workstation | Ampere | 6 GB | 288 | 216 | spec x 0.75 |
| rtx a2000 | workstation | Ampere | 12 GB | 288 | 216 | spec x 0.75 |
| rtx a1000 | workstation | Ampere | 8 GB | 192 | 144 | spec x 0.75 |
| rtx a400 | workstation | Ampere | 4 GB | 96 | 72 | spec x 0.75 |
| rtx 8000 | workstation | Turing | 48 GB | 672 | 504 | spec x 0.75 |
| rtx 6000 | workstation | Turing | 24 GB | 672 | 504 | spec x 0.75 |
| rtx 5000 | workstation | Turing | 16 GB | 448 | 336 | spec x 0.75 |
| rtx 4000 | workstation | Turing | 8 GB | 416 | 312 | spec x 0.75 |
| t1000 | workstation | Turing | 4 GB | 160 | 120 | spec x 0.75 |
| t1000 | workstation | Turing | 8 GB | 160 | 120 | spec x 0.75 |
| t600 | workstation | Turing | 4 GB | 160 | 120 | spec x 0.75 |
| t400 | workstation | Turing | 2 GB | 80 | 60 | spec x 0.75 |
| t400 | workstation | Turing | 4 GB | 80 | 60 | spec x 0.75 |
| gv100 | workstation | Volta | 32 GB | 870 | 652 | spec x 0.75 |
| gp100 | workstation | Pascal | 16 GB | 717 | 538 | spec x 0.75 |
| p6000 | workstation | Pascal | 24 GB | 432 | 324 | spec x 0.75 |
| p5000 | workstation | Pascal | 16 GB | 288 | 216 | spec x 0.75 |
| p4000 | workstation | Pascal | 8 GB | 243 | 182 | spec x 0.75 |
| p2200 | workstation | Pascal | 5 GB | 200 | 150 | spec x 0.75 |
| p2000 | workstation | Pascal | 5 GB | 140 | 105 | spec x 0.75 |
| p1000 | workstation | Pascal | 4 GB | 82 | 62 | spec x 0.75 |
| p620 | workstation | Pascal | 2 GB | 80 | 60 | spec x 0.75 |
| p400 | workstation | Pascal | 2 GB | 32 | 24 | spec x 0.75 |
| b200 | datacenter | Blackwell | 180 GB | 8000 | 6000 | spec x 0.75 |
| h200 nvl | datacenter | Hopper | 141 GB | 4800 | 3600 | spec x 0.75 |
| h200 | datacenter | Hopper | 141 GB | 4800 | 3600 | spec x 0.75 |
| h100 nvl | datacenter | Hopper | 94 GB | 3900 | 2925 | spec x 0.75 |
| h100 pcie | datacenter | Hopper | 80 GB | 2000 | 1500 | spec x 0.75 |
| h100 | datacenter | Hopper | 80 GB | 3350 | 2512 | spec x 0.75 |
| h800 pcie | datacenter | Hopper | 80 GB | 2000 | 1500 | spec x 0.75 |
| h800 | datacenter | Hopper | 80 GB | 3350 | 2512 | spec x 0.75 |
| h20 | datacenter | Hopper | 96 GB | 4000 | 3000 | spec x 0.75 |
| l40s | datacenter | Ada | 48 GB | 864 | 648 | spec x 0.75 |
| l40 | datacenter | Ada | 48 GB | 864 | 648 | spec x 0.75 |
| l20 | datacenter | Ada | 48 GB | 864 | 648 | spec x 0.75 |
| l4 | datacenter | Ada | 24 GB | 300 | 225 | spec x 0.75 |
| a100-sxm4 | datacenter | Ampere | 40 GB | 1555 | 1166 | spec x 0.75 |
| a100-sxm4 | datacenter | Ampere | 80 GB | 2039 | 1529 | spec x 0.75 |
| a100 | datacenter | Ampere | 40 GB | 1555 | 1166 | spec x 0.75 |
| a100 | datacenter | Ampere | 80 GB | 1935 | 1451 | spec x 0.75 |
| a800 | datacenter | Ampere | 40 GB | 1555 | 1166 | spec x 0.75 |
| a800 | datacenter | Ampere | 80 GB | 1935 | 1451 | spec x 0.75 |
| a40 | datacenter | Ampere | 48 GB | 696 | 522 | spec x 0.75 |
| a30 | datacenter | Ampere | 24 GB | 933 | 700 | spec x 0.75 |
| a10g | datacenter | Ampere | 24 GB | 600 | 450 | spec x 0.75 |
| a10 | datacenter | Ampere | 24 GB | 600 | 450 | spec x 0.75 |
| a2 | datacenter | Ampere | 16 GB | 200 | 150 | spec x 0.75 |
| t4 | datacenter | Turing | 16 GB | 320 | 240 | spec x 0.75 |
| v100s | datacenter | Volta | 32 GB | 1134 | 850 | spec x 0.75 |
| v100 | datacenter | Volta | 16 GB | 900 | 675 | spec x 0.75 |
| v100 | datacenter | Volta | 32 GB | 900 | 675 | spec x 0.75 |
| p100 | datacenter | Pascal | 16 GB | 732 | 549 | spec x 0.75 |
| p100 | datacenter | Pascal | 12 GB | 549 | 412 | spec x 0.75 |
| p40 | datacenter | Pascal | 24 GB | 347 | 260 | spec x 0.75 |
| p4 | datacenter | Pascal | 8 GB | 192 | 144 | spec x 0.75 |
| m40 | datacenter | Maxwell | 12 GB | 288 | 216 | spec x 0.75 |
| m40 | datacenter | Maxwell | 24 GB | 288 | 216 | spec x 0.75 |
| radeon rx 9070 xt | desktop | RDNA 4 | 16 GB | 640 | 480 | spec x 0.75 |
| radeon rx 9070 gre | desktop | RDNA 4 | 12 GB | 432 | 324 | spec x 0.75 |
| radeon rx 9070 | desktop | RDNA 4 | 16 GB | 640 | 480 | spec x 0.75 |
| radeon rx 9060 xt | desktop | RDNA 4 | 16 GB | 320 | 240 | spec x 0.75 |
| radeon rx 9060 xt | desktop | RDNA 4 | 8 GB | 320 | 240 | spec x 0.75 |
| radeon rx 7900 xtx | desktop | RDNA 3 | 24 GB | 960 | 720 | spec x 0.75 |
| radeon rx 7900 xt | desktop | RDNA 3 | 20 GB | 800 | 600 | spec x 0.75 |
| radeon rx 7900 gre | desktop | RDNA 3 | 16 GB | 576 | 432 | spec x 0.75 |
| radeon rx 7800 xt | desktop | RDNA 3 | 16 GB | 624 | 468 | spec x 0.75 |
| radeon rx 7700 xt | desktop | RDNA 3 | 12 GB | 432 | 324 | spec x 0.75 |
| radeon rx 7600 xt | desktop | RDNA 3 | 16 GB | 288 | 216 | spec x 0.75 |
| radeon rx 7600 | desktop | RDNA 3 | 8 GB | 288 | 216 | spec x 0.75 |
| radeon rx 6950 xt | desktop | RDNA 2 | 16 GB | 576 | 432 | spec x 0.75 |
| radeon rx 6900 xt | desktop | RDNA 2 | 16 GB | 512 | 384 | spec x 0.75 |
| radeon rx 6800 xt | desktop | RDNA 2 | 16 GB | 512 | 384 | spec x 0.75 |
| radeon rx 6800 | desktop | RDNA 2 | 16 GB | 512 | 384 | spec x 0.75 |
| radeon rx 6750 xt | desktop | RDNA 2 | 12 GB | 432 | 324 | spec x 0.75 |
| radeon rx 6700 xt | desktop | RDNA 2 | 12 GB | 384 | 288 | spec x 0.75 |
| radeon rx 6700 | desktop | RDNA 2 | 10 GB | 320 | 240 | spec x 0.75 |
| radeon rx 6650 xt | desktop | RDNA 2 | 8 GB | 280 | 210 | spec x 0.75 |
| radeon rx 6600 xt | desktop | RDNA 2 | 8 GB | 256 | 192 | spec x 0.75 |
| radeon rx 6600 | desktop | RDNA 2 | 8 GB | 224 | 168 | spec x 0.75 |
| radeon rx 6500 xt | desktop | RDNA 2 | 4 GB | 144 | 108 | spec x 0.75 |
| radeon rx 5700 xt | desktop | RDNA 1 | 8 GB | 448 | 336 | spec x 0.75 |
| radeon rx 5700 | desktop | RDNA 1 | 8 GB | 448 | 336 | spec x 0.75 |
| radeon rx 5600 xt | desktop | RDNA 1 | 6 GB | 336 | 252 | spec x 0.75 |
| radeon rx 5500 xt | desktop | RDNA 1 | 8 GB | 224 | 168 | spec x 0.75 |
| radeon rx 5500 xt | desktop | RDNA 1 | 4 GB | 224 | 168 | spec x 0.75 |
| radeon vii | desktop | GCN 5 | 16 GB | 1024 | 768 | spec x 0.75 |
| radeon rx vega 64 | desktop | GCN 5 | 8 GB | 484 | 363 | spec x 0.75 |
| radeon rx vega 56 | desktop | GCN 5 | 8 GB | 410 | 308 | spec x 0.75 |
| radeon rx 590 | desktop | GCN 4 | 8 GB | 256 | 192 | spec x 0.75 |
| radeon rx 580 | desktop | GCN 4 | 8 GB | 256 | 192 | spec x 0.75 |
| radeon rx 580 | desktop | GCN 4 | 4 GB | 224 | 168 | spec x 0.75 |
| radeon rx 570 | desktop | GCN 4 | 8 GB | 224 | 168 | spec x 0.75 |
| radeon rx 570 | desktop | GCN 4 | 4 GB | 224 | 168 | spec x 0.75 |
| radeon rx 480 | desktop | GCN 4 | 8 GB | 256 | 192 | spec x 0.75 |
| radeon rx 7600m xt | laptop | RDNA 3 | 8 GB | 288 | 216 | spec x 0.75 |
| radeon rx 7600m | laptop | RDNA 3 | 8 GB | 256 | 192 | spec x 0.75 |
| radeon rx 6800m | laptop | RDNA 2 | 12 GB | 384 | 288 | spec x 0.75 |
| radeon rx 6700m | laptop | RDNA 2 | 10 GB | 320 | 240 | spec x 0.75 |
| radeon ai pro r9700 | workstation | RDNA 4 | 32 GB | 640 | 480 | spec x 0.75 |
| radeon pro w7900 | workstation | RDNA 3 | 48 GB | 864 | 648 | spec x 0.75 |
| radeon pro w7800 | workstation | RDNA 3 | 32 GB | 576 | 432 | spec x 0.75 |
| radeon pro w7700 | workstation | RDNA 3 | 16 GB | 576 | 432 | spec x 0.75 |
| radeon pro w7600 | workstation | RDNA 3 | 8 GB | 288 | 216 | spec x 0.75 |
| radeon pro w6800 | workstation | RDNA 2 | 32 GB | 512 | 384 | spec x 0.75 |
| radeon pro w6600 | workstation | RDNA 2 | 8 GB | 224 | 168 | spec x 0.75 |
| radeon pro vii | workstation | GCN 5 | 16 GB | 1024 | 768 | spec x 0.75 |
| mi300x | datacenter | CDNA | 192 GB | 5300 | 3975 | spec x 0.75 |
| mi250x | datacenter | CDNA | 128 GB | 3277 | 2458 | spec x 0.75 |
| mi210 | datacenter | CDNA | 64 GB | 1638 | 1228 | spec x 0.75 |
| mi100 | datacenter | CDNA | 32 GB | 1229 | 922 | spec x 0.75 |
| mi60 | datacenter | GCN 5 | 32 GB | 1024 | 768 | spec x 0.75 |
| mi50 | datacenter | GCN 5 | 16 GB | 1024 | 768 | spec x 0.75 |
| mi50 | datacenter | GCN 5 | 32 GB | 1024 | 768 | spec x 0.75 |
| mi25 | datacenter | GCN 5 | 16 GB | 484 | 363 | spec x 0.75 |
| arc b580 | desktop | Xe2 | 12 GB | 456 | 342 | spec x 0.75 |
| arc b570 | desktop | Xe2 | 10 GB | 380 | 285 | spec x 0.75 |
| arc a770 | desktop | Xe | 16 GB | 560 | 420 | spec x 0.75 |
| arc a770 | desktop | Xe | 8 GB | 512 | 384 | spec x 0.75 |
| arc a750 | desktop | Xe | 8 GB | 512 | 384 | spec x 0.75 |
| arc a580 | desktop | Xe | 8 GB | 512 | 384 | spec x 0.75 |
| arc a380 | desktop | Xe | 6 GB | 186 | 140 | spec x 0.75 |
| arc a310 | desktop | Xe | 4 GB | 124 | 93 | spec x 0.75 |
| arc a770m | laptop | Xe | 16 GB | 512 | 384 | spec x 0.75 |
| arc a730m | laptop | Xe | 12 GB | 336 | 252 | spec x 0.75 |
| arc a550m | laptop | Xe | 8 GB | 224 | 168 | spec x 0.75 |
| arc a370m | laptop | Xe | 4 GB | 112 | 84 | spec x 0.75 |
| arc pro b60 | workstation | Xe2 | 24 GB | 456 | 342 | spec x 0.75 |
| arc pro b50 | workstation | Xe2 | 16 GB | 224 | 168 | spec x 0.75 |
| arc pro a60 | workstation | Xe | 12 GB | 384 | 288 | spec x 0.75 |
| arc pro a40 | workstation | Xe | 6 GB | 192 | 144 | spec x 0.75 |
