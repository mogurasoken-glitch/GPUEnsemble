"""How fast each GPU reads its own memory, for the cards the planner may meet on someone's PC.

Generating a token reads every weight of the layers on a card once, so how long a stage takes
is set by the card's memory bandwidth. Most entries are the published peak bandwidth (memory
bus width times data rate, as on NVIDIA's spec pages), and the planner expects
``GPU_EFFICIENCY`` of it. A few cards were measured with llama.cpp; for those the measured
figure is used as it is. To add a measurement, set ``measured_gbs`` on the card's entry and
say where it came from in ``measured_by``. ``python -m gpuensemble_planner gpus`` lists the table.

Names are matched against what nvidia-smi (or, for AMD and Intel cards, llama.cpp's Vulkan
build) reports, with "NVIDIA", "GeForce", "Quadro", "Tesla", "AMD", "Intel", "Instinct",
"Graphics", "(R)", "(TM)", "Generation" and memory sizes such as "6GB" taken off, as whole words and longest key first: "rtx 4070 ti super" wins over
"rtx 4070 ti", and "a10" does not match "a100". A key with several entries has one per memory
size (the RTX 3060 has 12 GB at 360 GB/s and 8 GB at 240 GB/s under one name); the card's
memory size picks one, and without it the first entry is used. A notebook card ("Laptop GPU")
is never given a desktop card's figure: its memory usually runs slower.

Not told apart, because nvidia-smi gives them the same name and size: the RTX 4070 with GDDR6
(480 GB/s, table 504), the RTX 3060 Ti with GDDR6X (608, table 448), the GTX 1650 with GDDR6
(192, table 128) and the GTX 1080 with 11 Gb/s memory (352, table 320). Notebooks of the GTX 10
and RTX 20 series carry the desktop names; their memory is about as fast as the desktop card's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Share of the published bandwidth that llama.cpp-style kernels reach when generating.
# Fitted on the E5 rig: 2080 Ti 463 of 616 GB/s, 2080 SUPER 362 of 496.
GPU_EFFICIENCY = 0.75


@dataclass(frozen=True)
class GpuSpec:
    key: str  # lower case, as found in the cleaned nvidia-smi name
    gbs: float  # published peak memory bandwidth, GB/s (10^9 bytes/s)
    vram_gb: float  # memory size of this entry, in GB as the maker counts it
    kind: str  # desktop, laptop, workstation or datacenter
    arch: str
    measured_gbs: float | None = None  # read bandwidth llama.cpp reached when generating, GB/s
    measured_by: str = ""

    @property
    def effective_gbs(self) -> float:
        """What the planner counts on: the measured figure, else the published one times the efficiency."""
        return self.measured_gbs if self.measured_gbs is not None else self.gbs * GPU_EFFICIENCY

    @property
    def source(self) -> str:
        return "measured" if self.measured_gbs is not None else "spec"


def _group(kind: str, arch: str, rows: list[tuple]) -> list[GpuSpec]:
    return [GpuSpec(row[0], float(row[1]), float(row[2]), kind, arch, *row[3:]) for row in rows]


_E5 = "E5 rig (Windows, CUDA 12.4, llama.cpp b12375), tg with Qwen3-14B, 2026-11-14"

# Qwen3-32B Q4_K_M, all 64 layers and the head on the card (18.0 GiB read per token), tg128 42.55 t/s.
_PC4090 = "RTX 4090 PC (Windows 11, CUDA 12.4, llama.cpp b12375), tg with Qwen3-32B, 2026-11-16"

GPUS: list[GpuSpec] = [
    # GeForce, desktop. Entries for the same name with other memory follow the usual one.
    *_group("desktop", "Blackwell", [
        ("rtx 5090 d v2", 1344, 24), ("rtx 5090 d", 1792, 32), ("rtx 5090", 1792, 32), ("rtx 5080", 960, 16), ("rtx 5070 ti", 896, 16), ("rtx 5070", 672, 12),
        ("rtx 5060 ti", 448, 16), ("rtx 5060 ti", 448, 8), ("rtx 5060", 448, 8), ("rtx 5050", 320, 8),
    ]),
    *_group("desktop", "Ada", [
        ("rtx 4090 d", 1008, 24), ("rtx 4090", 1008, 24, 820, _PC4090), ("rtx 4080 super", 736, 16), ("rtx 4080", 717, 16),
        ("rtx 4070 ti super", 672, 16), ("rtx 4070 ti", 504, 12), ("rtx 4070 super", 504, 12),
        ("rtx 4070", 504, 12), ("rtx 4060 ti", 288, 8), ("rtx 4060 ti", 288, 16), ("rtx 4060", 272, 8),
    ]),
    *_group("desktop", "Ampere", [
        ("rtx 3090 ti", 1008, 24), ("rtx 3090", 936, 24), ("rtx 3080 ti", 912, 12),
        ("rtx 3080", 760, 10), ("rtx 3080", 912, 12), ("rtx 3070 ti", 608, 8), ("rtx 3070", 448, 8),
        ("rtx 3060 ti", 448, 8), ("rtx 3060", 360, 12), ("rtx 3060", 240, 8),
        ("rtx 3050", 224, 8), ("rtx 3050", 168, 6),
    ]),
    *_group("desktop", "Turing", [
        ("titan rtx", 672, 24),
        ("rtx 2080 ti", 616, 11, 463, _E5), ("rtx 2080 super", 496, 8, 362, _E5), ("rtx 2080", 448, 8),
        ("rtx 2070 super", 448, 8), ("rtx 2070", 448, 8), ("rtx 2060 super", 448, 8),
        ("rtx 2060", 336, 6), ("rtx 2060", 336, 12),
        ("gtx 1660 ti", 288, 6), ("gtx 1660 super", 336, 6), ("gtx 1660", 192, 6),
        ("gtx 1650 super", 192, 4), ("gtx 1650", 128, 4), ("gtx 1630", 96, 4),
    ]),
    *_group("desktop", "Volta", [("titan v", 653, 12)]),
    *_group("desktop", "Pascal", [
        ("titan xp", 548, 12), ("titan x (pascal)", 480, 12),
        ("gtx 1080 ti", 484, 11), ("gtx 1080", 320, 8), ("gtx 1070 ti", 256, 8), ("gtx 1070", 256, 8),
        ("gtx 1060", 192, 6), ("gtx 1060", 192, 3), ("gtx 1060", 160, 5),
        ("gtx 1050 ti", 112, 4), ("gtx 1050", 112, 2), ("gtx 1050", 84, 3), ("gt 1030", 48, 2),
    ]),
    *_group("desktop", "Maxwell", [
        ("gtx titan x", 336, 12), ("gtx 980 ti", 336, 6), ("gtx 980", 224, 4), ("gtx 970", 224, 4),
        ("gtx 960", 112, 2), ("gtx 960", 112, 4),
    ]),
    # GeForce, notebooks. NVIDIA gives the fastest memory clock; some notebooks run it lower.
    *_group("laptop", "Blackwell", [
        ("rtx 5090 laptop", 896, 24), ("rtx 5080 laptop", 896, 16), ("rtx 5070 ti laptop", 672, 12),
        ("rtx 5070 laptop", 384, 8), ("rtx 5060 laptop", 384, 8), ("rtx 5050 laptop", 384, 8),
    ]),
    *_group("laptop", "Ada", [
        ("rtx 4090 laptop", 576, 16), ("rtx 4080 laptop", 432, 12), ("rtx 4070 laptop", 256, 8),
        ("rtx 4060 laptop", 256, 8), ("rtx 4050 laptop", 192, 6),
    ]),
    *_group("laptop", "Ampere", [
        ("rtx 3080 ti laptop", 512, 16), ("rtx 3080 laptop", 448, 16), ("rtx 3080 laptop", 448, 8),
        ("rtx 3070 ti laptop", 448, 8), ("rtx 3070 laptop", 448, 8), ("rtx 3060 laptop", 336, 6),
        ("rtx 3050 ti laptop", 192, 4), ("rtx 3050 laptop", 192, 4), ("rtx 3050 laptop", 168, 6),
        ("rtx 2050", 112, 4),
    ]),
    # Workstation notebooks ("RTX 3500 Ada Generation Laptop GPU").
    *_group("laptop", "Blackwell", [
        ("rtx pro 5000 blackwell laptop", 896, 24), ("rtx pro 4000 blackwell laptop", 896, 16),
        ("rtx pro 3000 blackwell laptop", 672, 12), ("rtx pro 2000 blackwell laptop", 448, 8),
        ("rtx pro 1000 blackwell laptop", 384, 8), ("rtx pro 500 blackwell laptop", 288, 6),
    ]),
    *_group("laptop", "Ada", [
        ("rtx 5000 ada laptop", 576, 16), ("rtx 4000 ada laptop", 432, 12), ("rtx 3500 ada laptop", 432, 12),
        ("rtx 3000 ada laptop", 256, 8), ("rtx 2000 ada laptop", 256, 8), ("rtx 1000 ada laptop", 192, 6),
        ("rtx 500 ada laptop", 128, 4),
    ]),
    *_group("laptop", "Ampere", [
        ("rtx a5500 laptop", 512, 16), ("rtx a5000 laptop", 448, 16), ("rtx a4000 laptop", 384, 8),
        ("rtx a3000 laptop", 264, 6), ("rtx a2000 laptop", 224, 4), ("rtx a2000 laptop", 224, 8),
        ("rtx a1000 laptop", 224, 4), ("rtx a500 laptop", 112, 4),
    ]),
    # RTX 20 and GTX 10 notebooks with Max-Q; the others carry the desktop names.
    *_group("laptop", "Turing", [
        ("rtx 2080 super with max-q", 352, 8), ("rtx 2080 with max-q", 384, 8),
        ("rtx 2070 super with max-q", 352, 8), ("rtx 2070 with max-q", 384, 8),
        ("rtx 2060 with max-q", 264, 6), ("gtx 1660 ti with max-q", 288, 6),
        ("gtx 1650 with max-q", 128, 4),
    ]),
    *_group("laptop", "Pascal", [
        ("gtx 1080 with max-q", 320, 8), ("gtx 1070 with max-q", 256, 8), ("gtx 1060 with max-q", 192, 6),
    ]),
    # Workstation cards (formerly Quadro).
    *_group("workstation", "Blackwell", [
        ("rtx pro 6000 blackwell", 1792, 96), ("rtx pro 5000 blackwell", 1344, 48),
        ("rtx pro 4500 blackwell", 896, 32), ("rtx pro 4000 blackwell sff", 432, 24),
        ("rtx pro 4000 blackwell", 672, 24), ("rtx pro 2000 blackwell", 288, 16),
    ]),
    *_group("workstation", "Ada", [
        ("rtx 6000 ada", 960, 48), ("rtx 5880 ada", 960, 48), ("rtx 5000 ada", 576, 32), ("rtx 4500 ada", 432, 24),
        ("rtx 4000 ada", 360, 20), ("rtx 4000 sff ada", 280, 20), ("rtx 2000 ada", 224, 16),
    ]),
    *_group("workstation", "Ampere", [
        ("rtx a6000", 768, 48), ("rtx a5500", 768, 24), ("rtx a5000", 768, 24), ("rtx a4500", 640, 20),
        ("rtx a4000", 448, 16), ("rtx a2000", 288, 6), ("rtx a2000", 288, 12),
        ("rtx a1000", 192, 8), ("rtx a400", 96, 4),
    ]),
    *_group("workstation", "Turing", [
        ("rtx 8000", 672, 48), ("rtx 6000", 672, 24), ("rtx 5000", 448, 16), ("rtx 4000", 416, 8),
        ("t1000", 160, 4), ("t1000", 160, 8), ("t600", 160, 4), ("t400", 80, 2), ("t400", 80, 4),
    ]),
    *_group("workstation", "Volta", [("gv100", 870, 32)]),
    *_group("workstation", "Pascal", [
        ("gp100", 717, 16), ("p6000", 432, 24), ("p5000", 288, 16), ("p4000", 243, 8),
        ("p2200", 200, 5), ("p2000", 140, 5), ("p1000", 82, 4), ("p620", 80, 2), ("p400", 32, 2),
    ]),
    # Data centre cards.
    *_group("datacenter", "Blackwell", [("b200", 8000, 180)]),
    *_group("datacenter", "Hopper", [
        ("h200 nvl", 4800, 141), ("h200", 4800, 141), ("h100 nvl", 3900, 94), ("h100 pcie", 2000, 80),
        ("h100", 3350, 80), ("h800 pcie", 2000, 80), ("h800", 3350, 80), ("h20", 4000, 96),
    ]),
    *_group("datacenter", "Ada", [("l40s", 864, 48), ("l40", 864, 48), ("l20", 864, 48), ("l4", 300, 24)]),
    *_group("datacenter", "Ampere", [
        ("a100-sxm4", 1555, 40), ("a100-sxm4", 2039, 80), ("a100", 1555, 40), ("a100", 1935, 80),
        ("a800", 1555, 40), ("a800", 1935, 80), ("a40", 696, 48), ("a30", 933, 24), ("a10g", 600, 24), ("a10", 600, 24), ("a2", 200, 16),
    ]),
    *_group("datacenter", "Turing", [("t4", 320, 16)]),
    *_group("datacenter", "Volta", [("v100s", 1134, 32), ("v100", 900, 16), ("v100", 900, 32)]),
    *_group("datacenter", "Pascal", [
        ("p100", 732, 16), ("p100", 549, 12), ("p40", 347, 24), ("p4", 192, 8),
    ]),
    *_group("datacenter", "Maxwell", [("m40", 288, 12), ("m40", 288, 24)]),
    # AMD, run through llama.cpp's Vulkan build. Names as AMD's driver gives them ("AMD Radeon RX 7900 XTX");
    # Linux adds the chip in brackets ("(RADV NAVI31)"), which the match leaves alone. Not measured yet:
    # every row is the published figure times GPU_EFFICIENCY, which was fitted on NVIDIA cards with CUDA.
    *_group("desktop", "RDNA 4", [
        ("radeon rx 9070 xt", 640, 16), ("radeon rx 9070 gre", 432, 12), ("radeon rx 9070", 640, 16),
        ("radeon rx 9060 xt", 320, 16), ("radeon rx 9060 xt", 320, 8),
    ]),
    *_group("desktop", "RDNA 3", [
        ("radeon rx 7900 xtx", 960, 24), ("radeon rx 7900 xt", 800, 20), ("radeon rx 7900 gre", 576, 16),
        ("radeon rx 7800 xt", 624, 16), ("radeon rx 7700 xt", 432, 12), ("radeon rx 7600 xt", 288, 16),
        ("radeon rx 7600", 288, 8),
    ]),
    *_group("desktop", "RDNA 2", [
        ("radeon rx 6950 xt", 576, 16), ("radeon rx 6900 xt", 512, 16), ("radeon rx 6800 xt", 512, 16),
        ("radeon rx 6800", 512, 16), ("radeon rx 6750 xt", 432, 12), ("radeon rx 6700 xt", 384, 12),
        ("radeon rx 6700", 320, 10), ("radeon rx 6650 xt", 280, 8), ("radeon rx 6600 xt", 256, 8),
        ("radeon rx 6600", 224, 8), ("radeon rx 6500 xt", 144, 4),
    ]),
    *_group("desktop", "RDNA 1", [
        ("radeon rx 5700 xt", 448, 8), ("radeon rx 5700", 448, 8), ("radeon rx 5600 xt", 336, 6),
        ("radeon rx 5500 xt", 224, 8), ("radeon rx 5500 xt", 224, 4),
    ]),
    *_group("desktop", "GCN 5", [
        ("radeon vii", 1024, 16), ("radeon rx vega 64", 484, 8), ("radeon rx vega 56", 410, 8),
    ]),
    *_group("desktop", "GCN 4", [
        ("radeon rx 590", 256, 8), ("radeon rx 580", 256, 8), ("radeon rx 580", 224, 4),
        ("radeon rx 570", 224, 8), ("radeon rx 570", 224, 4), ("radeon rx 480", 256, 8),
    ]),
    *_group("laptop", "RDNA 3", [("radeon rx 7600m xt", 288, 8), ("radeon rx 7600m", 256, 8)]),
    *_group("laptop", "RDNA 2", [("radeon rx 6800m", 384, 12), ("radeon rx 6700m", 320, 10)]),
    *_group("workstation", "RDNA 4", [("radeon ai pro r9700", 640, 32)]),
    *_group("workstation", "RDNA 3", [
        ("radeon pro w7900", 864, 48), ("radeon pro w7800", 576, 32), ("radeon pro w7700", 576, 16),
        ("radeon pro w7600", 288, 8),
    ]),
    *_group("workstation", "RDNA 2", [("radeon pro w6800", 512, 32), ("radeon pro w6600", 224, 8)]),
    *_group("workstation", "GCN 5", [("radeon pro vii", 1024, 16)]),
    *_group("datacenter", "CDNA", [
        ("mi300x", 5300, 192), ("mi250x", 3277, 128), ("mi210", 1638, 64), ("mi100", 1229, 32),
    ]),
    *_group("datacenter", "GCN 5", [("mi60", 1024, 32), ("mi50", 1024, 16), ("mi50", 1024, 32), ("mi25", 484, 16)]),
    # Intel Arc, also through Vulkan ("Intel(R) Arc(TM) B580 Graphics"). Not measured yet either.
    *_group("desktop", "Xe2", [("arc b580", 456, 12), ("arc b570", 380, 10)]),
    *_group("desktop", "Xe", [
        ("arc a770", 560, 16), ("arc a770", 512, 8), ("arc a750", 512, 8), ("arc a580", 512, 8),
        ("arc a380", 186, 6), ("arc a310", 124, 4),
    ]),
    *_group("laptop", "Xe", [
        ("arc a770m", 512, 16), ("arc a730m", 336, 12), ("arc a550m", 224, 8), ("arc a370m", 112, 4),
    ]),
    *_group("workstation", "Xe2", [("arc pro b60", 456, 24), ("arc pro b50", 224, 16)]),
    *_group("workstation", "Xe", [("arc pro a60", 384, 12), ("arc pro a40", 192, 6)]),
]

# Also "Generation" and memory sizes ("RTX 3050 6GB Laptop GPU"): the size is passed separately.
_REMOVED_WORDS = re.compile(r"\b(nvidia|geforce|quadro|tesla|generation|\d+ ?gb|amd|ati|intel|instinct|graphics)\b")
# "RTX A6000", "Tesla V100-PCIE-16GB": a key ends where a letter or digit does not follow.
_PATTERNS = {key: re.compile(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])")
             for key in {g.key for g in GPUS}}
_BY_LENGTH = sorted(_PATTERNS, key=len, reverse=True)
_NOTEBOOK = ("laptop", "mobile", "max-q")


def clean_name(name: str) -> str:
    """A card's name as the table matches it: lower case, maker and brand words taken off."""
    # "Intel(R) Arc(TM) A770 Graphics", "AMD Radeon(TM) RX 6800": the marks are not part of the name.
    lowered = name.lower().replace("(r)", " ").replace("(tm)", " ")
    return " ".join(_REMOVED_WORDS.sub(" ", lowered).split())


def find(name: str, vram_gb: float | None = None) -> GpuSpec | None:
    """The table's entry for a card, or None when the table does not know it.

    ``vram_gb`` is the card's memory as nvidia-smi or a rig file gives it (GiB or GB, it only
    picks among the sizes of one name).
    """
    cleaned = clean_name(name)
    key = next((k for k in _BY_LENGTH if _PATTERNS[k].search(cleaned)), None)
    if key is None:
        return None
    if any(word in cleaned for word in _NOTEBOOK) and not any(word in key for word in _NOTEBOOK):
        # A notebook's card has the desktop card's name but not its memory. Workstation cards
        # sold as Max-Q (RTX PRO 6000 Blackwell Max-Q) keep the full memory, so they stay.
        if not (key.startswith("rtx pro ") and "laptop" not in cleaned):
            return None
    entries = [g for g in GPUS if g.key == key]
    if vram_gb is None or len(entries) == 1:
        return entries[0]
    # nvidia-smi reports a little less than the size on the box (11264 MiB on an 11 GB card).
    return min(entries, key=lambda g: abs(g.vram_gb - vram_gb))


def markdown_table() -> str:
    """The table as Markdown, for people to read (``gpus`` command)."""
    lines = ["| Card | Kind | Architecture | Memory | Published GB/s | Planner GB/s | Source |",
             "|---|---|---|---|---|---|---|"]
    for g in GPUS:
        source = f"measured: {g.measured_by}" if g.measured_gbs is not None else "spec x 0.75"
        lines.append(f"| {g.key} | {g.kind} | {g.arch} | {g.vram_gb:g} GB | {g.gbs:g} | "
                     f"{g.effective_gbs:.0f} | {source} |")
    return "\n".join(lines)
