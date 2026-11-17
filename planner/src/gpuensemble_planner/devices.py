"""GPU and host descriptions for the planner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

GIB = 1024**3

from . import gpu_table
from .gpu_table import GPU_EFFICIENCY  # noqa: F401  (kept importable from here)

@dataclass
class Device:
    name: str
    vram_bytes: int  # memory the engine may use, after other programs and the display
    bandwidth: float  # bytes per second the card reads its own VRAM at
    reserve_bytes: int = 1 * GIB  # compute buffers, CUDA context, fragmentation
    pcie_bandwidth: float | None = None  # host->device bytes/s over this card's own PCIe link
    llama_device: str | None = None  # llama.cpp device name on its own PC, e.g. "CUDA0" or "Vulkan1"
    # A card on another PC: "host:port" of the ggml-rpc-server that serves it.
    # llama.cpp then names it RPC0, RPC1, ... (see llamacpp.rpc_device_names).
    rpc: str | None = None
    # Time each token spends waiting for that server, once per stage (inputs set and the graph
    # started without waiting, then one read of the result): 0.8 ms over loopback on the E5 rig.
    rpc_overhead: float = 0.0
    # Bytes/s the weights are uploaded to that server at when the model loads.
    network_bandwidth: float | None = None
    # The server keeps the weights it receives (worker --cache): later loads send only their hashes.
    rpc_cache: bool = False
    # That PC can download its share of the weights itself when asked (see ``fetch``).
    rpc_fetch: bool = False
    # That PC runs a whole model itself when asked, if the model fits on its cards alone (see ``serving``).
    rpc_serve: bool = False
    # Seconds for a question and its answer to that PC, when measured.
    rpc_rtt: float | None = None

    @property
    def usable_bytes(self) -> int:
        return max(0, self.vram_bytes - self.reserve_bytes)

    @property
    def is_remote(self) -> bool:
        return self.rpc is not None

    @property
    def is_far(self) -> bool:
        """At another site: a question and its answer were measured to take ``FAR_RTT`` or longer."""
        return self.rpc_rtt is not None and self.rpc_rtt >= FAR_RTT


@dataclass
class Host:
    ram_bytes: int  # RAM the engine may use for weights kept off the GPUs
    cpu_bandwidth: float  # bytes/s for layers computed on the CPU
    expert_bandwidth: float  # bytes/s for MoE experts computed on the CPU, with few of them in RAM
    # How experts that do not fit in VRAM are run: "cpu" computes them where
    # they are (llama.cpp -ncmoe); "stream" copies the active ones to the GPU
    # over PCIe each token; "auto" picks the faster per stage.
    expert_path: str = "auto"
    # Fixed cost per token when the first layers run on the CPU (graph split,
    # sync, thread wake-up). Fitted on the E5 rig with Qwen3-32B: 10 ms.
    cpu_overhead: float = 0.0
    # Fixed GPU time per layer (kernel launches, routing). MoE layers cost
    # more: gpt-oss-20b runs at about half its bandwidth bound without it.
    layer_overhead: float = 0.0
    # Extra VRAM the last GPU needs for the output head's compute buffer.
    last_stage_reserve: int = 0
    # CPU threads for llama.cpp (-t). None leaves llama.cpp's default, which on
    # the E5 rig is 16 of 32 and halves the speed of CPU experts.
    threads: int | None = None
    moe_layer_overhead: float = 0.0
    # Extra time per MoE layer whose experts run on the CPU while the rest of
    # the layer runs on a GPU: the hand-over to the CPU and back, every token.
    # Measured with the weights read into RAM, it is too small to see on the
    # E5 rig (0), so the rig file leaves it out.
    moe_cpu_layer_overhead: float = 0.0
    # How much slower the CPU runs experts per GiB of experts kept in RAM:
    # the rate is expert_bandwidth / (1 + this * GiB). On the E5 rig the same
    # expert work per layer took about 3.1 ms with 4.7 GiB of experts in RAM,
    # 4.1 ms with 9.5 GiB and 5.1 ms with 43 GiB (gpt-oss-20b and -120b, read
    # into RAM, 32 threads); fitted: 0.016.
    expert_slowdown_per_gib: float = 0.0
    # Bytes/s at which the CPU runs attention over the KV cache of its layers,
    # per cache type. Much lower than cpu_bandwidth on CPUs without F16C/AVX2.
    # A missing type falls back to cpu_bandwidth.
    cpu_kv_bandwidth: dict[str, float] = field(default_factory=dict)

    def kv_bandwidth(self, kv_type: str) -> float:
        return self.cpu_kv_bandwidth.get(kv_type, self.cpu_bandwidth)


def effective_bandwidth(name: str, vram_gb: float | None = None) -> float | None:
    """Bytes/s the planner counts on for a card (see gpu_table), or None if the table does not know it."""
    spec = gpu_table.find(name, vram_gb)
    return spec.effective_gbs * 1e9 if spec else None


# What detection adds to the name of a card that is not in the table.
GUESSED = " (bandwidth guessed)"


def known_bandwidth(name: str, vram_gb: float | None = None) -> float | None:
    """Published peak bandwidth of a card in bytes/s, or None if the table does not know it."""
    spec = gpu_table.find(name, vram_gb)
    return spec.gbs * 1e9 if spec else None


# Per-token cost of a stage on another PC, when the rig file does not give one.
# Measured over loopback on the E5 rig (llama-bench tg128, the 2080 SUPER behind
# ggml-rpc-server on 127.0.0.1): +0.76 ms for Qwen3-14B, +0.80 ms for gpt-oss-20b.
# A real network adds its round-trip time (see RPC_ROUND_TRIPS); over a LAN
# 2 ms is a guess until one is measured.
RPC_LOOPBACK_OVERHEAD = 0.8e-3
RPC_LAN_OVERHEAD = 2e-3
# Added when the worker lets the main PC in through its gate (worker --allow):
# every round trip then passes a Python hop. E5 rig, gpt-oss-20b over loopback:
# llama-bench tg 81.1-84.5 through the gate against 89.5-90.5 direct, +1.1 ms per token.
RPC_GATE_OVERHEAD = 1.1e-3
# A worker whose answers take this long is at another site, reached over the internet. Sending it
# the weights is then the upload of the main PC's line, which is the slow direction of most home lines.
FAR_RTT = 0.003
# 1 Gb/s Ethernet, when the rig file gives no network_gbs: for the load-time estimate, and for the
# logits that come back from an output head on another PC (see ``logits_seconds``).
DEFAULT_NETWORK_BANDWIDTH = 0.117e9
# Network round trips per token for each stage on another PC. llama.cpp (b12375) sets the inputs
# and starts the graph without waiting, and then waits once, for the result it reads back. Traced
# between two sites over Tailscale (round trip 7.6 ms, Qwen3-32B, benchmarks/phase2c-three-cards.md):
# the read of 32 layers' result took 20.1 ms, which is the round trip plus 8.7 GiB at the card's
# 756 GB/s. Over a LAN the 2 ms above is the floor.
RPC_ROUND_TRIPS = 1


def rpc_overhead_from_rtt(rtt: float, gate: bool = False) -> float:
    """Time per token for a stage on another PC whose measured round trip is ``rtt`` seconds."""
    overhead = max(RPC_LAN_OVERHEAD, RPC_LOOPBACK_OVERHEAD + RPC_ROUND_TRIPS * rtt)
    return overhead + (RPC_GATE_OVERHEAD if gate else 0.0)


def is_loopback(endpoint: str) -> bool:
    host = endpoint.rsplit(":", 1)[0].strip("[]").lower()
    return host == "localhost" or host == "::1" or host.startswith("127.")


def logits_seconds(dev: Device, nbytes: int) -> float:
    """Time per token to bring ``nbytes`` of logits back from the output head on ``dev``.

    The card that holds the head gives back a 32-bit float for each vocabulary
    entry with every token: 0.6 MB for Qwen3, 0.8 MB for gpt-oss. From a card on
    another PC they cross the network. Traced between two sites: the same 32
    layers read back in 26.3 ms with the head and in 20.1 ms without, of which
    0.8 ms is the head's own work, so 0.6 MB took about 5 ms (about 115 MB/s,
    the rate assumed here for any network that the rig file gives no speed for).
    """
    if not dev.is_remote or is_loopback(dev.rpc):
        return 0.0
    return nbytes / (dev.network_bandwidth or DEFAULT_NETWORK_BANDWIDTH)


def check_endpoint(endpoint: str) -> str:
    """``endpoint`` if it is host:port, else ValueError."""
    host, sep, port = endpoint.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"RPC server {endpoint!r} must be host:port, e.g. 192.168.1.20:50052")
    return endpoint


# A card on another PC is planned with this much more room left free than a card in this PC.
# That PC's desktop and other programs use the card too, and Windows then moves what does not
# fit into shared system memory without an error: the model loads and runs at half the speed.
# Measured with an RTX 4090 at another site and Qwen3.8-27B Q8_0: 54 layers there, 0.8 GiB free
# by llama.cpp's own count, 20.4 tok/s; 56 layers, nothing free by that count, 10.5 tok/s
# (benchmarks/phase2c-three-cards.md). With this margin the plan is the first of the two.
REMOTE_FREE_MARGIN = 1 * GIB


def remote_device(d: dict, reserve: float, index: int) -> Device:
    """A card on another PC, from a rig file's GPU entry with ``rpc``."""
    endpoint = check_endpoint(str(d["rpc"]))
    default = RPC_LOOPBACK_OVERHEAD if is_loopback(endpoint) else RPC_LAN_OVERHEAD
    if d.get("gate"):
        default += RPC_GATE_OVERHEAD
    overhead = d.get("rpc_overhead_ms")
    net = d.get("network_gbs")
    dev = _rig_device(d, reserve, d.get("llama_device", f"CUDA{index}"))
    if not is_loopback(endpoint):
        dev.reserve_bytes += REMOTE_FREE_MARGIN
    dev.rpc = endpoint
    dev.rpc_overhead = float(overhead) / 1000 if overhead is not None else default
    dev.network_bandwidth = float(net) * 1e9 if net else None
    dev.rpc_cache = bool(d.get("cache"))
    dev.rpc_fetch = bool(d.get("fetch"))
    dev.rpc_serve = bool(d.get("serve"))
    rtt = d.get("rtt_ms")
    dev.rpc_rtt = float(rtt) / 1000 if rtt is not None else None
    return dev


def _rig_device(d: dict, reserve: float, llama_device: str) -> Device:
    bw = d.get("bandwidth_gbs")
    name = d["name"]
    vram = float(d["vram_gib"])
    bandwidth = bw * 1e9 if bw else effective_bandwidth(name, vram)
    if GUESSED in name and effective_bandwidth(name, vram):
        # A worker running an older version guessed for a card that this version knows.
        bandwidth = effective_bandwidth(name, vram)
        name = name.replace(GUESSED, "")
    if bandwidth is None:
        raise ValueError(f"unknown bandwidth for {d['name']}; add bandwidth_gbs to the rig file")
    return Device(
        name=name,
        vram_bytes=int(vram * GIB),
        bandwidth=bandwidth,
        reserve_bytes=int(d.get("reserve_gib", reserve) * GIB),
        pcie_bandwidth=float(d["pcie_gbs"]) * 1e9 if d.get("pcie_gbs") else None,
        llama_device=llama_device,
    )


def load_gpus(entries: list[dict], reserve: float = 1.0) -> list[Device]:
    """Devices from a rig file's ``gpus`` list.

    Without ``llama_device``, local cards are CUDA0, CUDA1, ... in the order
    listed, and the cards of each RPC server likewise on that server.
    """
    devices = []
    counts: dict[str | None, int] = {}
    for d in entries:
        key = d.get("rpc")
        index = counts.get(key, 0)
        counts[key] = index + 1
        if key is None:
            devices.append(_rig_device(d, reserve, d.get("llama_device", f"CUDA{index}")))
        else:
            devices.append(remote_device(d, reserve, index))
    names = [d.name for d in devices]
    if len(set(names)) != len(names):
        raise ValueError("every GPU in the rig needs its own name")
    return devices


def load_rig(path: str) -> tuple[list[Device], Host]:
    """Load a rig description (see planner/examples/*.json)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    reserve = float(data.get("reserve_gib_per_gpu", 1.0))
    devices = load_gpus(data["gpus"], reserve)
    h = data.get("host", {})
    host = Host(
        ram_bytes=int(h.get("ram_for_weights_gib", 0) * GIB),
        cpu_bandwidth=float(h.get("cpu_bandwidth_gbs", 20)) * 1e9,
        expert_bandwidth=float(h.get("expert_bandwidth_gbs", h.get("cpu_bandwidth_gbs", 20))) * 1e9,
        expert_path=h.get("expert_path", "auto"),
        cpu_overhead=float(h.get("cpu_overhead_ms", 0)) / 1000,
        layer_overhead=float(h.get("layer_overhead_ms", 0)) / 1000,
        moe_layer_overhead=float(h.get("moe_layer_overhead_ms", 0)) / 1000,
        moe_cpu_layer_overhead=float(h.get("moe_cpu_layer_overhead_ms", 0)) / 1000,
        expert_slowdown_per_gib=float(h.get("expert_slowdown_per_gib", 0)),
        last_stage_reserve=int(float(h.get("last_stage_reserve_gib", 0)) * GIB),
        threads=int(h["threads"]) if h.get("threads") else None,
        cpu_kv_bandwidth={t: float(v) * 1e9 for t, v in h.get("cpu_kv_gbs", {}).items()},
    )
    if host.expert_path not in ("auto", "cpu", "stream"):
        raise ValueError(f"host.expert_path must be auto, cpu or stream, not {host.expert_path!r}")
    return devices, host


# Usable share of a PCIe link's raw rate (encoding and protocol overhead).
# Gen2 x8 on the E5 rig is 4 GB/s raw; the rig file uses 3.2.
PCIE_EFFICIENCY = 0.8
# Raw GB/s per lane for PCIe generations 1-6.
PCIE_LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.877}


def pcie_bandwidth(gen: int, width: int) -> float:
    """Bytes/s a card can copy from the host over a PCIe link."""
    return PCIE_LANE_GBS[gen] * width * PCIE_EFFICIENCY * 1e9


def _int_or_none(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:  # "[N/A]", "[Not Supported]"
        return None


# How much VRAM a detected card leaves the plan. An idle card is planned like
# the rig file (total minus the 1 GiB reserve); a card that also drives the
# display has less free memory, and gets its free memory minus 0.55 GiB. On the
# E5 rig this gives about the rig file's room: 7.0 GiB on the SUPER, ~9.82 on the Ti
# (the rig file has 9.85; a detected card should not get more than was measured).
DETECTED_FREE_MARGIN = int(0.55 * GIB)
QUERY_FIELDS = "index,name,memory.total,memory.free,pcie.link.gen.max,pcie.link.width.current"


def parse_nvidia_smi(out: str) -> list[Device]:
    """Devices from ``nvidia-smi --query-gpu=<QUERY_FIELDS> --format=csv,noheader,nounits``."""
    devices = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        index, name, total_mib, free_mib = parts[:4]
        bw = effective_bandwidth(name, float(total_mib) / 1024)
        label = f"GPU{index} {name}" if bw else f"GPU{index} {name}{GUESSED}"
        reserve = Device.__dataclass_fields__["reserve_bytes"].default
        usable = min(int(float(total_mib) * 1024**2) - reserve, int(float(free_mib) * 1024**2) - DETECTED_FREE_MARGIN)
        # An idle card drops to a lower PCIe generation to save power, so take
        # the highest the card and slot can run; the width does not change.
        gen = _int_or_none(parts[4]) if len(parts) > 4 else None
        width = _int_or_none(parts[5]) if len(parts) > 5 else None
        link = pcie_bandwidth(gen, width) if gen in PCIE_LANE_GBS and width else None
        # nvidia-smi numbers cards in PCI bus order; the launcher sets
        # CUDA_DEVICE_ORDER=PCI_BUS_ID so CUDA uses the same numbers.
        devices.append(Device(label, max(0, usable) + reserve, bw or 300e9 * GPU_EFFICIENCY,
                              reserve_bytes=reserve, pcie_bandwidth=link, llama_device=f"CUDA{index}"))
    return devices


def detect_nvidia() -> list[Device]:
    """Ask nvidia-smi for the cards, their free memory and their PCIe links.

    Bandwidth comes from gpu_table (75% of the published figure, or a measured one); unknown cards
    are treated as 300 GB/s and get a warning in their name so the plan is not
    silently wrong.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi was not found; describe the GPUs in a rig file instead")
    out = subprocess.run(
        [exe, f"--query-gpu={QUERY_FIELDS}",
         "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return parse_nvidia_smi(out)


def physical_ram_bytes() -> int | None:
    """Total RAM of this computer, or None if it cannot be read."""
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def ram_for_weights(total: int) -> int:
    """RAM the plan may fill with weights: all but max(8 GiB, 20%) for the OS and other programs.

    64 GiB gives 51.2 (the E5 rig file uses 52, which was measured to work).
    Below 16 GiB the system keeps half instead of 8 GiB, so a small PC can
    still hold the embedding table and a few layers (8 GiB gives 4).
    """
    return max(0, total - max(min(8 * GIB, total // 2), total // 5))


def detect_host(ram_gib: float | None = None) -> Host:
    """A host description from this computer, for when there is no rig file.

    RAM and thread count are read from the system. The CPU speeds and fixed
    costs are the E5 rig's measured values (an AVX-only Xeon from 2012), so
    on a newer CPU the estimate errs on the slow side; a rig file fitted to
    measurements is more accurate.
    """
    if ram_gib is not None:
        ram = int(ram_gib * GIB)
    else:
        total = physical_ram_bytes()
        ram = ram_for_weights(total) if total else 16 * GIB
    return Host(
        ram_bytes=ram,
        cpu_bandwidth=19.6e9,
        expert_bandwidth=17e9,
        cpu_overhead=0.010,
        layer_overhead=0.03e-3,
        moe_layer_overhead=0.22e-3,
        expert_slowdown_per_gib=0.016,
        last_stage_reserve=int(0.03 * GIB),
        threads=os.cpu_count(),
        cpu_kv_bandwidth={"f16": 3.0e9, "q8_0": 1.9e9},
    )
