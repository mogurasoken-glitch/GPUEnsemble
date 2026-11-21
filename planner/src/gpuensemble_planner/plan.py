"""Place a model's layers on several GPUs (layer split / pipeline parallel).

Model of the hardware
---------------------
Generating one token runs the stages one after another, and each stage is
limited by how fast it can read its own weights. So the time per token is

    sum over layers of  bytes read for that layer / bandwidth where it lives

The bytes read are the layer's weights plus its KV cache up to the current
position, so a token late in a long context is slower than an early one.
On the CPU, attention over the cache is far slower than reading the same
bytes of weights (phase 1a on the E5 rig: f16 cache at ~1.5 GB/s), so it has
its own rate per cache type (``host.cpu_kv_bandwidth``). Plans are optimised
for the middle of the context (``depth``, default half of it), and the speed
with an empty and a full context is reported too.

Every GPU holds one contiguous range of layers. Layers that do not fit on any
GPU stay on the CPU as a prefix (layers 0..c-1), the way llama.cpp offloads
the last ``-ngl`` layers. The last GPU in the order also holds the output head.

Two modes
---------
``whole``  every layer (its experts too) goes to a GPU or to the CPU as a whole.
           This is what llama.cpp's ``-ts`` split does.
``moe``    only the dense part of each layer and its KV cache must be on the
           GPU; routed experts fill whatever VRAM is left on that GPU and the
           rest stays in RAM (Strata's idea, phase 1b). Experts left in RAM
           are either computed by the CPU (``host.expert_bandwidth``, like
           llama.cpp ``-ncmoe``) or streamed to the GPU over that card's own
           PCIe link (``Device.pcie_bandwidth``); ``host.expert_path`` picks
           one or lets each stage take the faster. Experts are assumed to be
           used uniformly, so the estimate is pessimistic: a real cache keeps
           the busiest experts on the GPU. When llama.cpp runs the plan
           (``expert_path`` "cpu"), experts move a whole layer at a time, as
           ``-ot`` does, and each layer whose experts stay in RAM costs a
           hand-over to the CPU and back (``host.moe_cpu_layer_overhead``).
           The CPU runs experts more slowly per byte the more of them sit in
           RAM (``host.expert_slowdown_per_gib``). That amount is only known
           once the split is made, so the plan is made again at the rate for
           its own amount until it settles.

Cards on other PCs
------------------
A card on another PC (``Device.rpc``) is reached through llama.cpp's RPC
backend: its stage costs the same as on a local card plus a fixed time per
token for the round trip over the network (``Device.rpc_overhead``). Between
stages only the hidden state of each token crosses the network. The stage that
holds the output head also sends back the logits of every token, 4 bytes for
each vocabulary entry (``devices.logits_seconds``): 5 to 7 ms at 1 Gb/s, so
the head is better kept on a local card when the split leaves the choice.
Experts of a remote stage left in this PC's RAM would cost a round trip per
layer, and they cannot be streamed over this PC's PCIe bus.

Search
------
For each order of the GPUs, a dynamic program over (stage, layer) finds the
split with the least time that fits. Up to 7 GPUs every order is tried.
In ``moe`` mode the RAM limit for experts left off the GPUs is checked after
the split, so a split that is fastest but overfills RAM is dropped rather
than traded for a slower one that fits.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field

from .devices import DEFAULT_NETWORK_BANDWIDTH, GIB, Device, Host, is_loopback, logits_seconds
from .model import ModelProfile

MAX_PERMUTED_DEVICES = 7
# Plans within this much of the fastest generation speed count as equal, and
# the one with the faster prompt processing wins.
PROMPT_TIE = 0.03
# KV cache size relative to f16 (llama.cpp -ctk/-ctv). q8_0 stores 32 values
# in 34 bytes and is close to lossless; q4_0 stores them in 18 bytes.
KV_SCALE = {"f16": 1.0, "q8_0": 34 / 64, "q4_0": 18 / 64}
# A quantized KV cache must win by at least this much to be chosen automatically.
KV_GAIN = 0.02
# A model that is wholly on one other PC gets a note when starting it on that PC
# itself would be at least this much faster.
OTHER_PC_GAIN = 0.1
# llama.cpp's prompt compute buffer grows with the context: the attention mask
# (512-token batch x 2 bytes per context token) and, for a quantized cache, one
# layer's K and V converted to f16. On the E5 rig the SUPER ran out of memory at
# ctx 32768 allocating a 316 MiB buffer that the fixed 1 GiB reserve did not cover.
MASK_BYTES_PER_TOKEN = 512 * 2
# Sequences llama-server keeps apart (its slots: 4 unless -np says otherwise). The KV cache is
# shared by them, but a recurrent layer keeps one state of its own for each.
SEQUENCES = 4
# llama.cpp keeps, for a layer with a sliding window of n tokens, room for n tokens of each sequence
# and one batch (llama_kv_cache_iswa), not the whole context.
UBATCH = 512
# Larger prompt chunks (llama.cpp -ub) to try, largest first, when part of the model stays in RAM.
# A prompt chunk copies the CPU's weights to the first GPU once (op offload), whatever its size, so a
# chunk four times as long shares that copy among four times the tokens. On the E5 rig
# (engine/PLAN.md, 2026-11-18): Qwen3.8-27B with 8 CPU layers read a 4096-token prompt at 250
# tokens/s with 512-token chunks and at 497 with 2048; gpt-oss-120b at 49 (experts computed on the
# CPU) and at 117. Generation was the same when the placement was.
LARGER_UBATCHES = (4096, 2048, 1024)
# Generation may get this much slower, by the estimate, for the VRAM the longer chunks take.
# Llama 3.3 70B on the E5 rig with chunks of 1024 (two more layers on the CPU, estimate -3.6%):
# a 4096-token prompt at 52 tokens/s instead of 35.5, generation at 1.28 instead of 1.35. For an
# answer of a few hundred tokens that is no gain, so such a plan keeps 512 unless asked.
CHUNK_GEN_COST = 0.03
# llama.cpp's batch size (-b) to go with a larger chunk.
BATCH_FOR_LARGER_UBATCH = 4096
# For a MoE model with longer chunks: prompts shorter than this many tokens are computed on the CPU
# instead of copying their experts to the GPU (llama.cpp reads GGML_OP_OFFLOAD_MIN_BATCH; its
# default is 32). gpt-oss-120b on the E5 rig, chunks of 1024: a 256-token prompt at 41 tokens/s
# instead of 20, a 4096-token one at 62 instead of 70 (49.5 with chunks of 512 on the CPU).
MOE_OFFLOAD_MIN_BATCH = 600
# The chunk a MoE model with experts in RAM gets when none is asked for (if it fits the context and the VRAM).
MOE_AUTO_UBATCH = 4096
# ... and only while the estimate of generation drops by at most this much. gpt-oss-120b on the E5 rig:
# estimate -8%, measured -4%. Qwen3-30B-A3B there would lose about 15% by the estimate (not measured), so it keeps 512.
MOE_CHUNK_GEN_COST = 0.10
# VRAM a GPU's compute buffer takes for each token of a chunk beyond the 512 that the fixed reserve
# covers. Measured on the E5 rig per token: 0.14 MiB (gpt-oss-120b) and 0.34 MiB (Qwen3.8-27B) on a
# card without the head, and on the card with the head the logits (4 bytes for each vocabulary
# entry) plus up to 0.4 MiB. With 0.7 for the first, gpt-oss-120b at 2048 left 1.8 GB of the 2080
# SUPER unused and generation fell from 11.8 to 8.1 tokens/s.
#
# The logits are only needed when every token of the chunk gets an output: llama-bench's prompt test
# does that, llama-server asks for the last token's alone. Under llama-server the compute buffer of
# gpt-oss-120b on the E5 rig was the same on both cards (4096-token chunks: 1182 MiB on the card with
# the head, 1208 MiB on the other; 865 MiB at 2048), so a plan for the server does not set the logits
# aside (``all_logits=False``): with them, chunks of 2048 left 1 GB of the 2080 Ti unused
# (engine/prototype/RESULTS-server-stock.md).
CHUNK_BYTES_PER_TOKEN = 0.4 * 2**20
CHUNK_BYTES_PER_TOKEN_WITH_HEAD = 0.45 * 2**20
# Logits per token when the vocabulary size is not known.
DEFAULT_LOGITS_BYTES = 0.8 * 2**20


def window_tokens(window: int, context: int, ubatch: int = UBATCH) -> int:
    """Tokens of KV cache a layer holds: the whole context, or less for a sliding window."""
    return context if window <= 0 else min(context, window * SEQUENCES + ubatch)


class PlanError(RuntimeError):
    pass


@dataclass
class Stage:
    device: str
    first_layer: int
    last_layer: int  # inclusive; -1 when the stage holds no layers
    weight_bytes: int  # weights resident on this device
    kv_bytes: int
    head_bytes: int
    capacity_bytes: int
    seconds_per_token: float
    expert_fraction_on_gpu: float = 1.0  # moe mode only
    expert_path: str = ""  # moe mode: "cpu" or "stream" for the experts left in RAM

    @property
    def n_layers(self) -> int:
        return 0 if self.last_layer < self.first_layer else self.last_layer - self.first_layer + 1

    @property
    def used_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes + self.head_bytes


@dataclass
class Plan:
    model: str
    mode: str
    context: int
    stages: list[Stage]
    cpu_layers: int
    cpu_bytes: int  # weights kept in RAM: embedding table, CPU prefix, uncached experts
    seconds_per_token: float
    notes: list[str] = field(default_factory=list)
    kv_type: str = "f16"
    depth: int = 0  # tokens already in the cache that seconds_per_token is for
    seconds_empty: float = 0.0  # first token after an empty context
    seconds_full: float = 0.0  # last token of a full context
    expert_ram_bytes: int = 0  # routed expert weights kept in RAM (CPU layers and uncached experts)
    ubatch: int = UBATCH  # tokens of a prompt chunk (llama.cpp -ub); see LARGER_UBATCHES
    # Set when one other PC holds the whole model on its cards alone: that PC's address, its RPC
    # server ("host:port"), the plan that PC would make for itself, and that plan's time per token
    # (same context as seconds_per_token). See ``other_pc_note``.
    other_pc: str = ""
    other_pc_rpc: str = ""
    on_other_pc: "Plan | None" = None
    seconds_on_other_pc: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return _rate(self.seconds_per_token)

    @property
    def tokens_per_second_on_other_pc(self) -> float:
        return _rate(self.seconds_on_other_pc)

    @property
    def tokens_per_second_empty(self) -> float:
        return _rate(self.seconds_empty)

    @property
    def tokens_per_second_full(self) -> float:
        return _rate(self.seconds_full)


def _rate(seconds: float) -> float:
    return 1.0 / seconds if seconds > 0 else float("inf")


class _Prefix:
    """Prefix sums over the layers so any stage [s, e) is priced in O(1)."""

    def __init__(self, model: ModelProfile, context: int, kv_scale: float = 1.0, ubatch: int = UBATCH,
                 all_logits: bool = False) -> None:
        self.dense = [0]
        self.experts = [0]
        self.kv = [0]
        self.kv_token = [0.0]  # layers that look back at the whole context
        self.windows: dict[int, list[float]] = {}  # the same for each sliding window size
        # Compute buffer that every GPU needs on top of its fixed reserve.
        widest = max((layer.kv_bytes_per_token for layer in model.layers), default=0)
        self.compute = context * (widest + MASK_BYTES_PER_TOKEN)
        # What a chunk longer than 512 tokens adds to it: on every GPU, and on the one with the head.
        extra = max(0, ubatch - UBATCH)
        self.chunk = int(extra * CHUNK_BYTES_PER_TOKEN)
        logits = (model.logits_bytes or DEFAULT_LOGITS_BYTES) if all_logits else 0
        self.chunk_with_head = int(extra * (CHUNK_BYTES_PER_TOKEN_WITH_HEAD + logits))
        for w in {layer.kv_window for layer in model.layers if layer.kv_window > 0}:
            self.windows[w] = [0.0]
        for layer in model.layers:
            self.dense.append(self.dense[-1] + layer.dense_bytes)
            self.experts.append(self.experts[-1] + layer.expert_bytes)
            # The cache grows with the context (up to its window); a recurrent layer's state does not,
            # and is not quantized.
            self.kv.append(self.kv[-1] + int(layer.kv_bytes_per_token * window_tokens(layer.kv_window, context, ubatch) * kv_scale)
                           + layer.state_bytes * SEQUENCES)
            whole = 0.0 if layer.kv_window > 0 else layer.kv_bytes_per_token * kv_scale
            self.kv_token.append(self.kv_token[-1] + whole)
            for w, sums in self.windows.items():
                sums.append(sums[-1] + (layer.kv_bytes_per_token * kv_scale if layer.kv_window == w else 0.0))

    def span(self, s: int, e: int) -> tuple[int, int, int]:
        return (self.dense[e] - self.dense[s], self.experts[e] - self.experts[s], self.kv[e] - self.kv[s])

    def kv_read(self, s: int, e: int, depth: float) -> float:
        """KV cache bytes that layers [s, e) read for a token with ``depth`` tokens before it."""
        read = (self.kv_token[e] - self.kv_token[s]) * depth
        for w, sums in self.windows.items():
            read += (sums[e] - sums[s]) * min(depth, w)
        return read


def _ram_expert_seconds(active_bytes: float, dev: Device, host: Host) -> tuple[float, str]:
    """Time to run ``active_bytes`` of experts that live in RAM, and the path used."""
    if active_bytes <= 0:
        return 0.0, ""
    cpu = active_bytes / host.expert_bandwidth
    # A card on another PC is not on this PC's PCIe bus: nothing streams to it.
    link = None if dev.is_remote else dev.pcie_bandwidth
    if host.expert_path == "cpu" or link is None:
        if host.expert_path == "stream":
            raise PlanError(f"expert_path is stream but {dev.name} has no pcie_gbs" +
                            (" here (it is on another PC)" if dev.is_remote else ""))
        return cpu, "cpu"
    stream = active_bytes / link
    if host.expert_path == "stream" or stream < cpu:
        return stream, "stream"
    return cpu, "cpu"


def _capacity(dev: Device, host: Host, pre: _Prefix, is_last: bool) -> int:
    """Bytes of weights, KV cache and head that ``dev`` can take."""
    chunk = pre.chunk_with_head if is_last else pre.chunk
    # Not below zero: a card too small for the buffers still counts as a stage without layers.
    return max(0, dev.usable_bytes - pre.compute - chunk) - (host.last_stage_reserve if is_last else 0)


def _stage_cost(
    model: ModelProfile,
    pre: _Prefix,
    dev: Device,
    host: Host,
    s: int,
    e: int,
    is_last: bool,
    mode: str,
    depth: int = 0,
) -> tuple[float, int, int, float, str] | None:
    """Price layers [s, e) on ``dev`` for a token with ``depth`` tokens before it.

    Returns (seconds per token, bytes on the GPU, expert bytes left in RAM,
    fraction of experts on the GPU, path for RAM experts), or None when the
    stage does not fit.
    """
    dense, experts, kv = pre.span(s, e)
    head = model.head_bytes if is_last else 0
    cap = _capacity(dev, host, pre, is_last)
    fixed = (e - s) * (host.moe_layer_overhead if model.is_moe else host.layer_overhead)
    fixed += pre.kv_read(s, e, depth) / dev.bandwidth  # attention reads the cache
    if e > s:
        fixed += dev.rpc_overhead  # round trips to the PC the card is in
    if is_last:
        fixed += logits_seconds(dev, model.logits_bytes)  # the head's answer comes back over the network
    active = model.active_expert_ratio
    if mode == "whole":
        used = dense + experts + kv + head
        if used > cap:
            return None
        seconds = (dense + head + experts * active) / dev.bandwidth + fixed
        return seconds, dense + experts, 0, 1.0, ""
    # moe: dense part, KV and head must fit; experts fill the rest.
    base = dense + kv + head
    if base > cap:
        return None
    if host.expert_path == "cpu":
        on_gpu, ram_layers = _whole_layer_experts(pre, s, e, cap - base)
    else:
        on_gpu = min(experts, cap - base)
        moe_layers = sum(1 for i in range(s, e) if pre.experts[i + 1] > pre.experts[i])
        ram_layers = (experts - on_gpu) / experts * moe_layers if experts else 0.0
    in_ram = experts - on_gpu
    ram_seconds, path = _ram_expert_seconds(active * in_ram, dev, host)
    if path == "cpu":
        fixed += ram_layers * host.moe_cpu_layer_overhead
        # Experts left in this PC's RAM for a card on another PC: each such
        # layer sends its activations here and back, a round trip of its own.
        fixed += ram_layers * dev.rpc_overhead
    seconds = (dense + head + active * on_gpu) / dev.bandwidth + ram_seconds + fixed
    fraction = on_gpu / experts if experts else 1.0
    return seconds, dense + on_gpu, in_ram, fraction, path


def _whole_layer_experts(pre: _Prefix, s: int, e: int, budget: float) -> tuple[int, int]:
    """Expert bytes that fit on the GPU when they move a layer at a time, and the
    number of layers whose experts stay in RAM.

    llama.cpp pins experts to the CPU per layer (``-ot``), so the last layers of
    the stage keep theirs on the GPU while they fit; ``llamacpp`` makes the same
    choice when it writes the command.
    """
    on_gpu, ram_layers = 0, 0
    for i in range(e - 1, s - 1, -1):
        size = pre.experts[i + 1] - pre.experts[i]
        if size and size <= budget + 1:
            budget -= size
            on_gpu += size
        elif size:
            ram_layers += 1
    return on_gpu, ram_layers


def _plan_order(
    model: ModelProfile,
    order: list[Device],
    host: Host,
    context: int,
    mode: str,
    kv_type: str = "f16",
    depth: int = 0,
    ubatch: int = UBATCH,
) -> Plan | None:
    n = model.n_layers
    k = len(order)
    pre = _Prefix(model, context, KV_SCALE[kv_type], ubatch, all_logits=host.all_logits)
    active = model.active_expert_ratio
    cpu_kv = host.kv_bandwidth(kv_type)

    def cpu_seconds(c: int, d: int = depth) -> float:
        dense, experts, _ = pre.span(0, c)
        # Experts on the CPU run at the expert rate, like experts left in RAM.
        weights = dense / host.cpu_bandwidth + experts * active / host.expert_bandwidth
        return weights + pre.kv_read(0, c, d) / cpu_kv

    def cpu_ram(c: int) -> int:
        dense, experts, _ = pre.span(0, c)
        return dense + experts

    inf = float("inf")
    # best[j][i]: least time with layers [0, i) on the CPU and the first j devices.
    best = [[inf] * (n + 1) for _ in range(k + 1)]
    back = [[-1] * (n + 1) for _ in range(k + 1)]
    for c in range(n + 1):
        if cpu_ram(c) + model.embed_bytes <= host.ram_bytes:
            best[0][c] = cpu_seconds(c) + (host.cpu_overhead if c else 0.0)
    if k == 0:
        # No GPU at all (a PC without one, and no other PC's cards): every layer and the head on the CPU.
        head = model.head_bytes
        if best[0][n] == inf or cpu_ram(n) + model.embed_bytes + head > host.ram_bytes:
            return None
        total = cpu_seconds(n) + head / host.cpu_bandwidth
        return Plan(
            model=model.name, mode=mode, context=context, stages=[], cpu_layers=n,
            cpu_bytes=cpu_ram(n) + model.embed_bytes + head, seconds_per_token=total, kv_type=kv_type,
            depth=depth, seconds_empty=cpu_seconds(n, 0) + head / host.cpu_bandwidth,
            seconds_full=cpu_seconds(n, context) + head / host.cpu_bandwidth,
            expert_ram_bytes=pre.span(0, n)[1],
        )
    for j, dev in enumerate(order, start=1):
        is_last = j == k
        for i in range(n + 1):
            for s in range(i, -1, -1):
                # llama.cpp gives the head to the last card it is told to use, and is not told of a
                # card without layers. So the last card of the order holds the head with a layer of
                # its own, or when no card holds a layer; never alone behind another card's layers,
                # which would count the head's VRAM here and none of it on the card that will hold
                # it (nor the logits coming back, if that card is on another PC). The same split
                # with the empty card earlier is another order, priced as it will run.
                head_only = is_last and s == n
                before = best[0][n] if head_only else best[j - 1][s]
                if before == inf:
                    continue
                priced = _stage_cost(model, pre, dev, host, s, i, is_last, mode, depth)
                if priced is None:
                    # A smaller s only makes the stage longer, so it will not fit either.
                    break
                t = before + priced[0]
                if t < best[j][i]:
                    best[j][i] = t
                    back[j][i] = s
    if best[k][n] == inf:
        return None

    bounds = [n]
    i = n
    for j in range(k, 0, -1):
        i = n if back[k][n] == n else back[j][i]  # the head alone on the last card: every layer is on the CPU
        bounds.append(i)
    bounds.reverse()  # bounds[0] = CPU layers, bounds[j] = end of device j
    cpu_layers = bounds[0]

    stages: list[Stage] = []
    ram_bytes = model.embed_bytes + cpu_ram(cpu_layers)
    expert_ram = pre.span(0, cpu_layers)[1]
    total = cpu_seconds(cpu_layers)
    cpu_used = cpu_layers > 0
    for j, dev in enumerate(order, start=1):
        s, e = bounds[j - 1], bounds[j]
        is_last = j == k
        seconds, weights, in_ram, fraction, path = _stage_cost(model, pre, dev, host, s, e, is_last, mode, depth)  # type: ignore[misc]
        ram_bytes += in_ram
        expert_ram += in_ram
        total += seconds
        stages.append(
            Stage(
                device=dev.name,
                first_layer=s,
                last_layer=e - 1,
                weight_bytes=weights,
                kv_bytes=pre.span(s, e)[2],
                head_bytes=model.head_bytes if is_last else 0,
                capacity_bytes=_capacity(dev, host, pre, is_last),
                seconds_per_token=seconds,
                expert_fraction_on_gpu=fraction,
                expert_path=path,
            )
        )

    if cpu_used:
        total += host.cpu_overhead

    # Experts left in RAM are only known after the split, so RAM is checked here.
    if ram_bytes > host.ram_bytes:
        return None

    def total_at(d: int) -> float:
        t = cpu_seconds(cpu_layers, d) + (host.cpu_overhead if cpu_used else 0.0)
        for j, dev in enumerate(order, start=1):
            t += _stage_cost(model, pre, dev, host, bounds[j - 1], bounds[j], j == k, mode, d)[0]  # type: ignore[index]
        return t

    return Plan(
        model=model.name,
        mode=mode,
        context=context,
        stages=stages,
        cpu_layers=cpu_layers,
        cpu_bytes=ram_bytes,
        seconds_per_token=total,
        kv_type=kv_type,
        depth=depth,
        seconds_empty=total_at(0),
        seconds_full=total_at(context),
        expert_ram_bytes=expert_ram,
        ubatch=ubatch,
    )


def streams_cpu_weights(plan: Plan, model: ModelProfile, devices: list[Device] | None = None) -> bool:
    """Whether prompt processing copies the CPU's weights to a GPU.

    llama.cpp's op offload sends the weights of CPU layers to the first GPU
    for every prompt batch, over that GPU's PCIe link. For MoE experts kept on
    the CPU this is slower than computing them there on the E5 rig (phase 1a:
    pp 16 vs 42 tokens/s) with chunks of 512 tokens, so the launcher turns it
    off for MoE models unless the plan uses larger chunks (``Plan.ubatch``). It is
    off too when the first card is on another PC (as on a main PC without a
    GPU), where the weights would cross the network for every batch, and when
    there is no card at all.
    """
    if model.is_moe:
        # With chunks of 512 tokens the CPU is faster for experts; with larger ones the copy wins
        # (LARGER_UBATCHES), also for experts left in RAM by layers that are on a GPU.
        if plan.ubatch <= UBATCH or not (plan.cpu_layers or plan.expert_ram_bytes):
            return False
    elif plan.cpu_layers == 0:
        return False
    first = next((s for s in plan.stages if s.n_layers), None)
    if first is None:
        return False
    if devices is not None:
        dev = next((d for d in devices if d.name == first.device), None)
        if dev is not None and dev.is_remote:
            return False
    return True


def _first_link(plan: Plan, devices: list[Device]) -> float:
    by_name = {d.name: d for d in devices}
    for s in plan.stages:
        if s.n_layers:
            dev = by_name[s.device]
            # A card on another PC would get the CPU's weights over the network.
            return 0.0 if dev.is_remote else dev.pcie_bandwidth or 0.0
    return 0.0


def _load_time(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"


# How fast a slow link between sites carries weights: about 80 Mb/s. Over Tailscale between two
# homes, 21.6 GiB went at about 46 MB/s; a relayed or busier link can be several times slower.
SLOW_NETWORK_BANDWIDTH = 10e6


def remote_load_seconds(plan: Plan, devices: list[Device]) -> float:
    """Time to allow, on top of the usual load, for sending weights to cards on other PCs.

    Counted as if the link were slow and the other PC kept nothing, so that a
    first load over the internet is not given up while its weights still move.
    """
    by_name = {d.name: d for d in devices}
    sent = sum(s.weight_bytes for s in plan.stages
               if s.n_layers and by_name[s.device].is_remote and not is_loopback(by_name[s.device].rpc or ""))
    return sent / SLOW_NETWORK_BANDWIDTH


def remote_notes(plan: Plan, devices: list[Device]) -> list[str]:
    """What a plan with cards on other PCs sends over the network."""
    by_name = {d.name: d for d in devices}
    notes = []
    for s in plan.stages:
        dev = by_name[s.device]
        if not (dev.is_remote and s.n_layers):
            continue
        if dev.is_far and dev.rpc_cache and dev.rpc_fetch and not is_loopback(dev.rpc or ""):
            # `up` asks such a PC to download its share itself (cli.fetch_on_workers).
            notes.append(f"layers {s.first_layer}-{s.last_layer} run on {dev.name} on another PC ({dev.rpc}); "
                         f"the first time this model loads, that PC needs its {s.weight_bytes / GIB:.1f} GiB of "
                         "weights: it downloads them itself if it can, and this PC sends them if not; "
                         "that PC keeps them, so later loads send only their hashes")
            continue
        where = "through the RPC server on this PC" if is_loopback(dev.rpc or "") else "on another PC"
        when = "the first time this model loads" if dev.rpc_cache else "every time the model loads"
        note = (f"layers {s.first_layer}-{s.last_layer} run on {dev.name} {where} ({dev.rpc}); "
                f"{s.weight_bytes / GIB:.1f} GiB of weights go over the network {when}")
        if dev.network_bandwidth:
            rate = dev.network_bandwidth
            note += f" (about {_load_time(s.weight_bytes / rate)} at {rate / 1e9:.2g} GB/s)"
        elif dev.is_far:
            note += " (at the speed of this PC's upload to that site)"
        elif not is_loopback(dev.rpc or ""):
            note += f" (about {_load_time(s.weight_bytes / DEFAULT_NETWORK_BANDWIDTH)} over 1 Gb/s Ethernet)"
        if dev.rpc_cache:
            note += "; that PC keeps them, so later loads send only their hashes"
        notes.append(note)
    return notes


def waiting_note(plan: Plan, devices: list[Device]) -> str | None:
    """A warning for a plan whose layers are split between this PC's cards and another PC's.

    While the other PC computes its layers, this PC's cards have nothing to
    do, every token. Measured between two sites (benchmarks/phase2c-three-cards.md,
    RTX 2080 Ti and 2080 SUPER, Windows, default power settings): about 3 s into
    a request the driver takes that for light load and lowers their clocks, and
    their share of each token takes about twice as long. The estimate does not
    include that.
    """
    by_name = {d.name: d for d in devices}
    used = [by_name[s.device] for s in plan.stages if s.n_layers]
    far = any(d.is_remote and not is_loopback(d.rpc or "") for d in used)
    if far and any(not d.is_remote for d in used):
        return ("this PC's GPUs wait while the other PC computes its layers; if their driver lowers their clocks "
                "meanwhile (seen on Windows with the default power settings), generation runs 20-30% below this "
                "estimate")
    return None


def plan_on_other_pc(model: ModelProfile, devices: list[Device], host: Host, context: int,
                     kv_type: str = "auto", depth: int | None = None) -> tuple[str, Plan] | None:
    """The fastest plan on the cards of one other PC alone, as that PC would make it for itself.

    Returns the PC's RPC server ("host:port") and the plan, or None when no
    other PC holds the whole model on its cards. That PC's CPU and RAM are not
    known here (``host`` is this PC's), so a plan that needs them does not count.
    """
    best: tuple[str, Plan] | None = None
    for rpc in dict.fromkeys(d.rpc for d in devices if d.rpc and not is_loopback(d.rpc)):
        own = [dataclasses.replace(d, rpc=None, rpc_overhead=0.0, network_bandwidth=None, rpc_cache=False,
                                   rpc_fetch=False, rpc_rtt=None) for d in devices if d.rpc == rpc]
        try:
            there = _plan_here(model, own, host, context, "whole", kv_type, depth)
        except PlanError:
            continue
        if there.cpu_layers or there.expert_ram_bytes:
            continue
        if best is None or there.seconds_per_token < best[1].seconds_per_token:
            best = (rpc, there)
    return best


def faster_on_other_pc(plan: Plan) -> bool:
    """Whether the model would run faster by ``OTHER_PC_GAIN`` started on the other PC that holds all of it."""
    return plan.on_other_pc is not None and plan.seconds_per_token >= plan.seconds_on_other_pc * (1 + OTHER_PC_GAIN)


def other_pc_note(plan: Plan, model: ModelProfile, devices: list[Device], host: Host,
                  kv_type: str = "auto") -> str | None:
    """A hint for a model that one other PC holds on its cards alone.

    Started on this PC, every stage on that PC's cards waits for the network
    with each token, and so does the head's answer coming back
    (``Device.rpc_overhead``, ``devices.logits_seconds``). Started on that PC
    itself, the model runs without them. The plan records what that PC would
    do alone (``Plan.on_other_pc``, ``Plan.seconds_on_other_pc``); the note is
    given when that is at least ``OTHER_PC_GAIN`` faster than this plan.

    Measured between two sites (benchmarks/phase2c-three-cards.md): Qwen3-32B on
    an RTX 4090 ran at 25.2-25.95 tokens/s started on the main PC and at 41.4-41.5
    started on the 4090's PC, 14.4-15.6 ms per token apart; the two costs above
    come to 14.7 ms there.
    """
    found = plan_on_other_pc(model, devices, host, plan.context, kv_type, plan.depth)
    if found is None:
        return None
    plan.other_pc_rpc, plan.on_other_pc = found
    plan.other_pc = plan.other_pc_rpc.rsplit(":", 1)[0]
    plan.seconds_on_other_pc = plan.on_other_pc.seconds_per_token
    if not faster_on_other_pc(plan):
        return None
    return (f"the other PC ({plan.other_pc}) holds this model on its cards alone: started on that PC itself, it would "
            f"run at about {plan.tokens_per_second_on_other_pc:.0f} tokens/s instead of {plan.tokens_per_second:.0f} "
            "(started here, each token waits for the network)")


def serving_pc(plan: Plan, devices: list[Device]) -> tuple[str, int] | None:
    """Where ``up`` starts a plan's model instead of on this PC: the address and RPC port of the
    other PC that holds all of it on its cards, runs models itself when asked (``worker --serve``),
    and is faster that way by ``OTHER_PC_GAIN``. None when the model is to start on this PC."""
    if not faster_on_other_pc(plan):
        return None
    if not all(d.rpc_serve for d in devices if d.rpc == plan.other_pc_rpc):
        return None
    host, _, port = plan.other_pc_rpc.rpartition(":")
    return host, int(port)


def _kv_saving_moves_a_layer(plan: Plan, model: ModelProfile) -> bool:
    """Whether a q8_0 cache frees at least one average layer's weights on the GPUs."""
    saved = sum(s.kv_bytes for s in plan.stages) * (1 - KV_SCALE["q8_0"])
    layer = sum(l.dense_bytes + l.expert_bytes for l in model.layers) / max(1, model.n_layers)
    return saved >= layer


def _orders(devices: list[Device]) -> list[list[Device]]:
    if len(devices) <= MAX_PERMUTED_DEVICES:
        return [list(p) for p in itertools.permutations(devices)]
    # Too many to try them all: slowest first, fastest last (it holds the head).
    return [sorted(devices, key=lambda d: (d.bandwidth, d.vram_bytes))]


def make_plan(
    model: ModelProfile,
    devices: list[Device],
    host: Host,
    context: int = 8192,
    mode: str = "auto",
    kv_type: str = "auto",
    depth: int | None = None,
    chunk: int | str = "auto",
) -> Plan:
    """Find the fastest placement that fits (see ``_plan_here``), and say so when one other PC
    would run the model faster on its cards alone (``other_pc_note``)."""
    plan = _plan_here(model, devices, host, context, mode, kv_type, depth, chunk)
    note = other_pc_note(plan, model, devices, host, kv_type)
    if note:
        plan.notes.append(note)
    return plan


def _plan_here(
    model: ModelProfile,
    devices: list[Device],
    host: Host,
    context: int = 8192,
    mode: str = "auto",
    kv_type: str = "auto",
    depth: int | None = None,
    chunk: int | str = "auto",
) -> Plan:
    """Find the fastest placement that fits.

    The time per token is minimised for a token with ``depth`` tokens of
    context before it, by default half the context.

    ``mode="auto"`` uses ``whole`` for dense models and for MoE models that fit
    whole, and ``moe`` otherwise. ``kv_type="auto"`` keeps the KV cache in f16
    unless a q8_0 cache lets more layers onto the GPUs and is clearly faster
    (on the E5 rig, Qwen3-32B at 8192 tokens of context).

    ``chunk="auto"`` reads prompts in chunks of 512 tokens, or in longer ones
    (``LARGER_UBATCHES``) when a dense model has layers on the CPU and the VRAM
    the longer chunks take costs generation at most ``CHUNK_GEN_COST``. A MoE
    model with experts in RAM gets chunks of ``MOE_AUTO_UBATCH`` when they fit.
    A number asks for that size.
    """
    if kv_type == "auto":
        plain = None
        try:
            plain = _plan_here(model, devices, host, context, mode, "f16", depth, chunk)
        except PlanError:
            pass
        if plain is not None and plain.cpu_layers == 0 and not (plain.mode == "moe" and model.is_moe):
            return plain
        if plain is not None and not _kv_saving_moves_a_layer(plain, model):
            # A smaller cache would only squeeze in a layer at the very edge of
            # the VRAM estimate (Qwen3-32B at ctx 640), so keep f16.
            return plain
        try:
            small = _plan_here(model, devices, host, context, mode, "q8_0", depth, chunk)
        except PlanError:
            if plain is None:
                raise
            return plain
        if plain is None or small.seconds_per_token < plain.seconds_per_token * (1 - KV_GAIN):
            small.notes.append("KV cache in q8_0 (half the VRAM of f16, close to lossless) to keep more layers on the GPUs")
            return small
        return plain
    plan = _plan_chunk(model, devices, host, context, mode, kv_type, depth, UBATCH)
    if not wants_larger_chunks(plan, devices):
        if chunk not in ("auto", UBATCH, str(UBATCH)):
            plan.notes.append(f"prompt chunks stay at {UBATCH} tokens: longer ones only help when weights stay in "
                              "RAM and every card is in this PC")
        return plan
    # Part of the model stays in RAM: longer prompt chunks read prompts faster but take VRAM, so
    # use the longest that costs generation little.
    asked = chunk != "auto"
    # Experts in RAM: only the longest chunk is tried by itself. gpt-oss-120b on the E5 rig
    # (engine/prototype/RESULTS-server-stock.md, result 5): with chunks of 4096 an 8000-token prompt
    # answers after 41 s instead of 166 s, a 190-token follow-up after 5.2 s instead of 4.4 s
    # (``llamacpp`` keeps short prompts on the CPU), generation at 10.5 tokens/s instead of 11.0.
    # With chunks of 2048 the placement generated at 9.5, so the sizes in between are only used when asked for.
    moe_auto = model.is_moe and not asked
    sizes = [int(chunk)] if asked else [MOE_AUTO_UBATCH] if moe_auto else list(LARGER_UBATCHES)
    for ubatch in sizes:
        if ubatch <= UBATCH or ubatch * 2 > context:
            continue  # a chunk is at most the context; leave room for the answer
        try:
            larger = _plan_chunk(model, devices, host, context, mode, kv_type, depth, ubatch)
        except PlanError:
            continue
        if not streams_cpu_weights(larger, model, devices):
            continue
        cost = MOE_CHUNK_GEN_COST if moe_auto else CHUNK_GEN_COST
        if asked or larger.seconds_per_token <= plan.seconds_per_token * (1 + cost):
            larger.notes.append(
                f"prompts are read in chunks of {ubatch} tokens instead of {UBATCH}: each chunk copies the "
                "weights kept in RAM to the first GPU once, so long prompts are read faster (short ones a "
                "little slower); the longer chunks take some VRAM")
            return larger
    return plan


def wants_larger_chunks(plan: Plan, devices: list[Device]) -> bool:
    """Whether a plan could read prompts faster in longer chunks (``LARGER_UBATCHES``).

    Only when weights stay in RAM (CPU layers, or experts of layers on a GPU)
    and every card is in this PC: with a card on another PC it has not been
    measured.
    """
    if not (plan.cpu_layers or plan.expert_ram_bytes):
        return False
    by_name = {d.name: d for d in devices}
    used = [by_name[s.device] for s in plan.stages if s.n_layers]
    return bool(used) and not any(d.is_remote for d in used)


def _plan_chunk(
    model: ModelProfile,
    devices: list[Device],
    host: Host,
    context: int,
    mode: str,
    kv_type: str,
    depth: int | None,
    ubatch: int,
) -> Plan:
    """The fastest placement that fits for one KV cache type and one prompt chunk size."""
    plan = _make_plan(model, devices, host, context, mode, kv_type, depth, ubatch)
    if not host.expert_slowdown_per_gib:
        return plan
    # The CPU's rate for experts depends on how many of them the plan leaves in
    # RAM, so plan again at the rate for that amount until it stops changing.
    tried: set[int] = set()
    while plan.expert_ram_bytes and plan.expert_ram_bytes not in tried and len(tried) < 4:
        tried.add(plan.expert_ram_bytes)
        plan = _make_plan(model, devices, slowed_host(host, plan.expert_ram_bytes), context, mode, kv_type, depth, ubatch)
    return plan


def slowed_host(host: Host, expert_ram_bytes: int) -> Host:
    """``host`` with the CPU's expert rate for ``expert_ram_bytes`` of experts in RAM."""
    rate = host.expert_bandwidth / (1 + host.expert_slowdown_per_gib * expert_ram_bytes / GIB)
    return dataclasses.replace(host, expert_bandwidth=rate, expert_slowdown_per_gib=0.0)


def _make_plan(
    model: ModelProfile,
    devices: list[Device],
    host: Host,
    context: int,
    mode: str,
    kv_type: str,
    depth: int | None,
    ubatch: int = UBATCH,
) -> Plan:
    """The fastest placement that fits, for one KV cache type and the host as given."""
    if kv_type not in KV_SCALE:
        raise PlanError(f"unknown KV cache type {kv_type!r}; use auto, {', '.join(KV_SCALE)}")
    if mode not in ("auto", "whole", "moe"):
        raise PlanError(f"unknown mode {mode!r}")
    modes = [mode] if mode != "auto" else (["whole", "moe"] if model.is_moe else ["whole"])
    if depth is None:
        depth = context // 2
    depth = max(0, min(depth, context))

    plans: list[Plan] = []
    for m in modes:
        for order in _orders(devices):
            plan = _plan_order(model, order, host, context, m, kv_type, depth, ubatch)
            if plan is not None:
                plans.append(plan)
    best: Plan | None = None
    if plans:
        # Prefer plans with no CPU layers, then the fastest one.
        best = min(plans, key=lambda p: (p.cpu_layers > 0, p.seconds_per_token))
        if streams_cpu_weights(best, model, devices):
            # Prompt batches stream the CPU's weights over the first GPU's link,
            # so among plans about as fast, take the one with the fastest link first
            # (phase 1a: Qwen3-32B pp 199 -> 278 tokens/s, tg 7.62 -> 7.54).
            close = [
                p for p in plans
                if (p.cpu_layers > 0 or p.expert_ram_bytes) and p.seconds_per_token <= best.seconds_per_token * (1 + PROMPT_TIE)
            ]
            fastest_link = max(close, key=lambda p: (_first_link(p, devices), -p.seconds_per_token))
            if _first_link(fastest_link, devices) > _first_link(best, devices):
                fastest_link.notes.append(
                    "the GPU with the fastest PCIe link goes first: prompt batches copy the CPU's "
                    "weights over it (slightly slower generation, much faster prompts)"
                )
                best = fastest_link
    if best is None:
        total = sum(d.usable_bytes for d in devices) + host.ram_bytes
        raise PlanError(
            f"{model.name} does not fit: it needs about {model.total_weight_bytes / 1024**3:.1f} GiB of weights "
            f"plus KV cache, and the GPUs and RAM offer {total / 1024**3:.1f} GiB"
        )
    if not devices:
        best.notes.append("no GPU: the whole model runs on this PC's CPU")
    elif best.cpu_layers:
        best.notes.append(
            f"{best.cpu_layers} layers stay on the CPU; they will dominate the time per token"
        )
    if best.mode == "moe":
        best.notes.append(
            "expert caching assumes uniform expert use, so the speed estimate is pessimistic"
        )
    best.notes.extend(remote_notes(best, devices))
    note = waiting_note(best, devices)
    if note:
        best.notes.append(note)
    best.notes.extend(model.notes)
    return best
