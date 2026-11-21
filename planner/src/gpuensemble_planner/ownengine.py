"""Arguments for GPUEnsemble's own engine (``gpue-server``, see ``engine/``), as an experiment.

The engine reads long prompts faster than llama.cpp when most expert weights
stay in RAM: it places the layers on the cards in turn and streams each layer's
experts to its card while the other card computes. It runs gpt-oss models on
NVIDIA cards of this PC only, and it is not part of the release: ``up --engine
own`` needs the path of a ``gpue-server`` built by hand.

The placement here is the one measured on the development PC (gpt-oss-120b on
an 11 GB and an 8 GB card: experts of the last five layers on the cards, the
other 31 in RAM), written as a rule of bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .devices import GIB, Device
from .llamacpp import LlamaCommand
from .model import ModelProfile
from .plan import PlanError

CHUNK = 4096  # tokens of a prompt chunk; the measurements used this
MIN_BATCH = 600  # shorter chunks compute their RAM experts on the CPU
# What a card needs beside weights and KV cache while a 4096-token chunk runs (measured: 0.7 GiB graph
# buffer on the 8 GB card, 0.9 GiB between ready and the peak on the 11 GB card).
GRAPH_BYTES = 1 * GIB
RING_SLOTS = 1
# The engine's measurements ran with 16 threads on a PC with 16 cores (32 with hyper-threading). One
# run with 32 generated at 6.6-6.7 tokens/s where runs with 16 gave 6.4-8.1: not compared properly yet.
MAX_THREADS = 16


def _ranges(layers: list[int]) -> str:
    """[0, 1, 2, 5] -> "0-2,5"."""
    out: list[str] = []
    for i in sorted(layers):
        if out and out[-1][1] == i - 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def _kv_bytes(model: ModelProfile, layer: int, context: int) -> int:
    lp = model.layers[layer]
    tokens = min(context, lp.kv_window) if lp.kv_window else context
    return lp.kv_bytes_per_token * tokens


@dataclass
class CardShare:
    """What one card holds."""
    device: str  # the card's name
    llama_device: str
    layers: list[int]
    expert_layers: list[int]  # those of ``layers`` whose experts are on the card
    weight_bytes: int  # the layers without their experts, the experts kept, the output on the last card
    kv_bytes: int
    work_bytes: int  # the slot for experts streamed from RAM and the buffer a prompt chunk needs
    copy_bytes: int  # the other cards' attention, kept a second time for generated tokens
    capacity_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes + self.work_bytes + self.copy_bytes


@dataclass
class Placement:
    """Where the own engine puts ``model``: the layers on the cards in turn, most experts in RAM."""
    model: str
    context: int
    cards: list[CardShare]
    host_expert_layers: list[int]  # layers whose experts stay in RAM
    host_expert_bytes: int
    gather: int | None  # index in ``cards`` of the card that computes the attention of generated tokens
    notes: list[str] = field(default_factory=list)


def engine_placement(model: ModelProfile, devices: list[Device], context: int,
                     ram_bytes: int | None = None) -> Placement:
    """The own engine's placement of ``model`` on the local NVIDIA cards among ``devices``.

    ``ram_bytes`` is the RAM that weights may fill (``Host.ram_bytes``); without it RAM is not checked.
    Raises ``PlanError`` when the engine cannot run it.
    """
    if model.architecture != "gpt-oss" or not model.is_moe:
        raise PlanError(f"the own engine runs gpt-oss models only (this one is {model.architecture})")
    cards = [d for d in devices if not d.is_remote and (d.llama_device or "").startswith("CUDA")]
    if not cards:
        raise PlanError("the own engine needs an NVIDIA card in this PC")
    # the smallest card first: the layers go to the cards in turn, and the last layer and the output
    # (the larger share) then land on the largest card
    cards.sort(key=lambda d: d.usable_bytes)
    n = model.n_layers
    on_card: list[list[int]] = [[] for _ in cards]
    for il in range(n):
        on_card[il % len(cards)].append(il)
    last = (n - 1) % len(cards)
    slot = max(lp.expert_bytes for lp in model.layers)

    host_experts = set(range(n))
    shares: list[CardShare] = []
    left: list[int] = []  # per card: bytes free after everything below
    for i, dev in enumerate(cards):
        weights = sum(model.layers[il].dense_bytes for il in on_card[i]) + (model.head_bytes if i == last else 0)
        kv = sum(_kv_bytes(model, il, context) for il in on_card[i])
        work = GRAPH_BYTES + RING_SLOTS * slot
        room = dev.usable_bytes - (weights + kv + work)
        if room < 0:
            raise PlanError(f"{dev.name} is too small for the own engine with this model "
                            f"({(weights + kv + work) / GIB:.1f} GiB needed before any expert, "
                            f"{dev.usable_bytes / GIB:.1f} GiB usable)")
        kept: list[int] = []
        for il in reversed(on_card[i]):  # the last layers first, as measured
            if model.layers[il].expert_bytes > room:
                break
            room -= model.layers[il].expert_bytes
            weights += model.layers[il].expert_bytes
            host_experts.discard(il)
            kept.append(il)
        left.append(room)
        shares.append(CardShare(dev.name, dev.llama_device, on_card[i], sorted(kept), weights, kv, work, 0,
                                dev.usable_bytes))

    # Generated tokens compute their attention on the last card alone. With the layers on the cards in
    # turn, a generated token makes each card work in short bursts, and a card that drives a display
    # clocks down between them (engine/README.md: 7.3 -> 9.9 tokens/s on the development PC). The card
    # then holds a second copy of the other cards' layers whose experts are in RAM, without the experts,
    # and of their keys and values. No expert leaves a card for this: without the room, generation keeps
    # the placement, and a small kernel keeps the last card busy instead (8.4-8.9 tokens/s there).
    # With the attention gathered, the card holds its clocks through an answer only when it starts the
    # answer with them up: after a short further question it started low and stayed low (7.9 tokens/s).
    # The same small kernel therefore runs for the first two seconds of every answer (the engine sees
    # to that itself when GPUE_AWAKE asks for a backend of its own).
    gather = None
    if len(cards) > 1:
        twins = sum(model.layers[il].dense_bytes + _kv_bytes(model, il, context)
                    for il in sorted(host_experts) if il not in on_card[last])
        if twins <= left[last]:
            gather = last
            shares[last].copy_bytes = twins

    # The experts that stay off the cards are read into RAM whole (--ram). On the development PC 49.0 GiB
    # of them left about 1 GiB of 64 free while a long prompt ran, so no more than the planner's own
    # share of RAM is accepted.
    in_ram = sum(model.layers[il].expert_bytes for il in host_experts)
    if ram_bytes is not None and in_ram > ram_bytes:
        raise PlanError(f"the own engine would keep {in_ram / GIB:.1f} GiB of experts in RAM with these cards, "
                        f"and this PC has {ram_bytes / GIB:.1f} GiB for weights")

    notes = []
    if gather is not None:
        notes.append(f"generated tokens compute their attention on {shares[gather].device}")
    notes.append("the model is read into RAM; loading takes several minutes")
    return Placement(model.name, context, shares, sorted(host_experts), in_ram, gather, notes)


def engine_command(model: ModelProfile, devices: list[Device], context: int, threads: int | None = None,
                   placement: Placement | None = None) -> LlamaCommand:
    """The ``gpue-server`` arguments and environment for ``model`` on the local NVIDIA cards among ``devices``.

    Raises ``PlanError`` when the engine cannot run it. ``cpu_expert_layers``
    of the result lists the layers whose experts stay in RAM.
    """
    p = placement or engine_placement(model, devices, context)
    # --layers counts the last layers of the model and the output; with --alternate only the total
    # and one count per card matter
    counts = [len(share.layers) for share in p.cards]
    counts[(model.n_layers - 1) % len(p.cards)] += 1
    args = ["--devices", ",".join(share.llama_device for share in p.cards), "--layers", ",".join(str(c) for c in counts),
            "--alternate", "--ring", str(RING_SLOTS), "--ram", "--flash",
            "--min-batch", str(MIN_BATCH), "--chunk", str(CHUNK), "-c", str(context)]
    if p.host_expert_layers:
        args += ["--host-experts", _ranges(p.host_expert_layers)]
    if p.gather is not None:
        args += ["--gen-device", str(p.gather)]
    if threads:
        args += ["-t", str(min(threads, MAX_THREADS))]
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "GGML_OP_OFFLOAD_MIN_BATCH": str(MIN_BATCH),
        # A small kernel that keeps the last card busy while tokens are generated, so its clocks stay
        # up. With two or more cards it runs on a backend of its own (the fifth field), because the
        # engine holds the cards for a whole token: all through an answer when generation keeps the
        # placement, at the start of an answer when the attention is gathered on the last card.
        "GPUE_AWAKE": f"0,512,{len(p.cards) - 1}" + (",0,1" if len(p.cards) > 1 else ""),
    }
    return LlamaCommand(args=args, env=env, cpu_expert_layers=list(p.host_expert_layers), notes=list(p.notes))
