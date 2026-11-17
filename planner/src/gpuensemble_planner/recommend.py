"""Pick the best model this PC can run (the first step of "install and go").

Every model in the catalog (``catalog.json``, editable) is planned on the
detected or given hardware. The recommendation is the best-ranked model whose
estimated generation speed, with half the context filled, reaches a minimum
(5 tokens/s by default: about reading speed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .devices import GIB, Device, Host
from .model import ModelProfile
from .catalogs import BUNDLED as CATALOG, load_all
from .plan import Plan, PlanError, faster_on_other_pc, make_plan, serving_pc
from .remote import profile_of
MIN_SPEED = 5.0  # tokens/s with half the context filled


@dataclass
class Candidate:
    entry: dict
    model: ModelProfile
    plan: Plan | None
    error: str = ""
    # Another PC holds the whole model on its cards alone, runs models itself when asked and is the
    # faster place for it: `up` starts it there, and the speeds are those of that PC's own plan,
    # without the network between the two (plan.serving_pc).
    there: bool = False

    @property
    def speed(self) -> float:
        return _rate(self.seconds())

    def seconds(self, of: str = "seconds_per_token") -> float:
        """Seconds per token as `up` will run the model: ``of`` is the plan's field for the context filling."""
        if self.plan is None:
            return 0.0
        return getattr(self.plan.on_other_pc if self.there else self.plan, of)


def _rate(seconds: float) -> float:
    return 1.0 / seconds if seconds > 0 else 0.0


def load_catalog(path: str | None = None) -> list[dict]:
    """The models to choose from: a given file, else the checked list and the models found on Hugging Face
    (``catalogs.load_all``)."""
    if path:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        entries = data["models"] if isinstance(data, dict) else data
    else:
        entries = load_all()
    return sorted(entries, key=lambda e: e.get("rank", 999))


def evaluate(entries: list[dict], devices: list[Device], host: Host, context: int) -> list[Candidate]:
    """Plan every catalog model on this hardware.

    A model with no profile written in the list and none read yet (``catalog --update`` reads them)
    is left out: planning does not go on the network.
    """
    out = []
    for entry in entries:
        try:
            model = profile_of(entry, fetch=False)
        except Exception:  # not read yet, or a kept copy this version cannot use
            continue
        if not entry.get("file_gib"):
            entry = {**entry, "file_gib": round(model.total_weight_bytes / GIB, 2)}
        try:
            plan = make_plan(model, devices, host, context=context)
            out.append(Candidate(entry, model, plan, there=serving_pc(plan, devices) is not None))
        except PlanError as e:
            out.append(Candidate(entry, model, None, str(e)))
    return out


def pick(candidates: list[Candidate], min_speed: float = MIN_SPEED) -> Candidate | None:
    """The best-ranked model that fits and is fast enough, else the fastest one that fits."""
    fitting = [c for c in candidates if c.plan is not None]
    for c in sorted(fitting, key=lambda c: c.entry.get("rank", 999)):
        if c.speed >= min_speed:
            return c
    return max(fitting, key=lambda c: c.speed, default=None)


def _placement(c: Candidate) -> str:
    if c.plan is None:
        return "does not fit"
    p = c.plan.on_other_pc if c.there else c.plan  # the plan that will run
    gpus = sum(1 for s in p.stages if s.n_layers)
    parts = [f"{gpus} GPU" + ("s" if gpus > 1 else "")]
    if p.cpu_layers:
        parts.append(f"{p.cpu_layers} layers on CPU")
    if p.mode == "moe":
        on_gpu = sum(s.expert_fraction_on_gpu * s.n_layers for s in p.stages)
        parts.append(f"experts {on_gpu / c.model.n_layers:.0%} on GPU")
    if p.kv_type != "f16":
        parts.append(f"KV {p.kv_type}")
    if c.there:
        parts.append("runs on the other PC")
    elif faster_on_other_pc(p):
        parts.append(f"{p.tokens_per_second_on_other_pc:.1f} if started on the other PC")
    return ", ".join(parts)


def format_table(candidates: list[Candidate], best: Candidate | None, context: int, min_speed: float) -> str:
    lines = [
        f"context {context} tokens; speeds are estimated tokens/s with an empty / half-full / full context",
        "",
        f"   {'model':<32} {'size GiB':>8}  {'speed':>16}  placement",
    ]
    for c in candidates:
        mark = "*" if c is best else " "
        size = c.entry.get("file_gib") or c.model.total_weight_bytes / GIB
        if c.plan:
            empty, half, full = (_rate(c.seconds(of)) for of in ("seconds_empty", "seconds_per_token", "seconds_full"))
            speed = f"{empty:4.1f} /{half:5.1f} /{full:5.1f}"
        else:
            speed = "-"
        lines.append(f" {mark} {c.entry['name']:<32} {size:8.1f}  {speed:>16}  {_placement(c)}")
    lines.append("")
    if best is None:
        lines.append("nothing in the catalog fits on this PC")
    elif best.speed >= min_speed:
        lines.append(f"recommended: {best.entry['name']} (the best-ranked model at {min_speed:g}+ tokens/s)")
    else:
        lines.append(f"recommended: {best.entry['name']} (nothing reaches {min_speed:g} tokens/s; this is the fastest that fits)")
    return "\n".join(lines)
