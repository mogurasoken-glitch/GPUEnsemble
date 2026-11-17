"""What the Windows app shows, as one JSON object (``state`` command).

The app (``app/`` in the repository, GPUEnsemble.exe in the Windows zip) is
a window over this package: it runs ``state`` for the GPUs, the other PCs and
the models, and starts ``up``, ``pull`` and ``worker`` as child processes for
the rest. Everything it reads from this package goes through ``build``, so the
keys below are the app's contract: add keys freely, but do not rename or drop
one without changing the app too.

    {"version": 1, "context": 8192, "home": "...", "models_dir": "...",
     "llama_cpp": true,
     "host": {"ram_gib": 48.0, "threads": 32},
     "gpus": [{"name": ..., "vram_gib": ..., "here": true, "pc": "this PC", "endpoint": null}],
     "workers": [{"name", "address", "endpoint", "allowed", "this_pc", "gpus": [{"name", "vram_gib"}]}],
     "models": [{"id", "name", "rank", "file_gib", "downloaded", "path", "fits", "speed",
                 "speed_empty", "there", "placement", "layout", "error", "cpu_layers",
                 "stages": [{"gpu", "first", "last", "used_gib", "capacity_gib"}]}],
     "best": "gpt-oss-120b" | null,
     "notes": ["..."]}

``stages`` is how the plan splits the model's layers over the GPUs (``gpu``
is the name in ``gpus``; ``first``/``last`` are null for a GPU that only holds
the output head), with what each will use; empty for a model that does not fit.
``layout`` says the same as ``placement`` as values (``gpus``, ``cpu_layers``,
``experts_on_gpu`` 0-1 or null, ``kv``, ``there``, ``faster_there`` tokens/s or
null), or is null for a model that does not fit.
``speed`` is the estimated tokens/s with half the context filled (what
``recommend`` ranks by), ``speed_empty`` the one of a new chat. Values the app
shows as text (``placement``, ``notes``) are English, as the rest of the engine's output.
"""

from __future__ import annotations

from .devices import GIB, Device, Host
from .plan import faster_on_other_pc
from .recommend import Candidate, _placement

VERSION = 1


def _gib(n: float) -> float:
    return round(n / GIB, 1)


def _rate(seconds: float) -> float:
    return round(1.0 / seconds, 1) if seconds > 0 else 0.0


def gpu_entry(d: Device, pcs: dict[str, str]) -> dict:
    """One card; ``pcs`` names the PC behind each RPC endpoint."""
    pc = "this PC" if d.rpc is None else pcs.get(d.rpc, d.rpc)
    return {"name": d.name, "vram_gib": _gib(d.vram_bytes), "here": d.rpc is None, "pc": pc, "endpoint": d.rpc}


def worker_entry(w, this_pc: bool) -> dict:
    return {
        "name": w.name,
        "address": w.address,
        "endpoint": w.endpoint,
        "allowed": bool(w.allowed),
        "this_pc": this_pc,
        "gpus": [{"name": g.get("name", "?"), "vram_gib": g.get("vram_gib")} for g in w.gpus],
    }


def stage_entries(c: Candidate) -> tuple[list[dict], int]:
    """The plan that will run: its GPUs with their layers, and the layers left on the CPU."""
    if c.plan is None:
        return [], 0
    p = c.plan.on_other_pc if c.there else c.plan
    stages = [{"gpu": s.device, "first": s.first_layer if s.n_layers else None,
               "last": s.last_layer if s.n_layers else None,
               "used_gib": _gib(s.used_bytes), "capacity_gib": _gib(s.capacity_bytes)} for s in p.stages]
    return stages, p.cpu_layers


def layout(c: Candidate) -> dict | None:
    """``placement`` as values, for the app to say in the user's language."""
    if c.plan is None:
        return None
    p = c.plan.on_other_pc if c.there else c.plan
    experts = None
    if p.mode == "moe":
        experts = round(sum(s.expert_fraction_on_gpu * s.n_layers for s in p.stages) / c.model.n_layers, 2)
    return {
        "gpus": sum(1 for s in p.stages if s.n_layers),
        "cpu_layers": p.cpu_layers,
        "experts_on_gpu": experts,
        "kv": p.kv_type,
        "there": bool(c.there),
        "faster_there": round(p.tokens_per_second_on_other_pc, 1) if not c.there and faster_on_other_pc(p) else None,
    }


def model_entry(c: Candidate, path: str | None) -> dict:
    fits = c.plan is not None
    stages, cpu_layers = stage_entries(c)
    return {
        "id": c.entry["id"],
        "name": c.entry.get("name", c.entry["id"]),
        "rank": c.entry.get("rank"),
        "new": bool(c.entry.get("new")),
        "mine": bool(c.entry.get("mine")),
        "hf_repo": c.entry.get("hf_repo"),
        "file_gib": c.entry.get("file_gib"),
        "downloaded": path is not None,
        "path": path,
        "fits": fits,
        "speed": _rate(c.seconds()) if fits else 0.0,
        "speed_empty": _rate(c.seconds("seconds_empty")) if fits else 0.0,
        "there": bool(c.there),
        "placement": _placement(c),
        "layout": layout(c),
        "error": c.error,
        "cpu_layers": cpu_layers,
        "stages": stages,
    }


def build(devices: list[Device], host: Host, candidates: list[Candidate], best: Candidate | None,
          local_paths: dict[str, str | None], workers: list[tuple[object, bool]], context: int,
          home: str, models_dir: str, llama_cpp: bool, notes: list[str] | None = None) -> dict:
    """The state as the app reads it.

    ``local_paths`` maps a catalog id to its downloaded file (or None);
    ``workers`` holds each worker that answered with whether it is this PC itself.
    """
    pcs = {d.rpc: d.rpc for d in devices if d.rpc}
    for w, _ in workers:
        for d in devices:
            if d.rpc and d.rpc.rsplit(":", 1)[0] == w.address:
                pcs[d.rpc] = w.name
    return {
        "version": VERSION,
        "context": context,
        "home": home,
        "models_dir": models_dir,
        "llama_cpp": llama_cpp,
        "host": {"ram_gib": _gib(host.ram_bytes), "threads": host.threads},
        "gpus": [gpu_entry(d, pcs) for d in devices],
        "workers": [worker_entry(w, this_pc) for w, this_pc in workers],
        "models": [model_entry(c, local_paths.get(c.entry["id"])) for c in candidates],
        "best": best.entry["id"] if best is not None else None,
        "notes": list(notes or []),
    }
