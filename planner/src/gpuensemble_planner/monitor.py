"""What a running model is doing, card by card (``monitor`` command).

``up`` writes what it started to a run file (``RUN_FILE`` under the
GPUEnsemble home): the API's address, the model and the plan's stages with
the time each one is estimated to take per token. It removes the file when
the model stops. ``monitor`` reads that file, asks nvidia-smi for each card's
memory, temperature, clocks and load, and asks llama-server (started with
``--metrics``) how many tokens it has generated, and in how long.

llama.cpp does not report how long each card takes per token, so the time per
stage is the plan's estimate, scaled so that the stages add up to the measured
time per token when there is one. The card that takes the longest is the one
holding the others back (``slowest``). A card that works on the model in a
low performance state (P3 to P8) has been slowed down by the driver
(``clock_down``): with a split model that is what makes a run slower than
its estimate (benchmarks/clock-hold.md). It is only said of a card that takes
a tenth or more of the time per token: on the E5 rig a card behind 49 CPU
layers idles 97% of each token, and its clocks then cost nothing.

``snapshot`` is the JSON the Windows app reads (``monitor --json``); add keys
freely, but do not rename or drop one without changing the app too:

    {"version": 1, "running": true, "model": "gpt-oss-120b", "api": "http://127.0.0.1:8080/v1",
     "ready": true, "busy": false, "speed": 13.4 | null, "prompt_speed": 210.0 | null, "estimate": 13.1 | null,
     "tokens": 1234, "slowest": "GPU0 NVIDIA GeForce RTX 2080 Ti" | null, "slowest_cpu_experts": false,
     "stages": [{"gpu", "here", "index", "first", "last", "ms", "share", "cpu_experts"}],
     "cpu": {"layers": 4, "ms": 12.0, "share": 0.1} | null,
     "gpus": [{"index", "name", "used_gib", "total_gib", "temp_c", "load", "clock_mhz", "max_clock_mhz",
               "power_w", "pstate", "layers": "0-17" | null, "clock_down": false, "hot": false}],
     "notes": ["..."]}

``speed`` and ``prompt_speed`` are tokens/s of the answers that finished
since the last reading with any (llama-server adds an answer to its counters
when it ends); they keep their value while the model is idle, and are null
until the first answer. ``up``'s warm-up requests are left out: when the model
is ready it adds the counters of that moment to the run file (``warm``), and
the speeds count from there. Before that, they are null.

``cpu_experts`` marks a stage of an MoE model whose time includes experts that
stay in RAM and that the CPU computes (the plan counts them with the card that
holds the layers); ``slowest_cpu_experts`` says so of the slowest stage, which
is then slow because of the CPU, not the card.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Callable

from .childproc import pid_alive
from .devices import GIB, Device
from .plan import Plan

VERSION = 1
GPU_FIELDS = ("index,name,memory.used,memory.total,temperature.gpu,utilization.gpu,"
              "clocks.sm,clocks.max.sm,power.draw,pstate")
# A card that works on the model in this performance state or a deeper one is reported as slowed
# down (P0-P2 are the working states; the driver's "GPU idle" states start at P3). A card that does
# not tell its state is judged by its clock: below this share of its top clock.
CLOCK_DOWN_PSTATE = 3
CLOCK_DOWN = 0.5
# ... and only when its stage takes at least this share of the time per token.
CLOCK_DOWN_SHARE = 0.1
# The temperature at which the cards of the dev rigs start to lower their clocks themselves.
HOT_C = 83
# Load (%) above which a card counts as working on the model right now.
BUSY_LOAD = 20


def run_file() -> str:
    from .hub import home_dir

    return os.path.join(home_dir(), "run", "up.json")


def _gpu_index(d: Device) -> int | None:
    name = d.llama_device or ""
    if d.rpc is None and name.startswith("CUDA") and name[4:].isdigit():
        return int(name[4:])
    return None


def run_info(plan: Plan, devices: list[Device], model: str, host: str, port: int) -> dict:
    """What ``up`` writes to the run file: the API and the stages as planned."""
    by_name = {d.name: d for d in devices}
    stages = []
    for s in plan.stages:
        d = by_name.get(s.device)
        stages.append({
            "gpu": s.device,
            "here": d is None or d.rpc is None,
            "index": _gpu_index(d) if d is not None else None,
            "first": s.first_layer if s.n_layers else None,
            "last": s.last_layer if s.n_layers else None,
            "seconds": s.seconds_per_token,
            # gpt-oss and other MoE models: the CPU computes the experts of this stage's layers that stay in
            # RAM, and the plan counts that time in the stage.
            "cpu_experts": s.expert_path == "cpu",
        })
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return {
        "version": VERSION,
        "model": model,
        "host": shown,
        "port": port,
        "pid": os.getpid(),
        "started": time.time(),
        "seconds_per_token": plan.seconds_per_token,
        "cpu_layers": plan.cpu_layers,
        "stages": stages,
    }


def write_run(info: dict, path: str | None = None) -> str:
    path = path or run_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def clear_run(path: str | None = None) -> None:
    try:
        os.remove(path or run_file())
    except OSError:
        pass


def read_run(path: str | None = None, alive: Callable[[int], bool] = pid_alive) -> dict | None:
    """The running model as ``up`` wrote it, or None (no file, or the ``up`` that wrote it is gone).

    An ``up`` that was killed (the app's Stop button, Task Manager) cannot
    remove its file; one found with a dead process is removed here.
    """
    path = path or run_file()
    try:
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    pid = info.get("pid")
    if isinstance(pid, int) and not alive(pid):
        clear_run(path)
        return None
    return info


def mark_ready(metrics: dict[str, float] | None, path: str | None = None) -> None:
    """Called by ``up`` when the model is ready: the counters as the warm-up left them go into the run file."""
    path = path or run_file()
    try:
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        return
    info["ready"] = time.time()
    info["warm"] = {k: (metrics or {}).get(k, 0.0) for k in (TOKENS, TOKEN_SECONDS, PROMPT, PROMPT_SECONDS)}
    write_run(info, path)


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:  # "[N/A]", "[Not Supported]"
        return None


def parse_gpus(out: str) -> list[dict]:
    """Cards from ``nvidia-smi --query-gpu=<GPU_FIELDS> --format=csv,noheader,nounits``."""
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        used, total = _number(parts[2]), _number(parts[3])
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "used_gib": round(used / 1024, 1) if used is not None else None,
            "total_gib": round(total / 1024, 1) if total is not None else None,
            "temp_c": _number(parts[4]),
            "load": _number(parts[5]),
            "clock_mhz": _number(parts[6]),
            "max_clock_mhz": _number(parts[7]),
            "power_w": _number(parts[8]),
            "pstate": parts[9],
        })
    return gpus


def read_gpus(run=subprocess.run) -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi was not found")
    out = run([exe, f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
              capture_output=True, text=True, check=True, timeout=10).stdout
    return parse_gpus(out)


_METRIC = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text as llama-server's ``/metrics`` gives it: name -> value (labels dropped)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC.match(line)
        if m:
            value = _number(m.group(3))
            if value is not None:
                out[m.group(1)] = value
    return out


def read_metrics(host: str, port: int, opener=urllib.request.urlopen, timeout: float = 2.0) -> dict[str, float] | None:
    """llama-server's counters, or None when it does not answer (still loading, or no ``--metrics``)."""
    try:
        with opener(f"http://{host}:{port}/metrics", timeout=timeout) as r:
            return parse_metrics(r.read().decode("utf-8", "replace"))
    except (OSError, urllib.error.URLError, ValueError):
        return None


TOKENS = "llamacpp:tokens_predicted_total"
TOKEN_SECONDS = "llamacpp:tokens_predicted_seconds_total"
PROMPT = "llamacpp:prompt_tokens_total"
PROMPT_SECONDS = "llamacpp:prompt_seconds_total"
PROCESSING = "llamacpp:requests_processing"


class Rates:
    """Tokens/s from the change of llama-server's counters between two readings.

    The counters add the time spent generating, not the wall time, so the
    pauses between answers do not lower the speed. With nothing generated
    since the last reading, the last speed stays. The first reading counts
    from the counters ``up`` noted when the model was ready (its warm-up
    requests are not answers); for a model that is not ready yet, or another
    model than at the last reading, the speeds start again from nothing.
    """

    def __init__(self) -> None:
        self.run: tuple | None = None
        self.last: dict[str, float] | None = None
        self.speed: float | None = None
        self.prompt_speed: float | None = None

    @staticmethod
    def _rate(now: dict, last: dict, count: str, seconds: str) -> float | None:
        dn = now.get(count, 0.0) - last.get(count, 0.0)
        ds = now.get(seconds, 0.0) - last.get(seconds, 0.0)
        if dn > 0 and ds > 0:
            return dn / ds
        return None

    def update(self, metrics: dict[str, float] | None, run: dict | None = None) -> None:
        key = (run.get("pid"), run.get("started"), run.get("ready")) if run else None
        if key != self.run:
            self.run = key
            self.speed = self.prompt_speed = None
            self.last = dict(run["warm"]) if run and isinstance(run.get("warm"), dict) else None
        if metrics is None or run is None or self.last is None:
            return
        s = self._rate(metrics, self.last, TOKENS, TOKEN_SECONDS)
        if s is not None:
            self.speed = s
        p = self._rate(metrics, self.last, PROMPT, PROMPT_SECONDS)
        if p is not None:
            self.prompt_speed = p
        self.last = metrics


def _speed(x: float | None) -> float | None:
    """Tokens/s as shown: two decimals below 10 (1.37 is not 1.4), one above."""
    if not x:
        return None
    return round(x, 2 if x < 10 else 1)


def _pstate(text) -> int | None:
    text = str(text or "")
    return int(text[1:]) if text[:1] == "P" and text[1:].isdigit() else None


def _layers(first, last) -> str | None:
    if first is None or last is None:
        return None
    return str(first) if first == last else f"{first}-{last}"


def snapshot(run: dict | None, gpus: list[dict], metrics: dict[str, float] | None, rates: Rates,
             notes: list[str] | None = None) -> dict:
    """The monitor's reading as one JSON object (see the module's docstring)."""
    notes = list(notes or [])
    stages_out: list[dict] = []
    cpu = None
    slowest = None
    slowest_cpu_experts = False
    estimate = None
    speed = _speed(rates.speed)
    busy = bool(metrics and metrics.get(PROCESSING, 0) > 0)
    by_index: dict[int, dict] = {}
    if run is not None:
        total = float(run.get("seconds_per_token") or 0.0)
        estimate = _speed(1.0 / total) if total > 0 else None
        stage_sum = sum(float(s.get("seconds") or 0.0) for s in run.get("stages", []))
        cpu_seconds = max(0.0, total - stage_sum)
        # Scale the plan's split of the time to the time measured, when there is one.
        scale = (1.0 / rates.speed) / total if rates.speed and total > 0 else 1.0
        for s in run.get("stages", []):
            sec = float(s.get("seconds") or 0.0)
            entry = {
                "gpu": s.get("gpu"),
                "here": bool(s.get("here", True)),
                "index": s.get("index"),
                "first": s.get("first"),
                "last": s.get("last"),
                "ms": round(sec * scale * 1000, 1),
                "share": round(sec / total, 2) if total > 0 else None,
                "cpu_experts": bool(s.get("cpu_experts")),
            }
            stages_out.append(entry)
            if entry["index"] is not None:
                by_index[entry["index"]] = entry
        if run.get("cpu_layers"):  # the rest of the time without CPU layers is waiting (network, start of a step)
            cpu = {"layers": int(run.get("cpu_layers") or 0), "ms": round(cpu_seconds * scale * 1000, 1),
                   "share": round(cpu_seconds / total, 2) if total > 0 else None}
        timed = [s for s in stages_out if s["first"] is not None]
        if timed:
            top = max(timed, key=lambda s: s["ms"])
            if cpu is None or top["ms"] >= cpu["ms"]:
                slowest = top["gpu"]
                slowest_cpu_experts = top["cpu_experts"]
            else:
                slowest = "CPU"
    gpus_out = []
    for g in gpus:
        stage = by_index.get(g["index"])
        load = g.get("load") or 0
        clock, top = g.get("clock_mhz"), g.get("max_clock_mhz")
        working = stage is not None and stage["first"] is not None and (busy or load >= BUSY_LOAD)
        matters = working and (stage["share"] is None or stage["share"] >= CLOCK_DOWN_SHARE)
        state = _pstate(g.get("pstate"))
        if state is not None:
            lowered = state >= CLOCK_DOWN_PSTATE
        else:
            lowered = bool(clock and top and clock < CLOCK_DOWN * top)
        clock_down = bool(matters and lowered)
        temp = g.get("temp_c")
        out = dict(g)
        out["layers"] = _layers(stage["first"], stage["last"]) if stage else None
        out["clock_down"] = clock_down
        out["hot"] = bool(temp is not None and temp >= HOT_C)
        gpus_out.append(out)
        if clock_down:
            notes.append(f"{g['name']} is in {g.get('pstate') or 'a low state'} at {_v(clock)} of {_v(top)} MHz "
                         "while it works on the model: the driver has lowered its clocks")
        if out["hot"]:
            notes.append(f"{g['name']} is at {temp:.0f} C")
    if run is not None and speed and estimate and speed < 0.8 * estimate:
        notes.append(f"generating at {speed:g} tokens/s, below the estimate of {estimate:g}")
    api = f"http://{run['host']}:{run['port']}/v1" if run else None
    return {
        "version": VERSION,
        "running": run is not None,
        "model": run.get("model") if run else None,
        "api": api,
        "busy": busy,
        "speed": speed,
        "prompt_speed": _speed(rates.prompt_speed),
        "estimate": estimate,
        "ready": bool(run and run.get("ready")),
        "tokens": int(metrics.get(TOKENS, 0)) if metrics else None,
        "slowest": slowest,
        "slowest_cpu_experts": slowest_cpu_experts,
        "stages": stages_out,
        "cpu": cpu,
        "gpus": gpus_out,
        "notes": notes,
    }


def _v(x, fmt: str = "{:.0f}", none: str = "-") -> str:
    return none if x is None else fmt.format(x)


def format_text(snap: dict) -> str:
    """The reading as a few lines for a console."""
    lines = []
    if snap["running"]:
        if not snap["ready"]:
            now = "loading or warming up"
        elif snap["speed"] is None:
            now = "answering its first request" if snap["busy"] else "ready, no answer yet"
        elif snap["busy"]:
            now = f"answering; the answers before this one ran at {snap['speed']:g} tokens/s"
        else:
            now = f"idle; the last answers ran at {snap['speed']:g} tokens/s"
        lines.append(f"{snap['model']} at {snap['api']}: {now} (estimate {_v(snap['estimate'], '{:g}')})")
        if snap["slowest"]:
            with_cpu = " (with the experts of its layers that the CPU computes)" if snap["slowest_cpu_experts"] else ""
            lines.append(f"slowest part: {snap['slowest']}{with_cpu}")
        for s in snap["stages"]:
            where = "" if s["here"] else " (other PC)"
            experts = ", experts on the CPU" if s["cpu_experts"] else ""
            lines.append(f"  {s['gpu']}{where}: layers {_layers(s['first'], s['last']) or 'head only'}{experts}, "
                         f"~{s['ms']:.1f} ms per token")
        if snap["cpu"]:
            lines.append(f"  CPU: {snap['cpu']['layers']} layers, ~{snap['cpu']['ms']:.1f} ms per token")
    else:
        lines.append("no model is running (start one with up)")
    for g in snap["gpus"]:
        flags = (" CLOCKED DOWN" if g["clock_down"] else "") + (" HOT" if g["hot"] else "")
        lines.append(f"GPU{g['index']} {g['name']}: {_v(g['used_gib'], '{:.1f}')}/{_v(g['total_gib'], '{:.1f}')} GiB, "
                     f"{_v(g['temp_c'])} C, load {_v(g['load'])}%, {_v(g['clock_mhz'])}/{_v(g['max_clock_mhz'])} MHz, "
                     f"{_v(g['power_w'])} W {g['pstate']}{flags}")
    for n in snap["notes"]:
        lines.append(f"note: {n}")
    return "\n".join(lines)


def watch(every: float, count: int, emit: Callable[[dict], None], *, read_run_=None, read_gpus_=None,
          read_metrics_=None, sleep=time.sleep) -> None:
    """Take ``count`` readings (0: until stopped), ``every`` seconds apart, and hand each to ``emit``."""
    read_run_ = read_run_ or read_run
    read_gpus_ = read_gpus_ or read_gpus
    read_metrics_ = read_metrics_ or read_metrics
    rates = Rates()
    n = 0
    while True:
        notes = []
        run = read_run_()
        try:
            gpus = read_gpus_()
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            gpus = []
            notes.append(f"could not read the GPUs: {e}")
        metrics = read_metrics_(run["host"], run["port"]) if run else None
        rates.update(metrics, run)
        emit(snapshot(run, gpus, metrics, rates, notes))
        n += 1
        if count and n >= count:
            return
        sleep(every)
