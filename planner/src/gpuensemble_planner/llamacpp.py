"""Turn a plan into llama.cpp arguments (phase 1a).

Until the GPUEnsemble executor exists, the plan is carried out by stock
llama.cpp. That already removes the manual part of running a big model on
mismatched cards: no guessing ``-ngl``, ``-ts`` or ``-ncmoe``.

How a plan maps onto llama.cpp
------------------------------
* llama.cpp counts the output head as one more offloaded layer and gives it
  to the last device. So the CPU prefix (layers 0..c-1) is ``-ngl
  n_layers - c + 1`` (``-ngl 999`` when c is 0), and the last ``-ts`` entry
  gets one extra. Checked on the E5 rig: the load log then matches the plan
  layer for layer.
* ``-dev`` lists the GPUs in stage order and ``-ts`` gives each its exact
  number of layers, so llama.cpp splits the offloaded layers the same way.
* In ``moe`` mode, routed experts that the plan leaves in RAM are pinned to
  the CPU with ``-ot`` (the same mechanism as ``-ncmoe``). A layer's experts
  are one tensor each, so they move as a whole layer: per stage, the last
  layers keep their experts on the GPU until the stage's budget is spent.
* ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` makes CUDA number the cards the way
  nvidia-smi does, so ``CUDA0`` means the same card everywhere.
* Cards on other PCs are served there by ``ggml-rpc-server`` and reached with
  ``--rpc host:port,...``. llama.cpp names their devices RPC0, RPC1, ... in
  the order of ``--rpc`` and, per server, in the order of the server's ``-d``
  (checked on the E5 rig: ``RPC0: 127.0.0.1:50052``). Only the servers the
  plan uses are listed, so a PC that is switched off does not stop a plan
  that leaves it out.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field

from .childproc import spawn
from .devices import Device
from .model import ModelProfile
from .plan import Plan, streams_cpu_weights

ALL_LAYERS = 999


@dataclass
class LlamaCommand:
    args: list[str]
    env: dict[str, str]
    cpu_expert_layers: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rpc_servers: list[str] = field(default_factory=list)  # host:port of each --rpc server

    def shell(self, exe: str, windows: bool = False, model: str | None = None,
              extra: list[str] | None = None) -> str:
        """The command as one line to paste into a shell; with ``model`` and
        ``extra``, exactly what ``run`` starts."""
        args = (["-m", model] if model else []) + self.args + list(extra or [])
        if windows:
            env = "".join(f"$env:{k}='{v}'; " for k, v in self.env.items())
            quoted = " ".join(_ps_quote(a) for a in args)
            return f"{env}& {_ps_quote(exe)} {quoted}"
        env = "".join(f"{k}={shlex.quote(v)} " for k, v in self.env.items())
        return env + " ".join(shlex.quote(a) for a in [exe, *args])


def _ps_quote(s: str) -> str:
    if s and all(c.isalnum() or c in "-_./:\\=," for c in s):
        return s
    return "'" + s.replace("'", "''") + "'"


def _expert_layers_on_cpu(model: ModelProfile, plan: Plan) -> list[int]:
    """Layers whose routed experts stay in RAM, at whole-layer granularity."""
    on_cpu: list[int] = [i for i in range(plan.cpu_layers) if model.layers[i].expert_bytes]
    if plan.mode != "moe":
        return on_cpu
    for stage in plan.stages:
        if stage.n_layers == 0:
            continue
        layers = range(stage.first_layer, stage.last_layer + 1)
        total = sum(model.layers[i].expert_bytes for i in layers)
        budget = stage.expert_fraction_on_gpu * total
        # Keep the last layers of the stage on the GPU while they fit.
        for i in reversed(layers):
            size = model.layers[i].expert_bytes
            if size and size <= budget + 1:  # +1: float rounding when everything fits
                budget -= size
            elif size:
                on_cpu.append(i)
    return sorted(on_cpu)


def _layer_regex(layers: list[int]) -> str:
    return r"blk\.(" + "|".join(str(i) for i in layers) + r")\.ffn_.*_exps\."


def rpc_device_names(plan: Plan, devices: list[Device]) -> tuple[list[str], dict[str, str]]:
    """The ``--rpc`` servers a plan needs, and the llama.cpp name of each remote card.

    Every card a listed server serves gets a name, used or not, because the
    server offers all of them; the rig lists each server's cards in the order
    of that server's ``-d``.
    """
    by_name = {d.name: d for d in devices}
    used = {by_name[s.device].rpc for s in plan.stages if s.n_layers} - {None}
    servers: list[str] = []
    for d in devices:
        if d.rpc and d.rpc in used and d.rpc not in servers:
            servers.append(d.rpc)
    names: dict[str, str] = {}
    for server in servers:
        for d in devices:
            if d.rpc == server:
                names[d.name] = f"RPC{len(names)}"
    return servers, names


def llama_command(
    plan: Plan,
    model: ModelProfile,
    devices: list[Device],
    tool: str = "server",
    threads: int | None = None,
    ram_bytes: int | None = None,
) -> LlamaCommand:
    """Arguments for ``llama-server`` / ``llama-cli`` (tool="server") or ``llama-bench``.

    llama-bench reads a comma as "try several values", so its ``-ts`` and
    ``-dev`` use ``/`` and the context size is left to its own ``-p``/``-n``.
    """
    if tool not in ("server", "bench"):
        raise ValueError(f"tool must be server or bench, not {tool!r}")
    by_name = {d.name: d for d in devices}
    stages = [s for s in plan.stages if s.n_layers > 0]
    servers, rpc_names = rpc_device_names(plan, devices)
    names = []
    for s in stages:
        dev = by_name[s.device]
        if dev.is_remote:
            names.append(rpc_names[dev.name])
            continue
        if not dev.llama_device:
            raise ValueError(f"{dev.name} has no llama.cpp device name; set llama_device in the rig file")
        names.append(dev.llama_device)

    # +1: llama.cpp offloads the output head as a layer of the last device.
    ngl = ALL_LAYERS if plan.cpu_layers == 0 else model.n_layers - plan.cpu_layers + 1
    sep = "/" if tool == "bench" else ","
    # --rpc first: llama.cpp reads the arguments in order, and --list-devices
    # added after it then shows the remote cards too.
    args = ["--rpc", ",".join(servers)] if servers else []
    args += ["-ngl", str(ngl), "-sm", "layer", "-dev", sep.join(names)]
    if len(stages) > 1:
        counts = [s.n_layers for s in stages]
        counts[-1] += 1
        args += ["-ts", sep.join(str(c) for c in counts)]
    if tool == "server":
        args += ["-c", str(plan.context)]
    else:
        # llama-server turns flash attention on by itself (auto); llama-bench leaves it off unless told,
        # and then reads prompts 17% slower (Llama 3.3 70B on the E5 rig: pp512 29.2 against 34.0).
        args += ["-fa", "1"]
    if plan.kv_type != "f16":
        # A quantized V cache needs flash attention, which llama.cpp turns on by default (auto).
        args += ["-ctk", plan.kv_type, "-ctv", plan.kv_type]
    if threads and (plan.cpu_layers or (model.is_moe and plan.mode == "moe")):
        args += ["-t", str(threads)]

    notes: list[str] = []
    cpu_experts = _expert_layers_on_cpu(model, plan) if model.is_moe else []
    if cpu_experts:
        args += ["-ot", f"{_layer_regex(cpu_experts)}=CPU"]
    if (cpu_experts or plan.cpu_layers) and not streams_cpu_weights(plan, model):
        # Compute CPU-resident experts on the CPU during prompts too, instead of
        # copying them to the first GPU for every batch (phase 1a: pp 2.6x).
        args += ["-nopo", "1"] if tool == "bench" else ["--no-op-offload"]
        gpu_moe = [i for s in stages for i in range(s.first_layer, s.last_layer + 1) if i not in cpu_experts]
        notes.append(f"experts of {len(cpu_experts)} layers run on the CPU; {len(gpu_moe)} layers keep them on a GPU")
    if tool == "server" and ram_bytes and model.total_weight_bytes > ram_bytes:
        # The model is larger than the RAM set aside for weights. With mmap the
        # whole file stays mapped (GPU tensors too) and the OS keeps evicting
        # CPU expert pages: on the E5 rig gpt-oss-120b fell to 6.7 tok/s on new
        # prompts and prompt speed to 1-6 tok/s. Reading it into RAM instead
        # (and not into pinned host buffers, which ran out of memory) keeps
        # 7.8-8.0 from the first request, at the cost of a ~5 min load.
        args += ["-lm", "none", "--no-host"]
        notes.append("the model is read into RAM instead of memory-mapped; loading takes several minutes")
    return LlamaCommand(args=args, env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, cpu_expert_layers=cpu_experts, notes=notes,
                        rpc_servers=servers)


def _server_address(args: list[str]) -> tuple[str, int]:
    """Host and port llama-server will listen on, from its arguments."""
    host, port = "127.0.0.1", 8080
    for i, a in enumerate(args[:-1]):
        if a == "--host":
            host = args[i + 1]
        elif a == "--port":
            port = int(args[i + 1])
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return host, port


READY_TIMEOUT = 900.0  # seconds a model may take to load before llama-server is given up


class Stopped(Exception):
    """The caller's ``stop`` event was set while llama-server was loading."""


def wait_ready(host: str, port: int, proc: "subprocess.Popen | None" = None,
               timeout: float = READY_TIMEOUT, poll: float = 2.0, say=None, every: float = 60.0,
               stop=None) -> float:
    """Wait until llama-server answers /health with 200. Returns the seconds waited.

    Stops early with RuntimeError if ``proc`` exits while loading. With ``say``,
    tells every ``every`` seconds that it is still loading, since llama-server
    itself prints nothing for minutes while it reads a big model or sends its
    weights to other PCs. With ``stop`` (a ``threading.Event``), raises
    ``Stopped`` once it is set.
    """
    import time
    import urllib.error
    import urllib.request

    base = f"http://{host}:{port}"
    start = time.monotonic()
    told = start
    while True:
        if stop is not None and stop.is_set():
            raise Stopped()
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode} while loading")
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        now = time.monotonic()
        if now - start > timeout:
            raise TimeoutError(f"llama-server at {base} did not become ready in {timeout:.0f} s")
        if say is not None and now - told >= every:
            told = now
            say(f"still loading ({(now - start) / 60:.0f} min so far; Ctrl+C stops it)")
        if stop is not None:
            stop.wait(poll)
        else:
            time.sleep(poll)


def _stop(proc: "subprocess.Popen") -> None:
    """Terminate llama-server and make sure it is gone."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# Prompt lengths (tokens) for the warm-up. llama.cpp's CUDA matrix kernels are
# instantiated per batch-size class, and a build without kernels for the card
# (CUDA 12.4 on Turing) has the driver compile each class on first use: on the
# E5 rig the first 17-token prompt took 30 s and the next 3-token one 39 s.
WARMUP_PROMPTS = (1, 4, 12, 24, 48, 96, 192, 512)


def warm_up(host: str, port: int, sizes: tuple[int, ...] = WARMUP_PROMPTS, timeout: float = 900.0) -> float:
    """Send short completions of several prompt lengths so that the GPU kernels
    are compiled before the first real request. Returns the seconds it took."""
    import json
    import time
    import urllib.request

    start = time.monotonic()
    for n in sizes:
        body = json.dumps({"prompt": " ".join(["hello"] * n), "n_predict": 2, "cache_prompt": False}).encode()
        req = urllib.request.Request(f"http://{host}:{port}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    return time.monotonic() - start


def interruptible(fn, *args):
    """Call ``fn(*args)`` in a thread and wait for it in short steps.

    On Windows, Ctrl+C does not interrupt a blocking socket read, so a warm-up
    request stuck behind a kernel compile kept the launcher, and through it
    llama-server, running for over 30 s after Ctrl+C on the E5 rig. Waiting
    here instead lets the KeyboardInterrupt through at once.
    """
    import threading

    result: dict = {}

    def target() -> None:
        try:
            result["value"] = fn(*args)
        except BaseException as e:  # noqa: BLE001 - handed to the caller
            result["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(0.2)
    if "error" in result:
        raise result["error"]
    return result["value"]


class LineLog:
    """Writes whole lines to a stream from several threads.

    llama-server's log and the launcher's own messages share the terminal. The
    server's output goes through ``forward`` a line at a time, so a message
    such as "ready: ..." never lands in the middle of a server line (on the E5
    rig it did, and a script waiting for a line starting with "ready:" missed it).
    """

    def __init__(self, out=None):
        import sys
        import threading

        self.out = out if out is not None else sys.stderr
        self.lock = threading.Lock()

    def say(self, text: str) -> None:
        with self.lock:
            self.out.write(text + "\n")
            self.out.flush()

    def forward(self, stream) -> None:
        """Copy ``stream`` (bytes, e.g. a child's stdout) line by line until it ends."""
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            self.say(line)


def run(exe: str, command: LlamaCommand, model_path: str, extra: list[str] | None = None,
        warmup: bool = False, on_ready=None, ready_timeout: float = READY_TIMEOUT,
        log: "LineLog | None" = None, stop=None) -> int:
    """Start llama.cpp with the plan's arguments and wait for it.

    For llama-server, prints a "ready" line once the model has loaded (and
    warmed up), then calls ``on_ready(host, port)`` if given; it gives up
    after ``ready_timeout`` seconds of loading. The
    program is stopped whenever this function leaves, including on Ctrl+C
    during loading, and the OS ends it if Python itself is killed
    (see ``childproc``).

    ``log`` takes the server's output and this function's messages (stderr
    unless given). ``stop`` (a ``threading.Event``) lets another thread end
    it: once set, the server is stopped and 130 returned, as on Ctrl+C.
    """
    import os
    import sys

    from .llamabin import cuda_cache_env

    import subprocess
    import threading

    env = dict(os.environ, **command.env, **cuda_cache_env())
    argv = [exe, "-m", model_path, *command.args, *(extra or [])]
    server = "llama-server" in os.path.basename(exe).lower()
    log = log or LineLog()
    if server:
        proc, guard = spawn(argv, env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=log.forward, args=(proc.stdout,), daemon=True).start()
    else:
        proc, guard = spawn(argv, env)
    if not guard.active:
        log.say(f"note: if this launcher is killed, {os.path.basename(exe)} may keep running ({guard.how})")
    try:
        if not server:
            return _wait(proc, stop)
        host, port = _server_address(argv)
        try:
            seconds = wait_ready(host, port, proc, timeout=ready_timeout, say=log.say, stop=stop)
            timing = f"loaded in {seconds:.0f} s"
            if warmup:
                log.say(f"loaded in {seconds:.0f} s; warming up (the first start on a PC compiles GPU kernels, "
                        "which can take a few minutes once)")
                try:
                    warm = interruptible(warm_up, host, port)
                    timing += f", warmed up in {warm:.0f} s"
                except OSError as e:
                    log.say(f"warning: warm-up failed ({e}); the first requests may be slow")
            log.say(f"ready: http://{host}:{port}/v1 ({timing})")
            if on_ready is not None:
                try:
                    on_ready(host, port)
                except Exception as e:  # noqa: BLE001 - a convenience must not stop the server
                    log.say(f"note: {e}")
        except TimeoutError as e:
            log.say(f"warning: {e}; leaving it running")
        except RuntimeError as e:
            log.say(f"error: {e}")
            return proc.returncode or 1
        except Stopped:
            return 130
        return _wait(proc, stop)
    except KeyboardInterrupt:
        return 130
    finally:
        _stop(proc)


def _wait(proc: "subprocess.Popen", stop=None) -> int:
    """The program's exit code, or 130 once ``stop`` is set (the caller then stops the program)."""
    if stop is None:
        import subprocess

        # In short steps: on Windows a blocking wait keeps Ctrl+C and Ctrl+Break from being seen
        # until the program has ended by itself.
        while True:
            try:
                return proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                continue
    while proc.poll() is None:
        if stop.wait(0.2):
            return 130
    return proc.returncode
