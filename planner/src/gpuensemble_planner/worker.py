"""Lend this PC's GPUs to GPUEnsemble on another PC, through llama.cpp's RPC server.

On the PC with the extra GPUs::

    gpuensemble worker --host 192.168.10.2

starts ``ggml-rpc-server`` from the llama.cpp that ``setup`` installs, with
every NVIDIA card of this PC (or the ones given with ``--device``), and prints
the cards as rig-file entries. Save those as a file on the main PC and pass it
there with ``--remote FILE`` (or add them to its rig file's ``gpus``): the
planner then places layers on them too, and ``up`` connects to this PC.

The RPC server has no password. Anyone who can reach its address can use the
GPUs and read or change their memory, so it listens on 127.0.0.1 unless told
otherwise. On any other address ``--allow`` names the PCs that may connect:
the server then stays on 127.0.0.1 and a gate (``gate.py``) takes its place on
the network and lets in only those addresses.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from typing import Callable

import functools

from . import firewall, fnv, hub, serving, tailscale
from .childproc import spawn
from .devices import GIB, Device, check_endpoint, is_loopback
from .discovery import Approvals, Responder, discovery_port, load_approved, save_approved
from .fetch import FetchServer
from .gate import Gate, parse_allowed
from .gate import is_allowed as gate_allows
from .llamabin import Install, cuda_cache_env

DEFAULT_PORT = 50052
# llama.cpp renamed rpc-server to ggml-rpc-server; b12375 ships the new name.
SERVER_NAMES = ("ggml-rpc-server", "rpc-server")


def find_rpc_server(inst: Install) -> str | None:
    """Path of the RPC server in a llama.cpp install, or None if the build has none."""
    for name in SERVER_NAMES:
        path = inst.tool(name)
        if os.path.exists(path):
            return path
    return None


def unreachable_servers(servers: list[str], timeout: float = 3.0,
                        connect: Callable = socket.create_connection) -> list[str]:
    """The RPC servers that do not accept a connection.

    llama-server reports a server it cannot reach as "invalid device: RPC0"
    followed by its usage text, which reads like a wrong command; checking
    first gives a clear message instead.
    """
    missing = []
    for server in servers:
        host, _, port = server.rpartition(":")
        try:
            connect((host.strip("[]"), int(port)), timeout=timeout).close()
        except (OSError, ValueError):
            missing.append(server)
    return missing


def guess_address() -> str | None:
    """This PC's address on the network its default route uses (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1: never answered, only routed
            return s.getsockname()[0]
    except OSError:
        return None


def pick_devices(detected: list[Device], wanted: list[str] | None) -> list[Device]:
    """The cards to serve, in the order the server will offer them."""
    if not wanted:
        return list(detected)
    by_name = {d.llama_device: d for d in detected}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        have = ", ".join(str(d.llama_device) for d in detected) or "none"
        raise ValueError(f"no GPU {', '.join(missing)} on this PC (it has {have})")
    return [by_name[w] for w in wanted]


def rig_entries(devices: list[Device], endpoint: str, gate: bool = False, cache: bool = False,
                fetch: bool = False, serve: bool = False) -> list[dict]:
    """The cards as the main PC's rig file lists them, in the server's ``-d`` order.

    ``gate`` tells the planner that the connection passes the gate, which adds
    to the time per token (``devices.RPC_GATE_OVERHEAD``); ``cache`` that the
    server keeps the weights it receives, so only the first load sends them;
    ``fetch`` that this PC downloads its share of them itself when asked;
    ``serve`` that it runs a whole model itself when asked (``serving``).
    """
    where = endpoint.rsplit(":", 1)[0]
    entries = []
    for d in devices:
        entry = {
            "name": f"{d.name} on {where}",
            "vram_gib": round(d.vram_bytes / GIB, 2),
            "reserve_gib": round(d.reserve_bytes / GIB, 2),
            "bandwidth_gbs": round(d.bandwidth / 1e9),
            "rpc": endpoint,
            "llama_device": d.llama_device,
        }
        if gate:
            entry["gate"] = True
        if cache:
            entry["cache"] = True
        if fetch:
            entry["fetch"] = True
        if serve:
            entry["serve"] = True
        entries.append(entry)
    return entries


# What this PC's own RPC server holds while it waits for a main PC (its CUDA context): about 1 GiB
# on an RTX 4090 under Windows. nvidia-smi does not tell there which program holds how much, so
# free memory up to this much below what it was at the start is taken for the server's own.
OWN_SERVER_ALLOWANCE = int(1.5 * GIB)
# A main PC asks several times within a second (to find the worker, then to time the round trip):
# one reading of the cards serves them all.
MEMORY_TTL = 2.0
# How long an answer waits for nvidia-smi before it goes out with the last reading.
MEMORY_WAIT = 0.8


def current_devices(at_start: list[Device], now: list[Device]) -> list[Device]:
    """The cards with the room they have now, for a main PC that asks after the worker started.

    Another program on this PC may have taken a card's memory since then (or
    given it back): a plan made with the room of hours ago would not fit, and
    on Windows the driver then spills into shared memory, which is far slower.
    A card keeps its room of the start while its free memory is within
    ``OWN_SERVER_ALLOWANCE`` below it; with more room than at the start, or
    clearly less, it gets what is free now.
    """
    by_name = {d.llama_device: d for d in now}
    out = []
    for d in at_start:
        n = by_name.get(d.llama_device)
        if n is None or 0 <= d.vram_bytes - n.vram_bytes <= OWN_SERVER_ALLOWANCE:
            out.append(d)
        else:
            out.append(n)
    return out


class LiveEntries:
    """The rig entries a main PC is given, with each card's memory read when it asks.

    ``detect`` runs in a thread of its own, at most once per ``ttl`` seconds; an
    answer waits ``wait`` seconds for it and else goes out with the last reading,
    so a slow nvidia-smi cannot keep a main PC from finding the worker.
    """

    def __init__(self, at_start: list[Device], detect: Callable[[], list[Device]],
                 entries: Callable[[list[Device]], list[dict]], log: Callable[[str], None] = lambda s: None,
                 ttl: float | None = None, wait: float | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.at_start = at_start
        self.detect = detect
        self.entries = entries
        self.log = log
        self.ttl = MEMORY_TTL if ttl is None else ttl
        self.wait = MEMORY_WAIT if wait is None else wait
        self.clock = clock
        self.devices = list(at_start)
        self.read_at = clock()  # the start's reading is fresh
        self.lock = threading.Lock()
        self.reading: threading.Thread | None = None

    def _read(self) -> None:
        try:
            devices = current_devices(self.at_start, self.detect())
        except Exception:  # noqa: BLE001 - nvidia-smi failed or a card is gone: keep the last reading
            devices = None
        with self.lock:
            if devices is not None:
                for old, new in zip(self.devices, devices):
                    if new.vram_bytes != old.vram_bytes:
                        if new.vram_bytes < old.vram_bytes:
                            why = "another program uses its memory, or a model is loaded on it"
                        else:
                            why = "its memory is free again"
                        self.log(f"note: main PCs are now told {new.vram_bytes / GIB:.1f} GiB for {new.name} "
                                 f"({old.vram_bytes / GIB:.1f} before): {why}")
                self.devices = devices
            self.read_at = self.clock()
            self.reading = None

    def __call__(self) -> list[dict]:
        with self.lock:
            thread = self.reading
            if thread is None and self.clock() - self.read_at >= self.ttl:
                thread = self.reading = threading.Thread(target=self._read, daemon=True)
                thread.start()
        if thread is not None:
            thread.join(self.wait)
        with self.lock:
            return self.entries(self.devices)


def server_command(exe: str, host: str, port: int, devices: list[Device], cache: bool) -> list[str]:
    argv = [exe, "-H", host, "-p", str(port), "-d", ",".join(str(d.llama_device) for d in devices)]
    if cache:
        # Keeps the uploaded weights on this PC, so the next load of the same
        # model sends only their hashes.
        argv.append("-c")
    return argv


def rpc_cache_dir() -> str:
    """Where the RPC server keeps the weights it receives: llama.cpp adds ``rpc`` to its cache folder."""
    return os.path.join(os.environ.get("LLAMA_CACHE") or os.path.join(hub.home_dir(), "rpc-cache"), "rpc")


def server_env(cache: bool) -> dict[str, str]:
    """Environment for the RPC server.

    With ``-c`` llama.cpp keeps the weights under its cache folder (LLAMA_CACHE,
    else one in the user's profile); point it into the GPUEnsemble home, so
    that everything stays in one folder that is easy to find and delete.
    """
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", **cuda_cache_env())
    if cache and not env.get("LLAMA_CACHE"):
        env["LLAMA_CACHE"] = os.path.join(hub.home_dir(), "rpc-cache")
    return env


def serve(argv: list[str], log: Callable[[str], None]) -> int:
    """Run the RPC server until it exits or Ctrl+C; it never outlives this process."""
    env = server_env("-c" in argv[1:])
    proc, guard = spawn(argv, env)
    if not guard.active:
        log(f"note: if this program is killed, {os.path.basename(argv[0])} may keep running ({guard.how})")
    try:
        # Wait in short steps: on Windows Ctrl+C does not interrupt an endless wait.
        while proc.poll() is None:
            time.sleep(0.2)
        return proc.returncode
    except KeyboardInterrupt:
        return 130
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:  # noqa: BLE001 - make sure it is gone
                proc.kill()
                proc.wait()


def free_port() -> int:
    """A TCP port on 127.0.0.1 that nothing listens on right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(host: str, port: int, wanted: list[str] | None, cache: bool,
          detect: Callable[[], list[Device]], install: Callable[[], Install],
          exe: str | None = None, log: Callable[[str], None] | None = None,
          out=None, run: Callable[[list[str], Callable[[str], None]], int] = serve,
          allow: list[str] | None = None, allow_any: bool = False, lan: bool = False,
          stdin=None, approved_path: str | None = None,
          reachability: Callable[..., list[str]] = firewall.reachability_notes,
          tailnet: Callable[[], dict | None] | None = None, fetch: bool | None = None,
          serve: bool = False, launch: Callable[..., int] = serving.launch) -> int:
    """Everything ``gpuensemble worker`` does; the callables are for tests.

    ``lan`` listens on every network behind the gate, answers main PCs that
    look for workers (``discovery``), and lets this PC's user approve them;
    approved addresses are remembered in ``approved_path`` (worker.json in
    the GPUEnsemble home). On a network address, ``reachability(host=host)``
    is read in the background and its notes (Windows' network profile and
    firewall rules for the network that has ``host``) are shown.

    With ``lan`` and ``tailnet`` (``tailscale.read_status``), a main PC that
    asks through Tailscale and is signed in with this PC's account is let in
    without asking, for this run: nobody may be at a worker at another site.

    With ``serve``, a main PC that is let in may also ask this PC to run a
    whole model on the lent cards itself (``serving``); ``launch`` does that.
    """
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    out = out if out is not None else sys.stdout
    nets = None
    if allow or lan:
        nets = parse_allowed(allow or [], allow_empty=lan)
        if lan:
            remembered = load_approved(approved_path)
            nets += [n for n in parse_allowed(remembered, allow_empty=True) if n not in nets]
    if not is_loopback(host) and nets is None and not allow_any:
        raise ValueError(f"listening on {host} needs --allow with the main PC's address (e.g. --allow 192.168.10.1), "
                         "or --lan to approve main PCs as they ask: the RPC server has no password, so the worker "
                         "lets in only the PCs named there")
    if host in ("0.0.0.0", "::"):
        shown = guess_address() or "THIS_PC_ADDRESS"
        if not lan:
            log(f"listening on every network of this PC; the main PC reaches it at {shown} "
                "if that is the right network (check, and prefer --host with that address)")
    else:
        shown = host
    endpoint = check_endpoint(f"{shown}:{port}")
    devices = pick_devices(detect(), wanted)
    if not devices:
        raise ValueError("no NVIDIA GPU found on this PC")
    if exe is None:
        exe = find_rpc_server(install())
        if exe is None:
            raise OSError("this llama.cpp has no RPC server (ggml-rpc-server); run setup with a newer --tag")

    # Its share of the weights goes into the cache, and is worth downloading only where hashing it is fast.
    fetch = cache and (fnv.fast() if fetch is None else fetch)

    def entries_of(cards: list[Device]) -> list[dict]:
        # One description of this worker, for the entries printed at the start and for every main PC that asks.
        return rig_entries(cards, endpoint, gate=nets is not None, cache=cache, fetch=fetch, serve=serve)

    entries = entries_of(devices)
    if lan:
        log(f"main PCs on the LAN find these GPUs with `{os.environ.get('GPUENSEMBLE_PROG') or 'gpuensemble'} "
            "up --remote auto`; the first time, approve the main PC here")
        if tailnet is not None and tailnet() is not None:
            log("through Tailscale, PCs signed in with this PC's account find them too and are let in "
                "without asking (--ask-tailscale asks about them as well)")
    else:
        log("give the main PC these GPUs (save them as a file and pass it with --remote FILE):")
        out.write(json.dumps({"gpus": entries}, indent=2) + "\n")
        out.flush()

    def allowed(peer: str) -> bool:
        if nets is not None:
            return gate_allows(peer, nets)
        return allow_any or is_loopback(host)

    def admit(address: str) -> None:
        net = parse_allowed([address])[0]
        if net not in nets:
            nets.append(net)
        save_approved(load_approved(approved_path) + [address], approved_path)

    approvals = None
    if nets is not None:
        # Read what is typed in this window (a pipe works too, for scripts).
        if stdin is not None or sys.stdin is not None:
            approvals = Approvals(admit, log)
            threading.Thread(target=approvals.read, args=(stdin or sys.stdin,), daemon=True).start()

    def on_request(address: str, name: str) -> bool:
        if lan and tailnet is not None and tailscale.is_tailnet_address(address):
            mine = tailscale.own_pc(address, tailnet())
            if mine is not None:
                # Not remembered: Tailscale is asked again on the next start.
                net = parse_allowed([address])[0]
                if net not in nets:
                    nets.append(net)
                log(f"PC {name!r} at {address} is signed in to Tailscale with this PC's account "
                    f"({mine.name}); it may use these GPUs")
                return False
        if approvals is not None:
            approvals.request(address, name)
            return True
        log(f"PC {name!r} at {address} asks to use these GPUs; restart the worker with --allow {address} "
            "(or run it in a console window to approve it there)")
        return False

    closers = []
    try:
        if nets is None:
            if not is_loopback(host):
                log(f"warning: the RPC server has no password; anyone who can reach {endpoint} can use these GPUs "
                    "and read or change their memory. Use it only on a network you trust, and allow only the "
                    "main PC through the firewall (Windows may ask about ggml-rpc-server: allow private networks only).")
            argv = server_command(exe, host, port, devices, cache)
        else:
            inner = free_port()
            argv = server_command(exe, "127.0.0.1", inner, devices, cache)
            closers.append(Gate((host, port), ("127.0.0.1", inner), nets, log).start())
            let_in = ", ".join(str(n) for n in nets) or "no one yet"
            log(f"gate on {host}:{port} lets in {let_in}; the RPC server listens on 127.0.0.1:{inner}")
        try:
            # Asked later, the cards may have less room than now: read them again for each main PC.
            live = LiveEntries(devices, lambda: pick_devices(detect(), wanted), entries_of, log)
            closers.append(Responder((host, discovery_port(port)), socket.gethostname(), port, live,
                                     nets is not None, allowed, on_request, log).start())
        except OSError as e:
            log(f"note: main PCs cannot find this worker by themselves ({e}); give them --remote instead")
        others = {}
        if serve:
            # A model that fits on these cards alone runs here from end to end when the main PC asks.
            running = serving.Serving((host, serving.api_port(port)),
                                      functools.partial(launch, devices=lambda: pick_devices(detect(), wanted)), log)
            closers.append(running)
            others["serve"] = running.handle
        if fetch or serve:
            # The main PC asks here, on the discovery port's number, for this PC to download its share itself
            # or to run the model.
            try:
                closers.append(FetchServer((host, discovery_port(port)), rpc_cache_dir(), allowed, log,
                                           others=others, fetches=bool(fetch)).start())
            except OSError as e:
                log(f"note: this PC cannot be asked to download weights or to run a model itself ({e}); "
                    "the main PC sends the weights and runs the model")
        if not is_loopback(host):
            log("(Windows may ask whether Python may accept connections: allow private networks only)")

            def check() -> None:
                for note in reachability(host=host):
                    log(f"note: {note}")
            # Reading the firewall rules takes a few seconds; the server need not wait for it.
            threading.Thread(target=check, daemon=True).start()
        log("starting: " + " ".join(argv))
        log("Ctrl+C stops it")
        return run(argv, log)
    finally:
        for c in reversed(closers):
            c.close()
