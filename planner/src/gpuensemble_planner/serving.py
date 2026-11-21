"""Run a whole model on the PC whose cards hold all of it, and use it from the main PC.

A model that fits on another PC's cards alone is planned onto them, and the
main PC then adds nothing but the network: with llama.cpp's RPC every token
waits for a round trip and for the logits to come back. Measured between two
sites (benchmarks/phase2c-three-cards.md), Qwen3-32B on an RTX 4090 ran at
25.2-25.95 tokens/s started on the main PC and at 41.4-41.5 started on the
4090's PC.

So ``up`` on the main PC asks such a PC to start the model itself. A worker
started with ``--serve`` takes the request on the port that also takes the
requests to download weights (the worker's port + 1, TCP), from an address
it lets in::

    {"gpuensemble": "serve", "v": 1, "model": "qwen3-32b", "ctx": 8192}

It downloads the model if it does not have it, plans it on its own cards,
starts llama-server on 127.0.0.1, and puts a gate in front of it (the
worker's port + 2) that lets in only the PC that asked. It answers in lines
of JSON: ``{"log": "..."}`` for each line of its progress and of
llama-server's output, ``{"ready": true, "port": N}`` once the model answers,
and ``{"stopped": code}`` or ``{"error": "..."}`` at the end. The connection
stays open while the model runs: when the main PC closes it (``up`` stopped)
or goes away, the worker stops llama-server and frees the cards.

The main PC listens where the user expects the API (127.0.0.1:8080) and
passes every connection on to that gate, so a chat program sees no
difference. The text of the answer crosses the network once, as it is
written; nothing waits for the network per token.

What a main PC can ask for is narrow on purpose: a model by its id in the
worker's own catalog, which the worker downloads from the place its catalog
names, and a context size. It cannot name a file, an address to download
from, or arguments for llama-server.
"""

from __future__ import annotations

import json
import re
import select
import socket
import threading
import time
from typing import Callable

from . import hub
from .devices import Device, detect_host
from .gate import Gate, keep_alive, parse_allowed, reached_itself
from .gguf import read_model
from .llamabin import installed, setup
from .llamacpp import LineLog, llama_command, run
from .model import profile_from_gguf
from .plan import make_plan
from .recommend import load_catalog

MAGIC = "gpuensemble"
VERSION = 1
API_PORT_OFFSET = 2
MIN_CTX, MAX_CTX = 256, 1 << 20
STALL = 300.0  # seconds without a word from the other PC before the main PC stops waiting for it to start
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def api_port(rpc_port: int) -> int:
    """Where a worker's gate for llama-server's API listens."""
    return rpc_port + API_PORT_OFFSET


class Stopped(Exception):
    """The PC that asked went away or said stop."""


def parse_request(request: dict) -> tuple[str, int]:
    """The model id and context a request asks for; ValueError says what is wrong with it."""
    if request.get(MAGIC) != "serve" or request.get("v") != VERSION:
        raise ValueError("not a serve request of this version")
    model, ctx = request.get("model"), request.get("ctx")
    if not isinstance(model, str) or not _ID.match(model):
        raise ValueError("the request needs a model id from the catalog")
    if not isinstance(ctx, int) or isinstance(ctx, bool) or not MIN_CTX <= ctx <= MAX_CTX:
        raise ValueError(f"the request needs a context size from {MIN_CTX} to {MAX_CTX}")
    return model, ctx


# --- the worker's side -------------------------------------------------------

def launch(model_id: str, ctx: int, port: int, log: LineLog, stop: threading.Event,
           on_ready: Callable[[str, int], None], devices: Callable[[], list[Device]],
           catalog: str | None = None, base: str | None = None,
           host_of: Callable = detect_host, install: Callable = lambda log: installed() or setup(log=log),
           find: Callable = hub.find_local, pull: Callable = hub.pull,
           load: Callable = lambda path: profile_from_gguf(read_model(path)), start: Callable = run) -> int:
    """Download, plan and run ``model_id`` on this PC's ``devices()`` with llama-server on 127.0.0.1:``port``.

    Returns llama-server's exit code (130 once ``stop`` is set). Raises
    LookupError for a model this PC's catalog does not have and RuntimeError
    when it does not fit on the cards alone; the main PC then starts the
    model itself, with these cards over RPC as before.
    """
    entry = next((e for e in load_catalog(catalog) if e.get("id") == model_id), None)
    if entry is None:
        raise LookupError(f"{model_id} is not in this PC's catalog (its GPUEnsemble may be an older version)")

    def say(text: str) -> None:
        if stop.is_set():
            raise Stopped()  # ends a download; what has arrived stays for the next time
        log.say(text)

    path = find(entry, base)
    if path is None:
        path = pull(entry, base=base, say=say)
    inst = install(log.say)
    cards = devices()
    host = host_of(None)
    host.expert_path = "cpu"  # the plan is carried out by llama.cpp
    model = load(path)
    plan = make_plan(model, cards, host, context=ctx)
    if plan.cpu_layers or plan.expert_ram_bytes:
        raise RuntimeError(f"{entry['name']} does not fit on this PC's GPUs alone right now "
                           f"({plan.cpu_layers} layers would run on its CPU)")
    command = llama_command(plan, model, cards, tool="server", threads=host.threads, ram_bytes=host.ram_bytes)
    used = [s for s in plan.stages if s.n_layers]
    say(f"{entry['name']} on {len(used)} GPU(s) of this PC, context {ctx}: "
        f"estimated {plan.tokens_per_second:.1f} tokens/s with half the context filled")
    # --metrics: the main PC's monitor reads the speed of the answers through the gate, as it does here.
    extra = ["--host", "127.0.0.1", "--port", str(port), "-a", model_id, "--metrics"]
    say("starting: " + command.shell(inst.server, model=path, extra=extra))
    return start(inst.server, command, path, extra, warmup=True, on_ready=on_ready, log=log, stop=stop)


class _Forward(LineLog):
    """A log whose lines also go to the PC that asked."""

    def __init__(self, tell: Callable[[dict], None], out=None) -> None:
        super().__init__(out)
        self.tell = tell

    def say(self, text: str) -> None:
        super().say(text)
        self.tell({"log": text})


class Serving:
    """A worker's answer to main PCs that ask it to run a whole model (``worker --serve``).

    ``listen`` is where the gate for llama-server's API goes. ``launch`` runs
    the model: ``launch(model_id, ctx, port, log, stop, on_ready)``, see the
    function of that name. One model runs at a time.
    """

    def __init__(self, listen: tuple[str, int], launch: Callable[..., int],
                 log: Callable[[str], None] = lambda s: None, out=None, free_port: Callable[[], int] | None = None) -> None:
        self.listen = listen
        self.launch = launch
        self.log = log
        self.out = out
        self.free_port = free_port or _free_port
        self.busy = threading.Lock()
        self.running = ""  # "MODEL for the PC at ADDRESS", while one runs
        self.stop = threading.Event()  # the worker is closing

    def handle(self, request: dict, client: socket.socket, peer: str, say: Callable[[dict], None]) -> None:
        """One request, from its first line until the model has stopped."""
        try:
            model_id, ctx = parse_request(request)
        except ValueError as e:
            say({"error": str(e)})
            return
        if not self.busy.acquire(blocking=False):
            say({"error": f"this PC is already running {self.running}"})
            return
        try:
            self.running = f"{model_id} for the PC at {peer}"
            self._run_for(model_id, ctx, client, peer, say)
        finally:
            self.running = ""
            self.busy.release()

    def _run_for(self, model_id: str, ctx: int, client: socket.socket, peer: str,
                 say: Callable[[dict], None]) -> None:
        stop = threading.Event()
        telling = threading.Lock()  # llama-server's output and this thread both write to the main PC

        def tell(message: dict) -> None:
            with telling:
                try:
                    say(message)
                except OSError:
                    stop.set()  # the main PC went away

        def watch() -> None:
            # The main PC says nothing more. Its side closing, or the system finding it gone, ends the run.
            # Asked in short steps, never waiting inside recv(): on Windows a socket that is closed while
            # another thread waits in recv() is reset, and the main PC loses the last lines sent to it.
            try:
                while not stop.is_set():
                    readable, _, _ = select.select([client], [], [], 0.2)
                    if readable and not client.recv(4096):
                        break
            except (OSError, ValueError):
                pass
            stop.set()

        def closing() -> None:
            while not stop.wait(0.5):
                if self.stop.is_set():
                    stop.set()

        try:
            keep_alive(client)
        except OSError:
            pass  # a main PC that vanishes is then noticed only when something is written to it
        inner = self.free_port()
        try:
            gate = Gate(self.listen, ("127.0.0.1", inner), parse_allowed([peer]), self.log,
                        what="llama-server").start()
        except OSError as e:
            tell({"error": f"this PC cannot listen on port {self.listen[1]} for the model's API ({e})"})
            return
        self.log(f"the PC at {peer} asks this PC to run {model_id} itself (context {ctx}); "
                 f"its API will be on port {self.listen[1]}, for that PC only")
        watchers = [threading.Thread(target=target, daemon=True) for target in (watch, closing)]
        for thread in watchers:
            thread.start()
        started = time.monotonic()
        try:
            code = self.launch(model_id, ctx, inner, _Forward(tell, self.out), stop,
                               lambda host, port: tell({"ready": True, "port": self.listen[1], "model": model_id}))
            last = {"stopped": code}
        except Stopped:
            last = {"stopped": 130}
        except Exception as e:  # noqa: BLE001 - told to the main PC, which then starts the model itself
            self.log(f"could not run {model_id} for the PC at {peer} ({e})")
            last = {"error": str(e)}
        finally:
            gate.close()
            stop.set()
            for thread in watchers:
                thread.join(timeout=5)  # nothing reads the connection any more when it is closed
        if "stopped" in last:
            self.log(f"stopped {model_id} for the PC at {peer} after {(time.monotonic() - started) / 60:.0f} min "
                     f"(exit code {last['stopped']}); the GPUs are free again")
        tell(last)

    def close(self) -> None:
        """Stop the model that runs, if one does, and wait a moment for llama-server to end."""
        self.stop.set()
        if self.busy.acquire(timeout=20):
            self.busy.release()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- the main PC's side ------------------------------------------------------

class Session:
    """A model that another PC runs for this one; closing it stops the model there."""

    def __init__(self, sock: socket.socket, address: str, port: int, buffer: bytes = b"") -> None:
        self.sock = sock
        self.address = address
        self.port = port  # of that PC's gate for the API
        self._buffer = buffer

    def messages(self, timeout: float = 1.0):
        """The other PC's next messages, waiting up to ``timeout`` s for one; None once it has closed."""
        while b"\n" not in self._buffer:
            self.sock.settimeout(timeout)
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                return []
            if not data:
                return None
            self._buffer += data
        *lines, self._buffer = self._buffer.split(b"\n")
        out = []
        for line in lines:
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise OSError("it answered something else")
            if not isinstance(message, dict):
                raise OSError("it answered something else")
            out.append(message)
        return out

    def wait(self, on_log: Callable[[str], None]) -> int:
        """Pass on the other PC's log until the model stops there; its exit code (1 if the PC went away)."""
        while True:
            try:
                messages = self.messages()
            except OSError as e:
                on_log(f"the connection broke ({e})")
                return 1
            if messages is None:
                on_log("it closed the connection")
                return 1
            for message in messages:
                if "log" in message:
                    on_log(str(message["log"]))
                elif message.get("error"):
                    on_log(f"error: {message['error']}")
                    return 1
                elif "stopped" in message:
                    code = message["stopped"]
                    return code if isinstance(code, int) and not isinstance(code, bool) else 1

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def ask(address: str, port: int, model_id: str, ctx: int, on_log: Callable[[str], None],
        stall: float = STALL, connect: Callable = socket.create_connection) -> Session:
    """Ask the worker at ``address`` (its request port) to run the model, and wait until it answers requests.

    OSError says why it will not or cannot; the caller then starts the model
    itself. A worker that says nothing for ``stall`` seconds is given up.
    Ctrl+C while waiting closes the connection, which stops the worker's work.
    """
    request = {MAGIC: "serve", "v": VERSION, "model": model_id, "ctx": ctx}
    sock = connect((address, port), timeout=10)
    session = Session(sock, address, 0)
    try:
        if reached_itself(sock):
            raise OSError("nothing listens there")
        try:
            keep_alive(sock)
        except OSError:
            pass
        sock.sendall(json.dumps(request).encode("utf-8") + b"\n")
        heard = time.monotonic()
        while True:
            messages = session.messages()
            if messages is None:
                raise OSError("it closed the connection")
            if messages:
                heard = time.monotonic()
            elif time.monotonic() - heard > stall:
                raise OSError(f"it said nothing for {stall:.0f} s")
            for message in messages:
                if "log" in message:
                    on_log(str(message["log"]))
                elif message.get("error"):
                    raise OSError(str(message["error"]))
                elif "stopped" in message:
                    raise OSError(f"llama-server stopped there (exit code {message['stopped']})")
                elif message.get("ready"):
                    port_there = message.get("port")
                    if not isinstance(port_there, int) or isinstance(port_there, bool) or not 0 < port_there < 65536:
                        raise OSError("it answered something else")
                    session.port = port_there
                    return session
    except BaseException:
        session.close()
        raise


class Relay(Gate):
    """Listens where the user expects the API and passes every connection on to the other PC's gate."""

    def __init__(self, listen: tuple[str, int], target: tuple[str, int], log: Callable[[str], None]) -> None:
        # Who may reach it is decided by where it listens (127.0.0.1 unless the user chose otherwise),
        # as for llama-server itself.
        super().__init__(listen, target, parse_allowed(["0.0.0.0/0", "::/0"]), log, what="the model on the other PC")

