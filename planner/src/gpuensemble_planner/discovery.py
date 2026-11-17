"""Find workers on the LAN, and let the worker's user approve a main PC once.

The main PC asks with one UDP packet, sent to the LAN's broadcast addresses
(or to given addresses), on the worker's port + 1 (50053 by default)::

    {"gpuensemble": "find", "v": 1, "name": "<main PC's name>"}

Each worker answers the address that asked. A main PC its gate lets in gets
the GPUs, as rig entries; any other gets only the worker's name, so it can
say where to approve it::

    {"gpuensemble": "worker", "v": 1, "name": "PC2", "port": 50052, "allowed": true,
     "gate": true, "gpus": [{"name": ..., "vram_gib": ..., "llama_device": "CUDA0", ...}]}

A main PC that is not let in gets ``"asking": true`` the first time it asks
while the worker's user can approve it there (the prompt is shown then), so
it can wait for the ``y``; later answers say false, so a PC its user ignores
does not keep waiting.

Approving happens on the worker: it shows "PC <name> at <address> asks to
use these GPUs" and lets its user type ``y``. The address goes into the
gate's list at once and into ``worker.json`` in the GPUEnsemble home, so the
next start of the worker remembers it.

Trust rests on the main PC's IP address, which suits a home LAN; on a
network shared with people you do not trust, use a cable of its own.

A worker at another site is reached through a VPN, which no broadcast
crosses. With Tailscale, the main PC also asks each PC of its tailnet
directly, and a worker lets in a PC signed in with its own account without
asking (see ``tailscale``).
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import hub
from .devices import is_loopback, rpc_overhead_from_rtt
from .gate import exclusive

MAGIC = "gpuensemble"
VERSION = 1
MAX_PACKET = 8192
# PCs asked through a VPN get this long to answer, and a second question on the way: the first
# packet to a PC the VPN has no path to yet waits for one (Tailscale starts through a relay).
VPN_TIMEOUT = 3.0
VPN_ASK_AGAIN = 0.7


def discovery_port(rpc_port: int) -> int:
    return rpc_port + 1


# --- the worker's side -------------------------------------------------------

def allowed_file() -> str:
    return os.path.join(hub.home_dir(), "worker.json")


def load_approved(path: str | None = None) -> list[str]:
    """Addresses approved on earlier runs of the worker."""
    try:
        with open(path or allowed_file(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    items = data.get("allow", []) if isinstance(data, dict) else []
    return [str(a) for a in items if isinstance(a, str)]


def save_approved(addresses: list[str], path: str | None = None) -> None:
    path = path or allowed_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"allow": sorted(set(addresses))}, f, indent=2)
    os.replace(tmp, path)


class Responder:
    """Answers the main PCs' questions on UDP, and asks this PC's user to approve new ones.

    ``is_allowed(address)`` says whether the gate lets that address in;
    ``on_request(address, name)`` is called once per unknown address (to show
    the approval prompt) and returns whether this PC's user can approve it now. ``gpus`` are the rig entries to hand out, without
    their ``rpc`` address, which the main PC takes from the answer; given as a
    function, it is called for each answer to a PC that is let in, so that the
    entries can say how much memory the cards have free then.
    """

    def __init__(self, listen: tuple[str, int], name: str, rpc_port: int,
                 gpus: list[dict] | Callable[[], list[dict]], gate: bool,
                 is_allowed: Callable[[str], bool], on_request: Callable[[str, str], None] | None = None,
                 log: Callable[[str], None] = lambda s: None) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            exclusive(self.sock)
            self.sock.bind(listen)
        except OSError:
            self.sock.close()
            raise
        # Wake up now and then to notice close(): on Linux closing the socket
        # does not end a recvfrom() that is waiting in another thread.
        self.sock.settimeout(0.5)
        self.closed = False
        self.address = self.sock.getsockname()
        self.name = name
        self.rpc_port = rpc_port
        self.gpus = gpus if callable(gpus) else (lambda: gpus)
        self.gate = gate
        self.is_allowed = is_allowed
        self.on_request = on_request
        self.log = log
        self.asked: set[str] = set()
        self._thread: threading.Thread | None = None

    def start(self) -> "Responder":
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def answer(self, data: bytes, peer: str) -> bytes | None:
        """The reply to one packet, or None to stay silent."""
        try:
            query = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(query, dict) or query.get(MAGIC) != "find" or query.get("v") != VERSION:
            return None
        allowed = self.is_allowed(peer)
        asking = False
        if not allowed and peer not in self.asked and self.on_request is not None:
            self.asked.add(peer)
            asker = str(query.get("name", ""))[:64] or "a PC"
            asking = bool(self.on_request(peer, asker))
            allowed = self.is_allowed(peer)  # on_request lets its user's own PC in at once
        reply = {MAGIC: "worker", "v": VERSION, "name": self.name, "port": self.rpc_port, "allowed": allowed}
        if allowed:
            reply.update(gate=self.gate, gpus=[{k: v for k, v in g.items() if k != "rpc"} for g in self.gpus()])
        else:
            reply["asking"] = asking
        return json.dumps(reply).encode("utf-8")

    def _serve(self) -> None:
        while not self.closed:
            try:
                data, peer = self.sock.recvfrom(MAX_PACKET)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue  # Windows: an earlier answer's port was closed (ICMP unreachable)
            except OSError:
                return  # closed
            reply = self.answer(data, peer[0])
            if reply is not None:
                try:
                    self.sock.sendto(reply, peer)
                except OSError as e:
                    self.log(f"discovery: could not answer {peer[0]} ({e})")

    def close(self) -> None:
        self.closed = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.sock.close()


class Approvals:
    """The approval prompt on the worker's console.

    A main PC that asks and is not let in yet is shown with its name and
    address; typing ``y`` (or ``allow <address>``) and Enter lets in the
    last one shown. ``admit(address)`` puts it into the gate's list and the
    remembered list.
    """

    def __init__(self, admit: Callable[[str], None], log: Callable[[str], None]) -> None:
        self.admit = admit
        self.log = log
        self.pending: list[str] = []
        self.lock = threading.Lock()

    def request(self, address: str, name: str) -> None:
        with self.lock:
            self.pending.append(address)
        self.log(f"PC {name!r} at {address} asks to use these GPUs. Type y and Enter here to let it in "
                 f"(remembered for next time), or ignore it to refuse. (Or restart with --allow {address}.)")

    def handle(self, line: str) -> bool:
        """Act on one typed line; True if it let a PC in."""
        words = line.strip().split()
        if not words:
            return False
        with self.lock:
            if words[0].lower() in ("y", "yes") and self.pending:
                address = self.pending.pop()
            elif words[0].lower() == "allow" and len(words) == 2:
                address = words[1]
                try:
                    ipaddress.ip_address(address)
                except ValueError:
                    self.log(f"{address!r} is not an IP address")
                    return False
                if address in self.pending:
                    self.pending.remove(address)
            else:
                return False
        self.admit(address)
        self.log(f"{address} may now use these GPUs (GPUEnsemble waiting there goes on by itself; "
                 "otherwise start it there again)")
        return True

    def read(self, stream) -> None:
        """Read typed lines until the stream ends (run in a thread)."""
        for line in iter(stream.readline, ""):
            self.handle(line)


# --- the main PC's side ------------------------------------------------------

@dataclass
class Found:
    address: str
    port: int
    name: str
    allowed: bool
    gate: bool = False
    gpus: list[dict] = field(default_factory=list)
    asking: bool = False  # its user was just asked to let this PC in
    rtt: float | None = None  # seconds for a question and its answer, when measured

    @property
    def endpoint(self) -> str:
        return f"{self.address}:{self.port}"

    def rig_entries(self) -> list[dict]:
        """The worker's GPUs as rig entries, reached at the address that answered.

        With a measured round trip, each entry's time per token includes it
        (``devices.rpc_overhead_from_rtt``), so a worker at another site
        through a VPN is not planned as if it were on the LAN.
        """
        entries = []
        for g in self.gpus:
            entry = dict(g, rpc=self.endpoint)
            # The worker names each card "... on <the address it took for its own>", which is its
            # LAN address even when it was found through a VPN: say where it is reached instead.
            card, on, _ = str(g.get("name", "")).rpartition(" on ")
            if on:
                entry["name"] = f"{card} on {self.address}"
            if self.gate:
                entry["gate"] = True
            if self.rtt is not None and not is_loopback(self.endpoint):
                entry["rpc_overhead_ms"] = round(rpc_overhead_from_rtt(self.rtt, self.gate) * 1000, 2)
                entry["rtt_ms"] = round(self.rtt * 1000, 2)
            entries.append(entry)
        return entries


def _local_ipv4() -> list[str]:
    found: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(a for a in found if not a.startswith("127."))


def is_this_pc(found: Found, local: list[str] | None = None, hostname: str | None = None) -> bool:
    """Whether a worker that answered runs on this PC itself.

    A broadcast reaches this PC's own worker too; using it would count this
    PC's GPUs twice, once directly and once through the RPC server.
    """
    if found.address.startswith("127.") or found.address in (local if local is not None else _local_ipv4()):
        return True
    return found.name.lower() == (hostname or socket.gethostname()).lower()


def broadcast_targets(local: list[str] | None = None) -> list[str]:
    """Where to send the question: every network, and each local /24 network
    (Windows sends 255.255.255.255 out of one network card only)."""
    targets = ["255.255.255.255"]
    for address in local if local is not None else _local_ipv4():
        ip = ipaddress.ip_address(address)
        if ip.version == 4 and ip.is_private:
            targets.append(str(ipaddress.ip_network(f"{address}/24", strict=False).broadcast_address))
    return list(dict.fromkeys(targets))


def parse_reply(data: bytes, peer: str) -> Found | None:
    try:
        reply = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(reply, dict) or reply.get(MAGIC) != "worker" or reply.get("v") != VERSION:
        return None
    try:
        port = int(reply["port"])
    except (KeyError, TypeError, ValueError):
        return None
    gpus = reply.get("gpus") or []
    if not isinstance(gpus, list) or not all(isinstance(g, dict) for g in gpus):
        return None
    return Found(peer, port, str(reply.get("name", peer))[:64], bool(reply.get("allowed")),
                 bool(reply.get("gate")), gpus, bool(reply.get("asking")))


def measure_rtt(worker: Found, tries: int = 5, timeout: float = 1.0, name: str | None = None) -> float | None:
    """The shortest of ``tries`` question-and-answer times to a worker, in seconds; None if it never answered.

    The shortest leaves out a first packet that waits for a VPN to find its
    path (Tailscale starts through a relay) and moments when either PC is busy.
    """
    query = json.dumps({MAGIC: "find", "v": VERSION, "name": name or socket.gethostname()}).encode("utf-8")
    best = None
    for _ in range(tries):
        # A socket per try, so that a late answer to an earlier try cannot pass for a quick one.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            start = time.perf_counter()
            try:
                s.sendto(query, (worker.address, discovery_port(worker.port)))
                deadline = start + timeout
                while (left := deadline - time.perf_counter()) > 0:
                    s.settimeout(left)
                    data, peer = s.recvfrom(MAX_PACKET)
                    if peer[0] == worker.address and parse_reply(data, peer[0]) is not None:
                        took = time.perf_counter() - start
                        best = took if best is None else min(best, took)
                        break
            except OSError:
                continue  # this try got no answer (a timeout is an OSError too)
    return best


def find_workers(at: list[str] | None = None, port: int = 50053, timeout: float = 1.5,
                 name: str | None = None, also: list[str] | None = None) -> list[Found]:
    """Workers that answer within ``timeout`` seconds, asked at ``at`` or by broadcast.

    ``also`` are addresses asked one by one on top of the broadcast: the PCs
    of a VPN, which no broadcast reaches. They get ``VPN_TIMEOUT`` seconds
    and a second question.
    """
    query = json.dumps({MAGIC: "find", "v": VERSION, "name": name or socket.gethostname()}).encode("utf-8")
    found: dict[tuple[str, int], Found] = {}
    targets = list(at or broadcast_targets())
    extra = [] if at else [a for a in dict.fromkeys(also or []) if a not in targets]
    if extra:
        timeout = max(timeout, VPN_TIMEOUT)

    def ask(sock: socket.socket, addresses: list[str]) -> None:
        for target in addresses:
            try:
                sock.sendto(query, (target, port))
            except OSError:
                pass  # e.g. a network that is down

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        ask(s, targets + extra)
        start = time.monotonic()
        deadline = start + timeout
        again = start + VPN_ASK_AGAIN if extra else None
        while (now := time.monotonic()) < deadline:
            if again is not None and now >= again:
                ask(s, [a for a in extra if not any(w.address == a for w in found.values())])
                again = None
            s.settimeout(max(0.001, min(deadline, again if again is not None else deadline) - now))
            try:
                data, peer = s.recvfrom(MAX_PACKET)
            except ConnectionResetError:
                continue  # Windows: an address asked has nothing listening (ICMP unreachable)
            except socket.timeout:
                continue  # time to ask again, or the deadline
            except OSError:
                break
            worker = parse_reply(data, peer[0])
            if worker is not None:
                found[(worker.address, worker.port)] = worker
                if at and len(found) >= len(at):
                    break  # every address asked has answered
    return sorted(found.values(), key=lambda w: (w.name, w.address))


def one_address_each(workers: list[Found], far: Callable[[str], bool]) -> list[Found]:
    """Each worker once, when it answered at two addresses (on the LAN and through a VPN).

    The address that lets this PC in is kept; of two that do, the one that is
    not ``far`` (through the VPN), as the LAN is the shorter way.
    """
    best: dict[tuple[str, int], Found] = {}
    for w in workers:
        key = (w.name.lower(), w.port)
        if key not in best or (not w.allowed, far(w.address)) < (not best[key].allowed, far(best[key].address)):
            best[key] = w
    return sorted(best.values(), key=lambda w: (w.name, w.address))
