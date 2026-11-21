"""A gate in front of llama.cpp's RPC server that lets in only chosen addresses.

``ggml-rpc-server`` accepts anyone who can reach it and has no password. With
``worker --allow``, the server listens on 127.0.0.1 only, and this gate takes
its place on the network: it accepts a connection only from an allowed
address and then copies bytes both ways, unchanged. llama.cpp on the main PC
does not know the gate is there.

Every RPC call is a small request and answer, so both sockets have Nagle's
algorithm turned off (as llama.cpp does on its own); otherwise each answer
would wait for the delayed ACK.

A main PC that loses power or its network says no goodbye. The gate would wait
for it for ever, and the RPC server would keep the model that PC loaded in GPU
memory. So the gate has the system probe a connection that has gone quiet
(TCP keepalive) and closes both sides when the probes get no answer.
"""

from __future__ import annotations

import errno
import ipaddress
import socket
import sys
import threading
from typing import Callable, Iterable

BUFFER = 1 << 20

# A connection that has been quiet for KEEPALIVE_IDLE seconds is probed every
# KEEPALIVE_INTERVAL seconds; after KEEPALIVE_COUNT probes without an answer the
# main PC counts as gone (two minutes in all). A main PC that is only idle
# answers the probes, so it stays connected however long it waits.
KEEPALIVE_IDLE = 60
KEEPALIVE_INTERVAL = 6
KEEPALIVE_COUNT = 10  # Windows always sends 10


def parse_allowed(items: Iterable[str], allow_empty: bool = False) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Addresses or networks (``192.168.10.5``, ``192.168.10.0/24``) a gate lets in."""
    nets = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        try:
            nets.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            raise ValueError(f"{item!r} is not an IP address or network (e.g. 192.168.10.5 or 192.168.10.0/24)")
    if not nets and not allow_empty:
        raise ValueError("--allow needs at least one address")
    return nets


def is_allowed(peer: str, nets: list) -> bool:
    try:
        ip = ipaddress.ip_address(peer.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip.version == n.version and ip in n for n in nets)


def exclusive(sock: socket.socket) -> None:
    """Keep the port to ourselves. On Windows SO_REUSEADDR would let another
    program bind the same port and take the connections, so ask for exclusive
    use there; elsewhere SO_REUSEADDR only lets a restart reuse the port at once."""
    if sys.platform == "win32":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def reached_itself(sock: socket.socket) -> bool:
    """True when both ends of a connection are this one socket.

    A connection to a port of this PC that nothing listens on is refused,
    unless the system gives the connection that very port as its own: it is
    then connected to itself, reads back what it sends, and never hears no.
    Windows hands a port that was just freed to the next connection.
    """
    try:
        return sock.getsockname()[:2] == sock.getpeername()[:2]
    except OSError:
        return False


def _no_delay(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def keep_alive(sock: socket.socket) -> None:
    """``_keep_alive``, for other connections that must notice a PC that went away."""
    _keep_alive(sock)


def _keep_alive(sock: socket.socket) -> None:
    """Has the system notice a peer that went away without closing the connection."""
    total_ms = (KEEPALIVE_IDLE + KEEPALIVE_INTERVAL * KEEPALIVE_COUNT) * 1000
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if sys.platform == "win32":
        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, KEEPALIVE_IDLE * 1000, KEEPALIVE_INTERVAL * 1000))
        return
    for name, value in (("TCP_KEEPIDLE", KEEPALIVE_IDLE),   # Linux
                        ("TCP_KEEPALIVE", KEEPALIVE_IDLE),  # macOS's name for the same
                        ("TCP_KEEPINTVL", KEEPALIVE_INTERVAL),
                        ("TCP_KEEPCNT", KEEPALIVE_COUNT),
                        # Linux: keepalive does not run while an answer waits to be sent,
                        # and sending is retried for a quarter of an hour without this.
                        ("TCP_USER_TIMEOUT", total_ms)):
        option = getattr(socket, name, None)
        if option is not None:
            sock.setsockopt(socket.IPPROTO_TCP, option, value)


def _went_silent(e: OSError) -> bool:
    """True when the other side stopped answering, as opposed to closing or resetting the connection.

    The system reports it as a timeout, or on Windows as "dropped on reset" (the probes got no answer)
    or "aborted" (an answer could not be delivered)."""
    return isinstance(e, TimeoutError) or e.errno in (errno.ENETRESET, errno.ECONNABORTED,
                                                      errno.EHOSTUNREACH, errno.ENETUNREACH)


def _pump(src: socket.socket, dst: socket.socket) -> OSError | None:
    """Copies ``src`` to ``dst`` until ``src`` ends; returns the error if one of them broke instead."""
    error = None
    try:
        while True:
            data = src.recv(BUFFER)
            if not data:
                break
            dst.sendall(data)
    except OSError as e:
        error = e
    finally:
        # Tell the other side this direction is done, and wake the other pump.
        for s, how in ((dst, socket.SHUT_WR), (src, socket.SHUT_RD)):
            try:
                s.shutdown(how)
            except OSError:
                pass
    return error


def _join(client: socket.socket, target: tuple[str, int], log: Callable[[str], None], peer: str = "?",
          what: str = "the RPC server") -> None:
    try:
        upstream = socket.create_connection(target, timeout=10)
    except OSError as e:
        log(f"gate: cannot reach {what} at {target[0]}:{target[1]} ({e})")
        client.close()
        return
    upstream.settimeout(None)
    for s in (client, upstream):
        _no_delay(s)
    try:
        _keep_alive(client)
    except OSError as e:
        # The gate works without it; only a main PC that vanishes is then not noticed.
        log(f"gate: the system cannot probe the connection from {peer} ({e}); if that PC vanishes, "
            "restart the worker to free what it loaded")
    errors: list[OSError] = []

    def copy(src: socket.socket, dst: socket.socket) -> None:
        error = _pump(src, dst)
        if error is not None:
            errors.append(error)

    back = threading.Thread(target=copy, args=(upstream, client), daemon=True)
    back.start()
    copy(client, upstream)
    back.join()
    client.close()
    upstream.close()
    # The RPC server sits on this PC, so only the main PC's side can go silent.
    if any(_went_silent(e) for e in errors):
        log(f"gate: the PC at {peer} stopped answering; its connection is closed, "
            "and the RPC server frees what that PC had loaded")


class Gate:
    """Listens on ``listen`` and joins allowed clients to the RPC server at ``target``.

    ``what`` names the program at ``target`` in messages, for a gate in front of something else
    (``serving`` puts one in front of llama-server's API)."""

    def __init__(self, listen: tuple[str, int], target: tuple[str, int], allowed: list,
                 log: Callable[[str], None], what: str = "the RPC server", announce: bool = False) -> None:
        family = socket.AF_INET6 if ":" in listen[0] else socket.AF_INET
        self.sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            exclusive(self.sock)
            self.sock.bind(listen)
            self.sock.listen(8)
        except OSError:
            self.sock.close()
            raise
        self.address = self.sock.getsockname()[:2]
        self.target = target
        self.allowed = allowed
        self.log = log
        self.what = what
        self.announce = announce
        self.users: dict[str, int] = {}  # open connections per address
        self.users_lock = threading.Lock()
        self.refused = 0
        self._thread: threading.Thread | None = None

    def start(self) -> "Gate":
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()
        return self

    def _accept(self) -> None:
        while True:
            try:
                client, peer = self.sock.accept()
            except OSError:
                return  # closed
            if not is_allowed(peer[0], self.allowed):
                self.refused += 1
                self.log(f"gate: refused a connection from {peer[0]} (not in --allow)")
                client.close()
                continue
            threading.Thread(target=self._serve_one, args=(client, peer[0]), daemon=True).start()

    def _serve_one(self, client: socket.socket, peer: str) -> None:
        """One connection; with ``announce``, says when a PC starts and stops using what is behind the gate."""
        with self.users_lock:
            first = self.users.get(peer, 0) == 0
            self.users[peer] = self.users.get(peer, 0) + 1
        if first and self.announce:
            self.log(f"gate: the PC at {peer} is using these GPUs")
        try:
            _join(client, self.target, self.log, peer, self.what)
        finally:
            with self.users_lock:
                self.users[peer] -= 1
                last = self.users[peer] == 0
                if last:
                    del self.users[peer]
            if last and self.announce:
                self.log(f"gate: the PC at {peer} is done with these GPUs")

    def close(self) -> None:
        # On Linux a close alone leaves a blocked accept() listening; shutdown wakes it.
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
