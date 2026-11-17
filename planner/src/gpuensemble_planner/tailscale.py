"""This PC's Tailscale network, read with the tailscale program; nothing is changed.

Two PCs at different sites reach each other through a VPN such as Tailscale.
A broadcast stays on the LAN and does not cross it, so ``--remote auto``
could not find a worker there. ``tailscale status --json`` gives what is
needed, as a normal user:

* the other PCs of the tailnet that are online, to ask each of them directly;
* whether an address belongs to a PC signed in to Tailscale with the same
  account as this one. A worker started with ``--lan`` lets such a PC in
  without asking: it is its user's own, and nobody may be sitting at a worker
  at another site to type ``y``. Taildrop draws the same line: it takes files
  from one's own devices only. Tagged devices (servers) and devices shared
  from another account are not one's own, and are asked about as on a LAN.

Tailscale tells each PC apart by its key, so an address it lists cannot be
taken by another PC of the tailnet.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
from dataclasses import dataclass

MAX_PEERS = 64  # a big company tailnet is not a place to ask every PC


@dataclass
class Peer:
    address: str  # its IPv4 address on the tailnet
    name: str
    own: bool  # signed in with this PC's account: not tagged, not shared from another account


def find_exe() -> str | None:
    exe = shutil.which("tailscale")
    if exe:
        return exe
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432")):
        if base:
            path = os.path.join(base, "Tailscale", "tailscale.exe")
            if os.path.isfile(path):
                return path
    return None


def read_status(timeout: float = 5.0) -> dict | None:
    """``tailscale status --json`` as a dict; None without Tailscale, or when it is not connected."""
    exe = find_exe()
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "status", "--json"], capture_output=True, timeout=timeout,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        status = json.loads(out.stdout.decode("utf-8", errors="replace"))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not isinstance(status, dict) or status.get("BackendState") != "Running":
        return None
    return status


def _ipv4(node: dict) -> str | None:
    for address in node.get("TailscaleIPs") or []:
        try:
            if ipaddress.ip_address(address).version == 4:
                return str(address)
        except ValueError:
            continue
    return None


def _own(node: dict, me: dict) -> bool:
    user = me.get("UserID")
    return bool(user) and node.get("UserID") == user and not node.get("Tags") and not me.get("Tags") \
        and not node.get("ShareeNode")


def peers(status: dict | None) -> list[Peer]:
    """The other PCs of the tailnet that are online, at most ``MAX_PEERS``."""
    if not status:
        return []
    me = status.get("Self") if isinstance(status.get("Self"), dict) else {}
    nodes = status.get("Peer") if isinstance(status.get("Peer"), dict) else {}
    found = []
    for node in nodes.values():
        if not isinstance(node, dict) or not node.get("Online"):
            continue
        address = _ipv4(node)
        if address:
            found.append(Peer(address, str(node.get("HostName") or address)[:64], _own(node, me)))
    return sorted(found, key=lambda p: (not p.own, p.name, p.address))[:MAX_PEERS]


def own_pc(address: str, status: dict | None) -> Peer | None:
    """The tailnet PC with that address, if it is signed in with this PC's account."""
    try:
        wanted = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return None
    if not status:
        return None
    me = status.get("Self") if isinstance(status.get("Self"), dict) else {}
    nodes = status.get("Peer") if isinstance(status.get("Peer"), dict) else {}
    for node in nodes.values():
        if not isinstance(node, dict) or not _own(node, me):
            continue
        for other in node.get("TailscaleIPs") or []:
            try:
                if ipaddress.ip_address(other) == wanted:
                    return Peer(str(wanted), str(node.get("HostName") or wanted)[:64], True)
            except ValueError:
                continue
    return None


def is_tailnet_address(address: str) -> bool:
    """Whether an address is in the ranges Tailscale gives its PCs (100.64.0.0/10, fd7a:115c:a1e0::/48)."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    net = "100.64.0.0/10" if ip.version == 4 else "fd7a:115c:a1e0::/48"
    return ip in ipaddress.ip_network(net)
