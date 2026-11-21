"""Whether a main PC that asks is on this PC's home LAN, for ``worker --trust-lan``.

A worker with ``--trust-lan`` lets in, without asking, a main PC whose address
is on the same local network as one of this PC's own addresses, when Windows
treats that network as private (or as a work domain). That is what a user says
by choosing "Private network" for it: the PCs on it are trusted. On a public
network (a shop, a school, a hotel) nobody is let in this way, nor through a
VPN (Tailscale has its own rule, see ``tailscale``).

The networks are read with the firewall's notes (``firewall.read_networks``),
which takes a few seconds, so ``HomeLan`` reads them in the background and
again every ``REFRESH`` seconds: a laptop carried from home to a café stops
trusting the home addresses.

The trust is not written down: a PC let in this way is asked about again on
the next start, and one this PC's user refused is never let in this way
(``worker.json``'s ``deny``).
"""

from __future__ import annotations

import ipaddress
import threading
import time
from typing import Callable

from . import firewall, tailscale

REFRESH = 60.0
TRUSTED_PROFILES = ("private", "domain")
# A network wider than this is not a home LAN (a /16 is 65,536 addresses already).
WIDEST_PREFIX = 16


def home_network(address: str, networks: list[firewall.Network]) -> firewall.Network | None:
    """The private network of this PC that ``address`` is on, or None.

    Only IPv4 private addresses (192.168.x.x, 10.x.x.x, 172.16-31.x.x) count;
    Tailscale's range, link-local and loopback addresses do not.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return None
    if ip.version != 4 or not ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified:
        return None
    if tailscale.is_tailnet_address(address):
        return None
    for net in networks:
        if net.profile not in TRUSTED_PROFILES:
            continue
        for subnet in net.subnets:
            if subnet.version == 4 and subnet.prefixlen >= WIDEST_PREFIX and ip in subnet:
                own = {ipaddress.ip_address(a.split("%", 1)[0]) for a in net.addresses if "." in a}
                if ip in own:
                    return None  # this PC itself
                return net
    return None


class HomeLan:
    """This PC's networks, read in the background and kept fresh; ``find(address)`` answers at once."""

    def __init__(self, read: Callable[[], list[firewall.Network]] = firewall.read_networks,
                 refresh: float = REFRESH, clock: Callable[[], float] = time.monotonic,
                 on_change: Callable[[], None] | None = None) -> None:
        self.read = read
        self.refresh = refresh
        self.clock = clock
        self.on_change = on_change
        self.networks: list[firewall.Network] = []
        self.read_once = threading.Event()
        self.closed = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self) -> None:
        try:
            networks = self.read()
        except Exception:  # noqa: BLE001 - PowerShell missing or broken: keep what was known
            networks = None
        if networks is not None:
            self.networks = networks
        self.read_once.set()
        if self.on_change is not None:
            self.on_change()

    def start(self) -> "HomeLan":
        def run() -> None:
            while not self.closed.is_set():
                self.update()
                self.closed.wait(self.refresh)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def find(self, address: str, wait: float = 0.0) -> firewall.Network | None:
        """The home network ``address`` is on; waits up to ``wait`` seconds for the first reading."""
        if wait > 0:
            self.read_once.wait(wait)
        return home_network(address, self.networks)

    def close(self) -> None:
        self.closed.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
