"""Why other PCs may not reach a worker on Windows, read without changing anything.

Two settings decide whether another PC can reach Python on this one:

* the network's profile: Windows treats each network as public or private
  (or a domain), and a firewall rule applies to some of these only;
* the firewall rules for this Python. The first time Python listens on a
  network, Windows asks whether to allow it, and leaves allow rules for the
  profiles ticked and block rules for the others (Cancel blocks it). It does
  not ask again while any rule for that program exists.

``reachability_notes()`` reads both with PowerShell (``Get-NetConnectionProfile``,
and the firewall's rules for this Python's program file through its COM
object, which needs no administrator) and says what stops other PCs, and
where to change it. It never changes a setting. Given the address the worker
listens on, it speaks only of the network that has that address: a worker on
its Tailscale address does not care that the Ethernet is public.
"""

from __future__ import annotations

import ipaddress
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable

# Network categories as Get-NetConnectionProfile names them, and the firewall profile each one uses.
CATEGORY_PROFILE = {"Public": "public", "Private": "private", "DomainAuthenticated": "domain"}

# The firewall cmdlets (Get-NetFirewallRule, ...) need an administrator; the
# firewall's COM object reads the rules as a normal user, and quickly.
# "rules<TAB>read" marks that they could be read at all.
SCRIPT = r"""
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$ErrorActionPreference = 'SilentlyContinue'
Get-NetConnectionProfile | ForEach-Object { "net`t$($_.InterfaceAlias)`t$($_.Name)`t$($_.NetworkCategory)" }
Get-NetIPAddress | ForEach-Object { "addr`t$($_.InterfaceAlias)`t$($_.IPAddress)`t$($_.PrefixLength)" }
$exe = $env:GPUENSEMBLE_FIREWALL_PROGRAM
try {
  $policy = New-Object -ComObject HNetCfg.FwPolicy2 -ErrorAction Stop
  foreach ($r in $policy.Rules) {
    if ($r.Direction -eq 1 -and $r.Enabled -and $r.ApplicationName -and
        [Environment]::ExpandEnvironmentVariables($r.ApplicationName) -ieq $exe) {
      "rule`t$(if ($r.Action -eq 1) {'Allow'} else {'Block'})`t$($r.Profiles)`t$($r.Name)"
    }
  }
  "rules`tread"
} catch {}
"""

# NET_FW_PROFILE_TYPE2 bits of a rule's Profiles.
PROFILE_BITS = {1: "domain", 2: "private", 4: "public"}

ALLOW_APP = ("Windows Security > Firewall & network protection > Allow an app through firewall, "
             "and tick Python for {kind} networks")


@dataclass
class Network:
    alias: str  # the network card, e.g. Ethernet
    name: str  # the network, e.g. Network 2
    profile: str  # public, private or domain
    addresses: list[str] = field(default_factory=list)  # this PC's addresses on it
    subnets: list = field(default_factory=list)  # the networks those addresses are in (ipaddress networks)


@dataclass
class Rule:
    action: str  # Allow or Block
    profiles: set[str]
    name: str


def parse(lines: list[str]) -> tuple[list[Network], list[Rule] | None]:
    """The networks, and this Python's inbound rules (None when they could not be read)."""
    networks, rules, read, addresses, subnets = [], [], False, {}, {}
    for line in lines:
        parts = line.rstrip("\r\n").split("\t")
        if parts[0] == "net" and len(parts) == 4 and parts[3] in CATEGORY_PROFILE:
            networks.append(Network(parts[1], parts[2], CATEGORY_PROFILE[parts[3]]))
        elif parts[0] == "addr" and len(parts) in (3, 4):
            addresses.setdefault(parts[1], []).append(parts[2])
            if len(parts) == 4:
                try:
                    subnets.setdefault(parts[1], []).append(
                        ipaddress.ip_network(f"{parts[2].split('%')[0]}/{int(parts[3])}", strict=False))
                except ValueError:
                    pass
        elif parts[0] == "rule" and len(parts) == 4:
            try:
                bits = int(parts[2])
            except ValueError:
                continue
            rules.append(Rule(parts[1], {name for bit, name in PROFILE_BITS.items() if bits & bit}, parts[3]))
        elif parts == ["rules", "read"]:
            read = True
    for net in networks:
        net.addresses = addresses.get(net.alias, [])
        net.subnets = subnets.get(net.alias, [])
    return networks, rules if read else None


def _same_address(a: str, b: str) -> bool:
    try:
        return ipaddress.ip_address(a.split("%")[0]) == ipaddress.ip_address(b.split("%")[0])
    except ValueError:
        return a == b


def listening_on(networks: list[Network], host: str | None) -> list[Network]:
    """The networks a worker listening on ``host`` can be reached through.

    Every network for no address or a wildcard one, or when this PC's
    addresses could not be read; else the one that has ``host``, if any.
    """
    if not host or host in ("0.0.0.0", "::") or not any(n.addresses for n in networks):
        return networks
    return [n for n in networks if any(_same_address(a, host) for a in n.addresses)]


def notes(networks: list[Network], rules: list[Rule] | None) -> list[str]:
    """What keeps other PCs from reaching Python here, one note per network that has a problem.

    ``rules`` None means they could not be read; then only a public network is worth a word.
    """
    found = []
    for net in networks:
        where = f"the {net.profile} network '{net.name}' ({net.alias})"
        blocking = [r for r in rules or [] if r.action == "Block" and net.profile in r.profiles]
        allowed = any(r.action == "Allow" and net.profile in r.profiles for r in rules or [])
        if allowed and not blocking:  # a block rule wins over any allow rule
            continue
        make_private = (f"On a home LAN, make it private (Settings > Network & internet > {net.alias} > "
                        "Private network), and allow Python on private networks if Windows asks")
        if net.profile == "public":
            # Allowing Python "on private networks" (as the worker suggests) also makes Windows block it on
            # public ones; on a home LAN the network's profile is what is wrong.
            if rules is None:
                found.append(f"Windows treats the network '{net.name}' ({net.alias}) as public; if Python is "
                             "allowed on private networks only, other PCs cannot reach this worker through it. "
                             + make_private)
            else:
                found.append(f"Windows treats the network '{net.name}' ({net.alias}) as public, and Python is not "
                             "allowed in on public networks, so other PCs cannot reach this worker through it. "
                             + make_private)
        elif blocking:
            found.append(f"Windows Firewall blocks Python on {where} (rule '{blocking[0].name}'), so other PCs "
                         "cannot reach this worker through it. To let them in, open "
                         + ALLOW_APP.format(kind=net.profile))
        elif rules:
            # Windows asks only for a program it has no rule for; this one has rules, but none for this profile.
            found.append(f"Python is not allowed in on {where}, so other PCs cannot reach this worker through it. "
                         "To let them in, open " + ALLOW_APP.format(kind=net.profile))
    return found


def _powershell(program: str, timeout: float = 30.0) -> list[str]:
    env = dict(os.environ, GPUENSEMBLE_FIREWALL_PROGRAM=program)
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", SCRIPT],
                             capture_output=True, timeout=timeout, env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    return out.stdout.decode("utf-8", errors="replace").splitlines()


def read_networks(read: Callable[[str], list[str]] | None = None, exe: str | None = None) -> list[Network]:
    """This PC's networks with their profiles and subnets; [] where it cannot tell (not Windows)."""
    if read is None:
        if sys.platform != "win32":
            return []
        read = _powershell
    return parse(read(exe or program()))[0]


def program() -> str:
    """The python.exe that listens: a venv's python.exe only starts the base one."""
    return getattr(sys, "_base_executable", None) or sys.executable


def reachability_notes(read: Callable[[str], list[str]] | None = None, exe: str | None = None,
                       host: str | None = None) -> list[str]:
    """Notes on what keeps other PCs from reaching this Python on ``host``; [] when nothing does or it cannot tell."""
    if read is None:
        if sys.platform != "win32":
            return []
        read = _powershell
    networks, rules = parse(read(exe or program()))
    return notes(listening_on(networks, host), rules)
