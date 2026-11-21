"""Boost mode: lock the clocks of this PC's cards while a model runs (``up --boost``, Windows).

Holding a card open with Vulkan (``cardhold``) keeps a card that has no display
at full clocks while it waits for the other stages, but not the card that
drives the display: on the E5 rig the 2080 Ti still fell to P3 between
tokens. ``nvidia-smi -lgc MIN,MAX`` sets a floor for the card's graphics
clock, and with it gpt-oss-120b on three cards ran at 17.7-18.0 tok/s
instead of 15.3-15.4 (benchmarks/clock-hold.md).

Locking clocks needs administrator rights, so it is off by default and asked
for each time: ``up --boost`` starts a small helper as administrator (Windows
shows its confirmation dialog). The helper locks the cards, waits for this
process to end, and then resets them with ``nvidia-smi -rgc``, also when this
process is killed. The cards draw more power while locked. A reboot resets
the clocks too.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

from .devices import Device
from .plan import Plan

# The floor is this share of the card's highest graphics clock: 2190 MHz -> 1350 on the 2080 Ti,
# the floor that was measured (2130 -> 1320 on the 2080 SUPER).
FLOOR_SHARE = 0.62
STEP_MHZ = 15
HELPER_WAIT = 120.0  # seconds to wait for the user to answer the administrator dialog


@dataclass(frozen=True)
class Lock:
    index: int  # nvidia-smi index
    min_mhz: int
    max_mhz: int

    def arg(self) -> str:
        return f"{self.index}:{self.min_mhz}:{self.max_mhz}"

    @classmethod
    def parse(cls, text: str) -> "Lock":
        index, low, high = (int(part) for part in text.split(":"))
        return cls(index, low, high)


def local_cards(plan: Plan, devices: list[Device]) -> list[int]:
    """nvidia-smi indices of this PC's CUDA cards that hold layers in the plan."""
    by_name = {d.name: d for d in devices}
    cards: list[int] = []
    for s in plan.stages:
        if s.n_layers <= 0:
            continue
        dev = by_name[s.device]
        name = dev.llama_device or ""
        if dev.is_remote or not name.startswith("CUDA") or not name[4:].isdigit():
            continue
        if int(name[4:]) not in cards:
            cards.append(int(name[4:]))
    return cards


def lock_for(index: int, max_mhz: int) -> Lock:
    floor = int(max_mhz * FLOOR_SHARE) // STEP_MHZ * STEP_MHZ
    return Lock(index, floor, max_mhz)


def parse_max_clocks(out: str) -> dict[int, int]:
    """{index: highest graphics clock in MHz} from ``--query-gpu=index,clocks.max.sm --format=csv,noheader,nounits``."""
    clocks: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            clocks[int(parts[0])] = int(parts[1])
    return clocks


def _nvidia_smi() -> str:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise OSError("nvidia-smi was not found")
    return exe


def locks_for(indices: list[int], smi: str) -> list[Lock]:
    out = subprocess.run([smi, "--query-gpu=index,clocks.max.sm", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=30).stdout
    clocks = parse_max_clocks(out)
    return [lock_for(i, clocks[i]) for i in indices if i in clocks]


# --- Starting the helper as administrator ---

class _ShellExecuteInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint32), ("fMask", ctypes.c_ulong), ("hwnd", ctypes.c_void_p),
                ("lpVerb", ctypes.c_wchar_p), ("lpFile", ctypes.c_wchar_p), ("lpParameters", ctypes.c_wchar_p),
                ("lpDirectory", ctypes.c_wchar_p), ("nShow", ctypes.c_int), ("hInstApp", ctypes.c_void_p),
                ("lpIDList", ctypes.c_void_p), ("lpClass", ctypes.c_wchar_p), ("hkeyClass", ctypes.c_void_p),
                ("dwHotKey", ctypes.c_uint32), ("hIconOrMonitor", ctypes.c_void_p), ("hProcess", ctypes.c_void_p)]


SEE_MASK_NOCLOSEPROCESS = 0x40
SW_HIDE = 0
ERROR_CANCELLED = 1223


def helper_command(locks: list[Lock], smi: str, parent: int, status: str) -> tuple[str, str]:
    """The program and parameters the administrator helper is started with."""
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (f"import sys; sys.path.insert(0, {package_dir!r}); "
            "from gpuensemble_planner.boost import helper_main; sys.exit(helper_main(sys.argv[1:]))")
    args = ["-I", "-c", code, "--smi", smi, "--parent", str(parent), "--status", status]
    args += [lock.arg() for lock in locks]
    return sys.executable, subprocess.list2cmdline(args)


class Boost:
    """The administrator helper that keeps the cards locked while this process runs."""

    def __init__(self, locks: list[Lock], status: str, process) -> None:
        self.locks = locks
        self.status = status
        self._process = process

    @classmethod
    def start(cls, locks: list[Lock], smi: str, wait: float = HELPER_WAIT) -> "Boost":
        """Start the helper and wait until it has locked the cards. Raises OSError if that does not happen."""
        if sys.platform != "win32":
            raise OSError("boost mode is only on Windows")
        fd, status = tempfile.mkstemp(prefix="gpuensemble-boost-", suffix=".json")
        os.close(fd)
        os.remove(status)
        program, params = helper_command(locks, smi, os.getpid(), status)
        info = _ShellExecuteInfo(cbSize=ctypes.sizeof(_ShellExecuteInfo), fMask=SEE_MASK_NOCLOSEPROCESS,
                                 lpVerb="runas", lpFile=program, lpParameters=params, nShow=SW_HIDE)
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        if not shell32.ShellExecuteExW(ctypes.byref(info)):
            error = ctypes.get_last_error()
            if error == ERROR_CANCELLED:
                raise OSError("the administrator dialog was declined")
            raise OSError(f"could not start the helper as administrator (error {error})")
        boost = cls(locks, status, info.hProcess)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            state = boost.read_status()
            if state.get("locked") is not None:
                if not state["locked"]:
                    raise OSError(f"nvidia-smi did not lock the clocks: {state.get('error', '')}".strip())
                return boost
            if boost._exited():
                raise OSError("the helper ended before it locked the clocks")
            time.sleep(0.2)
        raise OSError("the helper did not lock the clocks in time")

    def read_status(self) -> dict:
        try:
            with open(self.status, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _exited(self) -> bool:
        if not self._process:
            return True
        return ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(self._process), 0) == 0

    def close(self, wait: float = 15.0) -> bool:
        """Ask the helper to reset the clocks now; True once it says it did."""
        try:
            with open(self.status + ".stop", "w"):
                pass
        except OSError:
            return False
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if self.read_status().get("reset") is not None:
                break
            time.sleep(0.2)
        reset = bool(self.read_status().get("reset"))
        if self._process:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self._process))
            self._process = None
        for path in (self.status, self.status + ".stop"):
            try:
                os.remove(path)
            except OSError:
                pass
        return reset


def start_for(plan: Plan, devices: list[Device], log: Callable[[str], None]) -> Boost | None:
    """Boost the plan's local cards, saying what was done; None when nothing was locked."""
    indices = local_cards(plan, devices)
    if not indices:
        log("note: --boost: no card of this PC runs a part of this model; nothing to lock")
        return None
    try:
        smi = _nvidia_smi()
        locks = locks_for(indices, smi)
        if not locks:
            raise OSError("nvidia-smi gave no clocks for these cards")
        log("boost: locking the clocks of " + ", ".join(f"CUDA{lk.index} ({lk.min_mhz}-{lk.max_mhz} MHz)"
                                                         for lk in locks)
            + "; Windows asks for administrator rights")
        boost = Boost.start(locks, smi)
    except (OSError, subprocess.SubprocessError) as e:
        log(f"note: --boost: the clocks are not locked ({e}); the model runs without boost")
        return None
    log("boost: on; the clocks are reset when the model stops")
    return boost


# --- The helper, running as administrator ---

def _write(status: str, state: dict) -> None:
    tmp = status + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, status)


def _parent_alive(pid: int) -> Callable[[], bool]:
    if sys.platform != "win32":
        def alive() -> bool:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        return alive
    SYNCHRONIZE = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return lambda: False

    def alive() -> bool:
        return ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0) != 0
    return alive


def helper_main(argv: list[str], run=subprocess.run, sleep=time.sleep, alive=None) -> int:
    """Lock, wait for the parent to end or for the stop file, reset. Returns 0 when the reset worked."""
    import argparse
    parser = argparse.ArgumentParser(prog="gpuensemble boost helper")
    parser.add_argument("--smi", required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("locks", nargs="+")
    args = parser.parse_args(argv)
    locks = [Lock.parse(text) for text in args.locks]
    state: dict = {"locks": [lk.arg() for lk in locks]}
    errors = []
    done: list[Lock] = []
    for lk in locks:
        r = run([args.smi, "-i", str(lk.index), "-lgc", f"{lk.min_mhz},{lk.max_mhz}"],
                capture_output=True, text=True)
        if r.returncode == 0:
            done.append(lk)
        else:
            errors.append(f"CUDA{lk.index}: {(r.stdout + r.stderr).strip()}")
    state["locked"] = bool(done) and not errors
    if errors:
        state["error"] = "; ".join(errors)
    _write(args.status, state)
    alive = alive or _parent_alive(args.parent)
    while done and not errors and alive() and not os.path.exists(args.status + ".stop"):
        sleep(1.0)
    reset_ok = True
    for lk in done:
        r = run([args.smi, "-i", str(lk.index), "-rgc"], capture_output=True, text=True)
        reset_ok = reset_ok and r.returncode == 0
    state["reset"] = reset_ok
    _write(args.status, state)
    return 0 if reset_ok else 1
