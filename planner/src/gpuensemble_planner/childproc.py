"""Start llama.cpp so that it never outlives the launcher.

``run`` already stops llama-server when Python exits normally or on Ctrl+C.
A hard kill of Python (Task Manager, ``taskkill /F``, a closed console, a
crash) skips that, and llama-server kept running with its VRAM. ``spawn``
ties the child to this process at the OS level instead:

* Windows: the child goes into a job object with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Only this process holds the job's
  handle, so when it dies for any reason Windows closes the handle and ends
  every process in the job.
* Linux: the child asks the kernel for SIGKILL when its parent dies
  (``prctl(PR_SET_PDEATHSIG)``).
* Elsewhere (macOS): no OS mechanism; only the normal cleanup applies.

If the OS call fails, the child still runs and ``Guard.active`` is False.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Guard:
    """Keeps the OS handle that ties the child to us; ``active`` says whether that worked."""

    active: bool
    how: str
    handle: object = None  # Windows job handle; must stay open while we run


# Handles of live jobs, so they are only closed when this process exits.
_JOBS: list[object] = []


def spawn(argv: list[str], env: dict[str, str] | None = None, **popen) -> tuple[subprocess.Popen, Guard]:
    """Start ``argv`` like ``subprocess.Popen`` and make it die with this process.

    Extra keyword arguments (``stdout=`` and so on) go to Popen.
    """
    if sys.platform == "win32":
        proc = subprocess.Popen(argv, env=env, **popen)
        return proc, _windows_job(proc)
    if sys.platform.startswith("linux"):
        proc = subprocess.Popen(argv, env=env, preexec_fn=_linux_pdeathsig, **popen)
        return proc, Guard(True, "PR_SET_PDEATHSIG")
    proc = subprocess.Popen(argv, env=env, **popen)
    return proc, Guard(False, "none on this OS")


def _linux_pdeathsig() -> None:
    """Runs in the child between fork and exec."""
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


def _windows_job(proc: subprocess.Popen) -> Guard:
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return Guard(False, f"CreateJobObject failed ({ctypes.get_last_error()})")
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                            ctypes.byref(info), ctypes.sizeof(info)):
        return Guard(False, f"SetInformationJobObject failed ({ctypes.get_last_error()})")
    # Popen keeps the child's process handle in _handle on Windows.
    if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):  # type: ignore[attr-defined]
        return Guard(False, f"AssignProcessToJobObject failed ({ctypes.get_last_error()})")
    _JOBS.append(job)
    return Guard(True, "job object", job)


def pid_alive(pid: int) -> bool:
    """Whether a process with this id still runs (for tests and diagnostics)."""
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        finally:
            kernel32.CloseHandle(handle)
        return code.value == STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie (dead, not yet reaped) still answers kill(0); check its state.
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True
