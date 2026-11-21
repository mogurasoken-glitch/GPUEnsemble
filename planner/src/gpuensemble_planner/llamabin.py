"""Install llama.cpp for this PC (``gpuensemble-plan setup``).

Until the GPUEnsemble executor exists, plans run on stock llama.cpp. This
module picks the right prebuilt release from github.com/ggml-org/llama.cpp,
downloads it into ``<home>/bin/llama.cpp/<tag>-<backend>/`` and checks that
it starts and sees the GPUs.

Which build
-----------
* NVIDIA with a driver that runs CUDA 12.4 (Windows 551.61+, Linux 550.54+):
  the CUDA 12.4 build plus its CUDA runtime DLLs (Windows). It was the
  fastest backend for every model on the development rig.
* NVIDIA with an older driver, or Linux (no CUDA builds are published): Vulkan.
* An AMD Radeon or Intel Arc card (with or without NVIDIA cards beside it): Vulkan, which
  runs every maker's cards in one process.
* No GPU: the CPU build. It can still use the cards of other PCs (llama.cpp's RPC is in every
  build), so a PC without a GPU can be the main PC.

The release is pinned to the one measured on the development rig
(``DEFAULT_TAG``); ``--tag latest`` takes the newest.

Old Visual C++ runtime (Windows)
--------------------------------
Current builds crash at start (0xC0000005 in MSVCP140.dll) when Windows has a
Visual C++ runtime older than 14.40; on the development rig it was 14.28.
``setup`` then looks for newer copies of the three runtime DLLs that other
programs already installed (Microsoft Edge ships them) and puts them next to
llama.cpp, which touches nothing outside the GPUEnsemble folder. If none are
found it says where to get the Visual C++ Redistributable.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable

from . import hub

REPO = "ggml-org/llama.cpp"
API = "https://api.github.com"
DEFAULT_TAG = "b12375"  # measured on the development rig (CUDA 12.4, driver 560.94)
CUDA_VERSION = "12.4"
CUDA_MIN_DRIVER = {"win32": (551, 61), "linux": (550, 54)}
VC_RUNTIME = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
VC_MIN_VERSION = (14, 40)
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
ACCESS_VIOLATION = 0xC0000005
DLL_NOT_FOUND = 0xC0000135


@dataclass
class Asset:
    name: str
    url: str
    size: int
    sha256: str | None = None


@dataclass
class Install:
    tag: str
    backend: str
    folder: str  # where llama-server and its libraries are
    devices: list[str] = field(default_factory=list)

    @property
    def server(self) -> str:
        return self.tool("llama-server")

    def tool(self, name: str) -> str:
        """Path of a llama.cpp program in this install (``.exe`` if the build has one)."""
        exe = os.path.join(self.folder, name + ".exe")
        return exe if os.path.exists(exe) else os.path.join(self.folder, name)


def cuda_cache_env() -> dict[str, str]:
    """A CUDA kernel cache big enough for llama.cpp. The default 256 MB filled up on
    the development rig and every start compiled the kernels again (minutes)."""
    return {} if "CUDA_CACHE_MAXSIZE" in os.environ else {"CUDA_CACHE_MAXSIZE": str(4 << 30)}


def _windows() -> bool:
    return sys.platform == "win32"


def llama_root() -> str:
    return os.path.join(hub.home_dir(), "bin", "llama.cpp")


def _platform() -> str:
    if sys.platform == "win32":
        return "win32"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def nvidia_driver(run: Callable = subprocess.run) -> tuple[int, ...] | None:
    """The NVIDIA driver version, e.g. (560, 94), or None without an NVIDIA GPU."""
    try:
        out = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                  capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    first = out.stdout.strip().splitlines()[0].strip()
    try:
        return tuple(int(x) for x in first.split("."))
    except ValueError:
        return None


def choose_backend(platform: str, driver: tuple[int, ...] | None, others: list[str] | None = None) -> tuple[str, str]:
    """(backend, reason) for this OS, NVIDIA driver and the names of any AMD or Intel cards.

    AMD and Intel cards run on the Vulkan build, which runs NVIDIA cards too: with both makers in
    one PC, Vulkan is the build that pools them all (the CUDA build would see the NVIDIA ones only).
    """
    if others:
        both = " and the NVIDIA card(s)" if driver is not None else ""
        return "vulkan", f"{', '.join(others)} found; the Vulkan build runs {'them' if len(others) > 1 else 'it'}{both}"
    if driver is None:
        return "cpu", "no NVIDIA GPU found"
    shown = ".".join(str(x) for x in driver)
    need = CUDA_MIN_DRIVER.get(platform)
    if platform == "win32" and need and driver >= need:
        return "cuda", f"NVIDIA driver {shown} runs CUDA {CUDA_VERSION}"
    if platform == "win32":
        return "vulkan", (f"NVIDIA driver {shown} is older than {need[0]}.{need[1]}, which CUDA {CUDA_VERSION} "
                          "needs; update the driver for about 10-40% more speed")
    return "vulkan", "llama.cpp publishes no CUDA build for this OS"


def _asset_patterns(tag: str, platform: str, backend: str) -> list[str]:
    t = re.escape(tag) if tag != "latest" else r"b\d+"
    archive = r"\.(?:zip|tar\.gz)"
    if platform == "win32":
        if backend == "cuda":
            return [rf"llama-{t}-bin-win-cuda-{re.escape(CUDA_VERSION)}-x64\.zip",
                    rf"cudart-llama-bin-win-cuda-{re.escape(CUDA_VERSION)}-x64\.zip"]
        return [rf"llama-{t}-bin-win-{backend}-x64\.zip"]
    if platform == "linux":
        return [rf"llama-{t}-bin-ubuntu-vulkan-x64{archive}" if backend == "vulkan"
                else rf"llama-{t}-bin-ubuntu-x64{archive}"]
    if platform == "darwin":
        return [rf"llama-{t}-bin-macos-arm64{archive}"]
    raise LookupError(f"no llama.cpp build for {platform}")


def release_assets(tag: str, opener: Callable = urllib.request.urlopen) -> tuple[str, list[Asset]]:
    """(tag, assets) of a llama.cpp release; ``tag`` may be "latest"."""
    path = "latest" if tag == "latest" else f"tags/{tag}"
    req = hub._request(f"{API}/repos/{REPO}/releases/{path}", os.environ.get("GITHUB_TOKEN") or None,
                       headers={"Accept": "application/vnd.github+json"})
    try:
        with opener(req, timeout=60) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise LookupError(f"llama.cpp has no release {tag}") from e
        if e.code in (403, 429):
            raise OSError("GitHub refused the request (hourly limit for anonymous use); "
                          "wait an hour or set GITHUB_TOKEN") from e
        raise
    assets = []
    for a in data.get("assets", []):
        digest = a.get("digest") or ""
        assets.append(Asset(a["name"], a["browser_download_url"], int(a.get("size", 0)),
                            digest[len("sha256:"):] if digest.startswith("sha256:") else None))
    return data["tag_name"], assets


def pick_assets(assets: list[Asset], tag: str, platform: str, backend: str) -> list[Asset]:
    chosen = []
    for pattern in _asset_patterns(tag, platform, backend):
        match = [a for a in assets if re.fullmatch(pattern, a.name)]
        if not match:
            raise LookupError(f"release {tag} has no file matching {pattern}")
        chosen.append(match[0])
    return chosen


def _extract(archive: str, dest: str) -> None:
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)  # drops absolute paths and ".." parts
    else:
        with tarfile.open(archive) as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(dest, filter="data")
            else:  # Python < 3.12: refuse anything that leaves dest
                root = os.path.realpath(dest)
                for m in t.getmembers():
                    target = os.path.realpath(os.path.join(dest, m.name))
                    if not target.startswith(root + os.sep) or m.issym() or m.islnk():
                        raise OSError(f"{archive}: unsafe entry {m.name}")
                t.extractall(dest)


def _find_server(root: str) -> str | None:
    """The folder that holds llama-server (the build decides whether it ends in .exe)."""
    for folder, _, names in os.walk(root):
        if "llama-server.exe" in names or "llama-server" in names:
            return folder
    return None


def check(install: Install, run: Callable = subprocess.run) -> tuple[int, list[str], str]:
    """Start ``llama-server --list-devices``: (exit code, devices, output)."""
    try:
        out = run([install.server, "--list-devices"], capture_output=True, text=True, encoding="utf-8",
                  errors="replace", timeout=300,
                  env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", **cuda_cache_env()))
    except subprocess.TimeoutExpired:
        return -1, [], "llama-server --list-devices did not finish in 5 minutes"
    text = (out.stdout or "") + (out.stderr or "")
    devices = [m.group(1) for m in re.finditer(r"^\s*((?:CUDA|Vulkan|ROCm|Metal|SYCL)\d+: .*)$", text, re.M)]
    return out.returncode, devices, text


def _file_version(path: str) -> tuple[int, int, int, int] | None:
    """Windows file version of a DLL, e.g. (14, 28, 29914, 0)."""
    if not _windows():
        return None
    import ctypes
    from ctypes import wintypes

    version = ctypes.WinDLL("version")
    size = version.GetFileVersionInfoSizeW(path, None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(path, 0, size, buf):
        return None
    ptr, length = ctypes.c_void_p(), wintypes.UINT()
    if not version.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
        return None
    words = ctypes.cast(ptr, ctypes.POINTER(wintypes.DWORD * 13)).contents
    ms, ls = words[2], words[3]  # VS_FIXEDFILEINFO.dwFileVersionMS / LS
    return ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF


def _runtime_candidates() -> list[str]:
    """Folders of installed programs that carry their own Visual C++ runtime."""
    roots = [os.environ.get(v, "") for v in ("ProgramFiles(x86)", "ProgramFiles")]
    folders = []
    for root in filter(None, roots):
        for pattern in (r"Microsoft\Edge\Application\*", r"Microsoft\EdgeWebView\Application\*",
                        r"Microsoft\EdgeCore\*"):
            folders += glob.glob(os.path.join(root, pattern))
    return [f for f in folders if all(os.path.exists(os.path.join(f, n)) for n in VC_RUNTIME)]


def system_runtime_version() -> tuple[int, ...] | None:
    windir = os.environ.get("SystemRoot", r"C:\Windows")
    return _file_version(os.path.join(windir, "System32", "msvcp140.dll"))


def add_local_runtime(folder: str, candidates: list[str] | None = None,
                      version: Callable[[str], tuple | None] = _file_version) -> str | None:
    """Copy the newest Visual C++ runtime DLLs found on this PC next to llama.cpp.

    Returns the folder they came from, or None when no copy is new enough.
    """
    best, best_version = None, VC_MIN_VERSION
    for c in candidates if candidates is not None else _runtime_candidates():
        versions = [version(os.path.join(c, n)) for n in VC_RUNTIME]
        if any(v is None for v in versions):
            continue
        low = min(versions)
        if low >= best_version:
            best, best_version = c, low
    if best is None:
        return None
    for n in VC_RUNTIME:
        shutil.copy2(os.path.join(best, n), os.path.join(folder, n))
    return best


def _crashed(code: int) -> bool:
    return code & 0xFFFFFFFF in (ACCESS_VIOLATION, DLL_NOT_FOUND)


def installed(root: str | None = None) -> Install | None:
    """The llama.cpp that ``setup`` installed last, if it is still there."""
    try:
        with open(os.path.join(root or llama_root(), "installed.json"), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    inst = Install(data["tag"], data["backend"], data["folder"], data.get("devices", []))
    return inst if os.path.exists(inst.server) else None


def setup(tag: str = DEFAULT_TAG, backend: str = "auto", root: str | None = None, force: bool = False,
          opener: Callable = urllib.request.urlopen, run: Callable = subprocess.run,
          platform: str | None = None, log: Callable[[str], None] | None = None,
          others: list[str] | None = None) -> Install:
    """Download, unpack and check llama.cpp; reuses an earlier install of the same build."""
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    platform = platform or _platform()
    root = root or llama_root()
    if backend == "auto":
        from .devices import other_cards  # devices imports this module

        if others is None:
            # Ask the system only when the commands are real ones (tests pass their own ``run``).
            others = [a.name for a in other_cards()] if platform == _platform() and run is subprocess.run else []
        backend, why = choose_backend(platform, nvidia_driver(run), others)
        log(f"llama.cpp backend: {backend} ({why})")
    previous = None if force else installed(root)
    if previous and (previous.tag, previous.backend) == (tag, backend):
        log(f"llama.cpp {tag} ({backend}) is already installed in {previous.folder}")
        return previous
    tag, assets = release_assets(tag, opener)
    if previous and (previous.tag, previous.backend) == (tag, backend):
        log(f"llama.cpp {tag} ({backend}) is already installed in {previous.folder}")
        return previous
    chosen = pick_assets(assets, tag, platform, backend)
    folder = os.path.join(root, f"{tag}-{backend}")
    downloads = os.path.join(root, "downloads")
    staging = folder + ".new"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    for a in chosen:
        log(f"downloading {a.name} ({a.size / 1e6:.0f} MB)")
        archive = hub.download(a.url, os.path.join(downloads, a.name), a.size, a.sha256, opener=opener)
        _extract(archive, staging)
    exe_dir = _find_server(staging)
    if exe_dir is None:
        raise OSError(f"no llama-server in {', '.join(a.name for a in chosen)}")
    if len(chosen) > 1 and exe_dir != staging:
        # The CUDA runtime DLLs sit at the top of their archive; llama.cpp looks next to itself.
        for n in os.listdir(staging):
            if n.lower().endswith(".dll") and not os.path.exists(os.path.join(exe_dir, n)):
                shutil.move(os.path.join(staging, n), exe_dir)
    if not _windows():
        for n in os.listdir(exe_dir):
            p = os.path.join(exe_dir, n)
            if os.path.isfile(p) and n.startswith("llama-") and "." not in n:
                os.chmod(p, 0o755)
    shutil.rmtree(folder, ignore_errors=True)
    if os.path.exists(folder):
        raise OSError(f"cannot replace {folder}; is a llama.cpp from there still running?")
    os.replace(staging, folder)
    inst = Install(tag, backend, os.path.normpath(os.path.join(folder, os.path.relpath(exe_dir, staging))))

    code, devices, text = check(inst, run)
    if code != 0 and platform == "win32" and _crashed(code):
        sys_version = system_runtime_version()
        shown = ".".join(map(str, sys_version)) if sys_version else "unknown"
        log(f"llama.cpp crashed at start (0x{code & 0xFFFFFFFF:08X}); this PC's Visual C++ runtime is {shown}")
        source = add_local_runtime(inst.folder)
        if source:
            log(f"copied a newer Visual C++ runtime from {source} next to llama.cpp")
            code, devices, text = check(inst, run)
    if code != 0:
        hint = ""
        if platform == "win32" and _crashed(code):
            hint = (f"\nInstall the latest Microsoft Visual C++ Redistributable ({VC_REDIST_URL}) "
                    "and run setup again.")
        raise OSError(f"llama.cpp does not start (exit code {code}):\n{text.strip()[-2000:]}{hint}")
    if backend != "cpu" and not devices:
        raise OSError(f"llama.cpp ({backend}) starts but sees no GPU:\n{text.strip()[-2000:]}\n"
                      "Try `setup --backend vulkan`, or `--backend cpu` to run without a GPU.")
    inst.devices = devices
    with open(os.path.join(root, "installed.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": inst.tag, "backend": inst.backend, "folder": inst.folder, "devices": devices}, f, indent=2)
    for a in chosen:  # unpacked and working: the archives are no longer needed
        with contextlib.suppress(OSError):
            os.remove(os.path.join(downloads, a.name))
    return inst
