"""Download catalog models from Hugging Face (``gpuensemble-plan pull``).

A catalog entry names a repository (``hf_repo``) and a quantization tag
(``hf_quant``, e.g. ``Q4_K_M`` or ``mxfp4``). The repository's file list
comes from the Hugging Face API, so file names and split parts
(``-00001-of-00003.gguf``) do not have to be written into the catalog.
Downloads resume after an interruption (also by themselves, a few times) and
are checked against the size and SHA-256 that Hugging Face publishes.
"""

from __future__ import annotations

import errno
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

HF = "https://huggingface.co"
CHUNK = 8 * 1024 * 1024
RETRIES = 5
_SPLIT = re.compile(r"-\d{5}-of-(\d{5})\.gguf$", re.IGNORECASE)
# Small models that ride along with a big one for faster guessing (multi-token prediction, draft models).
_SIDE_FILE = re.compile(r"(^|[-_.])(mtp|dflash|draft)([-_.]|$)", re.IGNORECASE)


@dataclass
class RemoteFile:
    path: str  # path inside the repository, "/"-separated
    size: int
    sha256: str | None = None


def home_dir() -> str:
    """Where GPUEnsemble keeps models and programs (GPUENSEMBLE_HOME overrides)."""
    if os.environ.get("GPUENSEMBLE_HOME"):
        return os.environ["GPUENSEMBLE_HOME"]
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return os.path.join(os.environ["LOCALAPPDATA"], "GPUEnsemble")
    return os.path.join(os.path.expanduser("~"), ".cache", "gpuensemble")


def models_dir() -> str:
    """Where models are downloaded: GPUENSEMBLE_MODELS (the Windows app's chosen folder), else ``home/models``."""
    if os.environ.get("GPUENSEMBLE_MODELS"):
        return os.environ["GPUENSEMBLE_MODELS"]
    return os.path.join(home_dir(), "models")


def _request(url: str, token: str | None = None, headers: dict[str, str] | None = None) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": "gpuensemble", **(headers or {})})
    if token:
        # Not sent on to the storage host that Hugging Face redirects downloads to.
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def list_repo(repo: str, token: str | None = None, opener: Callable = urllib.request.urlopen) -> list[RemoteFile]:
    """Every file in a model repository's main branch, with size and SHA-256."""
    url: str | None = f"{HF}/api/models/{repo}/tree/main?recursive=true"
    files: list[RemoteFile] = []
    while url:
        try:
            with opener(_request(url, token), timeout=60) as r:
                entries = json.load(r)
                link = r.headers.get("Link", "") if getattr(r, "headers", None) else ""
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise LookupError(f"{repo}: not found on Hugging Face, or it needs a login "
                                  f"(set HF_TOKEN) (HTTP {e.code})") from e
            raise
        for e in entries:
            if e.get("type") != "file":
                continue
            lfs = e.get("lfs") or {}
            files.append(RemoteFile(e["path"], int(lfs.get("size", e.get("size", 0))), lfs.get("oid")))
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return files


def _set_key(path: str) -> str:
    """A file's path with any split suffix removed, so the parts of one model share a key."""
    return _SPLIT.sub(".gguf", path)


def select_gguf(files: list[RemoteFile], quant: str) -> list[RemoteFile]:
    """The GGUF file(s) of one quantization, all split parts in order.

    The tag must appear as a whole word in the file's path (``Q4_K_M`` does
    not match ``IQ4_K_M``; a folder named ``Q4_K_M/`` counts). Among several
    matches the file nearest the top and then the shortest name wins, so
    ``Qwen3-14B-Q4_K_M.gguf`` beats ``Qwen3-14B-UD-Q4_K_M.gguf``. Projector
    files (``mmproj``) and the small models some repositories add for faster
    guessing (``mtp-``, ``dflash-``, ``draft``) are skipped.
    """
    tag = re.compile(rf"(?<![A-Za-z0-9]){re.escape(quant)}(?![A-Za-z0-9])", re.IGNORECASE)
    sets: dict[str, list[RemoteFile]] = {}
    for f in files:
        name = f.path.rsplit("/", 1)[-1].lower()
        if not name.endswith(".gguf") or "mmproj" in name or _SIDE_FILE.search(name):
            continue
        if not tag.search(f.path[: -len(".gguf")]):
            continue
        sets.setdefault(_set_key(f.path), []).append(f)
    if not sets:
        raise LookupError(f"no {quant} .gguf file found")
    key = min(sets, key=lambda k: (k.count("/"), len(k), k))
    parts = sorted(sets[key], key=lambda f: f.path)
    m = _SPLIT.search(parts[0].path)
    if m and len(parts) != int(m.group(1)):
        raise LookupError(f"{key}: found {len(parts)} of {int(m.group(1))} parts")
    return parts


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _progress(done: int, total: int, started: float, name: str, say: Callable[[str], None] = _say) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0
    say(f"  {name}: {done / 1024**3:6.2f} / {total / 1024**3:.2f} GiB  "
        f"{rate / 1e6:6.1f} MB/s  about {eta / 60:.0f} min left")


def _hash_file(path: str, h) -> None:
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    _hash_file(path, h)
    return h.hexdigest()


def _fetch(url: str, part: str, size: int, token: str | None, opener: Callable, state: list,
           report_every: float, name: str, say: Callable[[str], None] = _say) -> None:
    """Append the rest of ``url`` to ``part``. ``state[0]`` is a SHA-256 of what ``part``
    holds; it is kept in step with the file even when this fails half way."""
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with opener(_request(url, token, headers), timeout=60) as r:
        if have and getattr(r, "status", 200) != 206:
            have = 0  # the server ignored the range: start over
            state[0] = hashlib.sha256()
        h = state[0]
        started = last = time.monotonic()
        start_have = have
        with open(part, "ab" if have else "wb") as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
                h.update(chunk)
                have += len(chunk)
                now = time.monotonic()
                if now - last >= report_every:
                    _progress(have - start_have, size - start_have, started, name, say)
                    last = now


def _retryable(e: Exception) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code >= 500 or e.code == 429
    if getattr(e, "errno", None) == errno.ENOSPC:
        return False
    return isinstance(e, (urllib.error.URLError, OSError, http.client.HTTPException))


def download(url: str, dest: str, size: int, sha256: str | None = None, token: str | None = None,
             opener: Callable = urllib.request.urlopen, report_every: float = 10.0,
             retries: int = RETRIES, sleep: Callable[[float], None] = time.sleep,
             say: Callable[[str], None] = _say) -> str:
    """Download ``url`` to ``dest`` through ``dest.part``, resuming what an earlier attempt left.

    The SHA-256 is computed while the file arrives (on a resume, the part
    already on disk is read once first). A connection that drops is retried
    ``retries`` times in a row without progress before giving up. Progress
    goes to ``say`` (stderr unless given); an exception it raises that is not
    an OSError ends the download, and what has arrived stays for a resume.
    """
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    name = os.path.basename(dest)
    if os.path.exists(part) and os.path.getsize(part) > size:
        os.remove(part)
    state = [hashlib.sha256()]
    if os.path.exists(part):
        say(f"  {name}: resuming after {os.path.getsize(part) / 1024**3:.2f} GiB")
        if sha256:
            _hash_file(part, state[0])
    failures = 0
    while (have := os.path.getsize(part) if os.path.exists(part) else 0) < size or not os.path.exists(part):
        try:
            _fetch(url, part, size, token, opener, state, report_every, name, say)
        except Exception as e:  # noqa: BLE001 - sorted by _retryable
            if not _retryable(e):
                raise
            now_have = os.path.getsize(part) if os.path.exists(part) else 0
            failures = 0 if now_have > have else failures + 1
            if failures > retries:
                raise OSError(f"{name}: download failed ({e}); run pull again to resume") from e
            say(f"  {name}: connection lost ({e}); retrying")
            sleep(min(2 ** failures, 30))
            continue
        if os.path.getsize(part) == have:
            failures += 1  # the server sent nothing new
            if failures > retries:
                break
    got_size = os.path.getsize(part)
    if got_size != size:
        if got_size > size:
            os.remove(part)
        raise OSError(f"{name}: got {got_size} bytes, expected {size}; run pull again")
    got = state[0].hexdigest()
    if sha256 and got != sha256:
        os.remove(part)
        raise OSError(f"{name}: SHA-256 mismatch ({got} != {sha256}); "
                      "the download was removed, run pull again")
    if sha256:
        say(f"  {name}: SHA-256 matches")
    os.replace(part, dest)
    return dest


def local_path(entry: dict, base: str | None = None) -> str:
    """Folder a catalog model is stored in."""
    return os.path.join(base or models_dir(), entry["hf_repo"].replace("/", "__"))


def source_url(entry: dict, path: str, base: str | None = None) -> str | None:
    """Where a file of a pulled catalog model is on Hugging Face; None if it is not in that model's folder."""
    if "hf_repo" not in entry:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(local_path(entry, base)))
    except ValueError:
        return None  # Windows: on another drive
    if rel.startswith("..") or os.path.isabs(rel):
        return None
    return f"{HF}/{entry['hf_repo']}/resolve/main/{urllib.parse.quote(rel.replace(os.sep, '/'))}"


def _local_files(folder: str) -> list[RemoteFile]:
    out = []
    for root, _, names in os.walk(folder):
        for n in names:
            full = os.path.join(root, n)
            out.append(RemoteFile(os.path.relpath(full, folder).replace(os.sep, "/"), os.path.getsize(full)))
    return out


def find_local(entry: dict, base: str | None = None) -> str | None:
    """The first .gguf part of a model that was pulled completely, if any."""
    if "hf_repo" not in entry or "hf_quant" not in entry:
        return None
    folder = local_path(entry, base)
    if not os.path.isdir(folder):
        return None
    try:
        parts = select_gguf(_local_files(folder), entry["hf_quant"])
    except LookupError:
        return None
    return os.path.join(folder, *parts[0].path.split("/"))


def pull(entry: dict, base: str | None = None, token: str | None = None,
         opener: Callable = urllib.request.urlopen, say: Callable[[str], None] = _say, **download_options) -> str:
    """Download a catalog model (every split part) and return the path of its first part."""
    if "hf_repo" not in entry or "hf_quant" not in entry:
        raise LookupError(f"{entry.get('name', entry.get('id'))} has no download source in the catalog")
    found = find_local(entry, base)
    if found:
        say(f"{entry['name']}: already downloaded")
        return found
    token = token or os.environ.get("HF_TOKEN") or None
    parts = select_gguf(list_repo(entry["hf_repo"], token, opener), entry["hf_quant"])
    folder = local_path(entry, base)
    os.makedirs(folder, exist_ok=True)
    dests = [os.path.join(folder, *p.path.split("/")) for p in parts]
    total = sum(p.size for p in parts)
    missing = sum(p.size - _have(d) for p, d in zip(parts, dests))
    free = shutil.disk_usage(folder).free
    if missing > free:
        raise OSError(f"{entry['name']} needs {missing / 1024**3:.1f} GiB more but {folder} has "
                      f"{free / 1024**3:.1f} GiB free (set GPUENSEMBLE_HOME or --dir to use another drive)")
    say(f"{entry['name']}: {len(parts)} file(s), {total / 1024**3:.1f} GiB from {HF}/{entry['hf_repo']}")
    for p, d in zip(parts, dests):
        url = f"{HF}/{entry['hf_repo']}/resolve/main/{urllib.parse.quote(p.path)}"
        download(url, d, p.size, p.sha256, token, opener, say=say, **download_options)
    return dests[0]


def _have(dest: str) -> int:
    """Bytes of ``dest`` already on disk, finished or partial."""
    for path in (dest, dest + ".part"):
        if os.path.exists(path):
            return os.path.getsize(path)
    return 0
