"""Profiles of catalog models that are not downloaded, read from the start of their files on Hugging Face.

A GGUF file starts with its header: the metadata and the table of tensors with
their sizes. That is everything the planner needs, and it is a few MB of a file
of many GB. So a catalog entry needs only ``hf_repo`` and ``hf_quant``: the
first time a model without a ``profile`` is planned, the header of each of its
files is read with HTTP range requests and the profile built from it is kept
in ``<home>/profiles/``, so it is read once per PC.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Callable

from .gguf import GGUFFile, read_header
from .hub import HF, _request, home_dir, list_repo, select_gguf
from .model import ModelProfile, profile_from_gguf

READ_AHEAD = 2 * 1024 * 1024  # bytes asked for at a time
MAX_HEADER = 64 * 1024 * 1024  # a header larger than this is not a GGUF file we can use


class RangeFile:
    """The start of a file on a web server, read forward with HTTP range requests."""

    def __init__(self, url: str, size: int, token: str | None = None,
                 opener: Callable = urllib.request.urlopen, read_ahead: int = READ_AHEAD):
        self.url, self.size, self.token, self.opener, self.read_ahead = url, size, token, opener, read_ahead
        self.pos = 0  # position of the next byte read()
        self.buf = b""
        self.buf_start = 0
        self.fetched = 0  # bytes downloaded in all

    def tell(self) -> int:
        return self.pos

    def _fill(self, need_end: int) -> None:
        start = self.buf_start + len(self.buf)
        end = min(self.size, max(need_end, start + self.read_ahead))
        if end > MAX_HEADER:
            raise OSError(f"{self.url}: the header is larger than {MAX_HEADER // 1024**2} MB")
        if end <= start:
            return
        headers = {"Range": f"bytes={start}-{end - 1}"}
        with self.opener(_request(self.url, self.token, headers), timeout=60) as r:
            data = r.read()
            status = getattr(r, "status", 206)
        if status == 200:
            # The server ignored the range and sent the whole file from the start.
            data = data[start:end]
        if not data:
            raise OSError(f"{self.url}: no data at byte {start}")
        self.fetched += len(data)
        # Keep only what is still ahead of the read position.
        keep = self.buf[self.pos - self.buf_start:] if self.pos < start else b""
        self.buf_start = start - len(keep)
        self.buf = keep + data

    def read(self, n: int) -> bytes:
        while self.buf_start + len(self.buf) < min(self.size, self.pos + n):
            before = self.buf_start + len(self.buf)
            self._fill(self.pos + n)
            if self.buf_start + len(self.buf) == before:
                break
        i = self.pos - self.buf_start
        out = self.buf[i:i + n]
        self.pos += len(out)
        return out


def read_remote(entry: dict, token: str | None = None, opener: Callable = urllib.request.urlopen) -> GGUFFile:
    """The header(s) of a catalog model's file(s) on Hugging Face, as ``read_model`` gives them for a local file."""
    token = token or os.environ.get("HF_TOKEN") or None
    parts = select_gguf(list_repo(entry["hf_repo"], token, opener), entry["hf_quant"])
    first: GGUFFile | None = None
    for index, p in enumerate(parts):
        url = f"{HF}/{entry['hf_repo']}/resolve/main/{urllib.parse.quote(p.path)}"
        g = read_header(RangeFile(url, p.size, token, opener), p.path, p.size)
        if first is None:
            first = g
            first.parts = [p.path for p in parts]
        else:
            for t in g.tensors:
                t.part = index
            first.tensors.extend(g.tensors)
    assert first is not None
    return first


# Raised whenever profile_from_gguf reads a header differently, so kept profiles are read again.
PROFILE_VERSION = 4


def cache_path(entry: dict, base: str | None = None) -> str:
    name = f"{entry['hf_repo'].replace('/', '__')}__{entry['hf_quant']}.v{PROFILE_VERSION}.json"
    return os.path.join(base or os.path.join(home_dir(), "profiles"), name)


def profile_of(entry: dict, fetch: bool = True, base: str | None = None, token: str | None = None,
               opener: Callable = urllib.request.urlopen) -> ModelProfile:
    """A catalog model's profile: written in the catalog, else kept from an earlier read, else read now.

    Raises LookupError (or OSError) when it has none and it cannot be read now.
    """
    if entry.get("profile"):
        p = entry["profile"]
        return ModelProfile.from_uniform(p) if "uniform" in p else ModelProfile.from_dict(p)
    if "hf_repo" not in entry or "hf_quant" not in entry:
        raise LookupError(f"{entry.get('id')}: no profile and no download source")
    path = cache_path(entry, base)
    if os.path.exists(path):
        try:
            return ModelProfile.from_json_file(path)
        except (OSError, ValueError, KeyError, TypeError):
            pass  # a broken copy: read it again
    if not fetch:
        raise LookupError(f"{entry.get('id')}: its profile has not been read yet")
    profile = profile_from_gguf(read_remote(entry, token, opener))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(profile.to_json())
    os.replace(tmp, path)
    return profile


def fill_profiles(entries: list[dict], fetch: bool = True, base: str | None = None,
                  say: Callable[[str], None] | None = None, **kw) -> list[dict]:
    """Put a profile into every entry that has none; leave out the ones whose profile cannot be had."""
    out = []
    for e in entries:
        if e.get("profile"):
            out.append(e)
            continue
        try:
            prof = profile_of(e, fetch=fetch, base=base, **kw)
        except Exception as err:  # one model's odd file must not hide the others
            if say:
                say(f"{e.get('name', e.get('id'))}: skipped ({err})")
            continue
        out.append({**e, "profile": json.loads(prof.to_json())})
    return out
