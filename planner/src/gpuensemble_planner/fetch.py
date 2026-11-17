"""A worker at another site downloads its share of the weights itself.

The first time a model loads, llama.cpp sends the weights of the layers on
another PC over the network. Between two sites that is the upload of the
main PC's line: 21.6 GiB went at 46 MB/s in the first two-site test (8 of
the load's 13 minutes), and a home line with a slow upload takes hours.
The worker of that test then fetched the same 20.8 GiB itself in 133 s.

A worker started with ``--cache`` keeps each tensor it receives in a file
named by a hash of its bytes (see ``fnv``). So the main PC, which knows where
its model came from on Hugging Face and which tensors its plan puts on the
worker, can ask the worker to get exactly those bytes from there instead::

    {"gpuensemble": "fetch", "v": 1, "urls": ["https://huggingface.co/.../x.gguf"],
     "sizes": [63387346208],                       # of each file, as the main PC has it
     "pieces": [[0, 7040, 441460736], ...]}        # url index, offset, size

The worker downloads each piece with an HTTP range request, hashes it, and
puts it into the RPC server's cache under that name. It answers in lines of
JSON: ``{"progress": [done, total]}`` about once a second, then
``{"done": true, "fetched": n, "cached": m, "bytes": b}`` or ``{"error": "..."}``.
When llama.cpp then loads the model, the worker has every tensor already.

Nothing depends on this going right: a piece that is missing, or whose bytes
differ from the main PC's file, is simply sent by llama.cpp as before. But a
file that is not the one on Hugging Face (another repository's build of the
model, or one replaced there since) would be downloaded for nothing, so the
worker first compares the file's size there with the main PC's.

What the main PC sent on an earlier load is in the cache too, under hashes
that do not say which tensors they are. A file of a piece's size is not
enough to go by (the layers of a model have tensors of the same sizes: in
the three-card test the worker took layers 4-16, which it had, for layers
23-35, and the main PC sent 20 GiB after all), so the worker compares the
start and the end of each such piece with the files of its size.

The request comes over TCP on the discovery port's number (the worker's port
+ 1), from an address the worker lets in, and names only files on
huggingface.co.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from . import hub
from .fnv import FileHasher
from .gate import exclusive, reached_itself
from .gguf import GGUFFile

MAGIC = "gpuensemble"
VERSION = 1
# llama.cpp sends a hash first, and its server keeps the bytes, only for tensors above this size.
HASH_THRESHOLD = 10 * 1024 * 1024
SOURCE = hub.HF + "/"
MAX_REQUEST = 4 * 1024 * 1024  # bytes of JSON
MAX_PIECES = 20000
PARALLEL = 4
RETRIES = 5
CHUNK = 1024 * 1024
MARGIN = 1024 ** 3  # disk space left free
STALL = 180.0  # seconds without progress before the main PC stops waiting
# A connection that brings less than SLOW_RATE bytes a second over SLOW_WINDOW seconds is
# replaced by a new one, which resumes where it left off. Hugging Face's CDN now and then
# serves one range from a server that trickles: on the E5, gpt-oss-120b's last piece came at
# 100 KB/s for half an hour while new connections to the same file got 15-18 MB/s.
SLOW_WINDOW = 30.0
SLOW_RATE = 512 * 1024
INDEX = "fetched.json"
SAMPLE = 64 * 1024  # bytes at a piece's start and at its end that tell which cached file holds it
_NAME = re.compile(r"^[0-9a-f]{16}$")
_BLOCK = re.compile(r"^blk\.(\d+)\.")
_EXPERTS = re.compile(r"^blk\.\d+\.ffn_.*_exps\.")


@dataclass(frozen=True)
class Piece:
    url: str
    offset: int
    size: int

    @property
    def key(self) -> str:
        return f"{self.url}|{self.offset}|{self.size}"


class Stopped(Exception):
    """The main PC went away, or the worker is closing."""


class FetchError(Exception):
    """A download that trying again will not mend."""


class _Slow(Exception):
    """The connection brings too little: open a new one."""


# --- which bytes ---------------------------------------------------------------

def stage_tensors(g: GGUFFile, first: int, last: int, cpu_experts: list[int], head: bool) -> list:
    """The tensors of layers ``first``..``last`` that llama.cpp puts on a stage's card and would hash.

    Experts of the layers in ``cpu_experts`` stay in RAM (``-ot ...=CPU``);
    ``head`` adds the output head, which goes to the last stage's card.
    """
    found = []
    for t in g.tensors:
        if t.size is None or t.size <= HASH_THRESHOLD:
            continue
        m = _BLOCK.match(t.name)
        if m:
            layer = int(m.group(1))
            if first <= layer <= last and not (layer in cpu_experts and _EXPERTS.match(t.name)):
                found.append(t)
        elif head and t.name.startswith("output"):
            found.append(t)
    return found


# --- the worker's side ---------------------------------------------------------

def _load_index(folder: str) -> dict[str, str]:
    try:
        with open(os.path.join(folder, INDEX), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str) and _NAME.match(v)}


def _save_index(folder: str, index: dict[str, str]) -> None:
    tmp = os.path.join(folder, INDEX + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f)
    os.replace(tmp, os.path.join(folder, INDEX))


def missing_pieces(pieces: list[Piece], folder: str, index: dict[str, str]) -> list[Piece]:
    """The pieces the index does not place in the cache: those not fetched (or recognised) before."""
    missing = []
    for p in pieces:
        name = index.get(p.key)
        if name:
            try:
                if os.path.getsize(os.path.join(folder, name)) == p.size:
                    continue
            except OSError:
                pass
        missing.append(p)
    return missing


def _unnamed(folder: str, index: dict[str, str]) -> dict[int, list[str]]:
    """The cached files the index does not name, by size: the main PC sent them on an earlier load."""
    named = set(index.values())
    found: dict[int, list[str]] = {}
    try:
        for entry in os.scandir(folder):
            if _NAME.match(entry.name) and entry.name not in named and entry.is_file():
                found.setdefault(entry.stat().st_size, []).append(entry.name)
    except OSError:
        pass
    return found


def _ends(size: int) -> list[tuple[int, int]]:
    """Where the samples of a piece of this size are: (offset in the piece, length)."""
    n = min(SAMPLE, size)
    return [(0, n)] if size <= 2 * SAMPLE else [(0, n), (size - n, n)]


def _download(piece: Piece, dest: str, opener: Callable, token: str | None, stop: threading.Event,
              on_bytes: Callable[[int], None], sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic) -> None:
    """Write the piece's bytes to ``dest``, resuming after a dropped or a slow connection."""
    failures = 0
    while True:
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have == piece.size:
            return
        if have > piece.size:
            os.remove(dest)
            have = 0
        if stop.is_set():
            raise Stopped()
        first, last = piece.offset + have, piece.offset + piece.size - 1
        try:
            with opener(hub._request(piece.url, token, {"Range": f"bytes={first}-{last}"}), timeout=60) as r:
                if getattr(r, "status", 206) != 206:
                    raise FetchError(f"the server did not send the range asked for (HTTP {getattr(r, 'status', '?')})")
                with open(dest, "ab" if have else "wb") as f:
                    left = piece.size - have
                    since, got = clock(), 0
                    while left > 0 and (chunk := r.read(min(CHUNK, left))):
                        if stop.is_set():
                            raise Stopped()
                        f.write(chunk)
                        on_bytes(len(chunk))
                        left -= len(chunk)
                        got += len(chunk)
                        now = clock()
                        if now - since >= SLOW_WINDOW:
                            if left > 0 and got < SLOW_RATE * (now - since):
                                raise _Slow()
                            since, got = now, 0
        except Stopped:
            raise
        except _Slow:
            continue  # it brought something, so this ends; a new connection is mostly far faster
        except Exception as e:  # noqa: BLE001 - sorted by hub._retryable
            if not hub._retryable(e):
                raise
            now = os.path.getsize(dest) if os.path.exists(dest) else 0
            failures = 0 if now > have else failures + 1
            if failures > RETRIES:
                raise OSError(f"download failed ({e})") from e
            sleep(min(2 ** failures, 30))
            continue
        if (os.path.getsize(dest) if os.path.exists(dest) else 0) == have:
            failures += 1  # the server sent nothing
            if failures > RETRIES:
                raise OSError(f"{piece.url} ended early")


def recognise(pieces: list[Piece], folder: str, index: dict[str, str], opener: Callable, token: str | None,
              stop: threading.Event, parallel: int = PARALLEL) -> list[Piece]:
    """Find the pieces that the main PC sent on an earlier load; returns those the cache really lacks.

    llama.cpp keeps what it is sent under the hash of the bytes, which does
    not say which tensor they are, and the tensors of a model's layers are
    often all of one size. So for each piece that has cached files of its
    size, its start and its end are downloaded (64 KiB each) and compared
    with those files. The ones found are added to the index.
    """
    unnamed = _unnamed(folder, index)
    maybe = [p for p in pieces if unnamed.get(p.size)]
    if not maybe:
        return pieces

    def sample(job: tuple[int, Piece]) -> list[bytes]:
        number, piece = job
        got = []
        for k, (start, length) in enumerate(_ends(piece.size)):
            part = os.path.join(folder, f"fetch-sample-{number}-{k}.part")
            try:
                _download(Piece(piece.url, piece.offset + start, length), part, opener, token, stop, lambda n: None)
                with open(part, "rb") as f:
                    got.append(f.read())
            finally:
                try:
                    os.remove(part)
                except OSError:
                    pass
        return got

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        samples = list(pool.map(sample, enumerate(maybe)))
    ends: dict[str, list[bytes] | None] = {}

    def ends_of(name: str, size: int) -> list[bytes] | None:
        if name not in ends:
            try:
                read = []
                with open(os.path.join(folder, name), "rb") as f:
                    for start, length in _ends(size):
                        f.seek(start)
                        read.append(f.read(length))
                ends[name] = read
            except OSError:
                ends[name] = None
        return ends[name]

    found = set()
    for piece, want in zip(maybe, samples):
        names = unnamed[piece.size]
        name = next((n for n in names if ends_of(n, piece.size) == want), None)
        if name is not None:
            names.remove(name)
            index[piece.key] = name
            found.add(piece.key)
    if found:
        _save_index(folder, index)
    return [p for p in pieces if p.key not in found]


_TOTAL = re.compile(r"bytes\s+\d+-\d+/(\d+)")


def remote_size(url: str, opener: Callable = urllib.request.urlopen, token: str | None = None) -> int | None:
    """The size of the file at ``url``, from the answer to a request for its first byte; None if it does not say."""
    with opener(hub._request(url, token, {"Range": "bytes=0-0"}), timeout=60) as r:
        headers = getattr(r, "headers", None)
        m = _TOTAL.match(str(headers.get("Content-Range") or "")) if headers is not None else None
        return int(m.group(1)) if m else None


def _check_sizes(urls: list[str], sizes: dict[str, int], opener: Callable, token: str | None) -> None:
    """FetchError if a file on Hugging Face is not as large as the main PC's copy of it."""
    for url in urls:
        if url not in sizes:
            continue
        try:
            there = remote_size(url, opener, token)
        except Exception as e:  # noqa: BLE001 - sorted by hub._retryable
            if not hub._retryable(e):
                raise
            continue  # the download itself tries again, and says what is wrong
        if there is not None and there != sizes[url]:
            raise FetchError(f"{url.rsplit('/', 1)[-1]} on Hugging Face has {there} bytes and the main PC's has "
                             f"{sizes[url]}: they are not the same file")


def fetch_pieces(pieces: list[Piece], folder: str, progress: Callable[[int, int], None] | None = None,
                 stop: threading.Event | None = None, opener: Callable = urllib.request.urlopen,
                 hasher: Callable[[str], FileHasher] = FileHasher, token: str | None = None,
                 parallel: int = PARALLEL, sizes: dict[str, int] | None = None) -> dict:
    """Download the pieces the cache lacks into ``folder``, each under its hash; returns what was done.

    ``sizes`` gives the size of each file as the main PC has it (by URL).
    """
    stop = stop or threading.Event()
    os.makedirs(folder, exist_ok=True)
    for entry in os.scandir(folder):
        if entry.name.startswith("fetch-") and entry.name.endswith(".part"):
            os.remove(entry.path)  # left by a download that was stopped
    index = _load_index(folder)
    token = token or os.environ.get("HF_TOKEN") or None
    todo = missing_pieces(pieces, folder, index)
    if sizes:
        _check_sizes(list(dict.fromkeys(p.url for p in todo)), sizes, opener, token)
    todo = recognise(todo, folder, index, opener, token, stop, parallel)
    total = sum(p.size for p in todo)
    free = shutil.disk_usage(folder).free
    if total + MARGIN > free:
        raise OSError(f"{total / 1024**3:.1f} GiB are needed but {folder} has {free / 1024**3:.1f} GiB free")
    done = 0
    lock = threading.Lock()
    hashing = threading.Lock()

    def on_bytes(n: int) -> None:
        nonlocal done
        with lock:
            done += n
            now = done
        if progress is not None:
            progress(now, total)

    with hasher(folder) as files:
        def one(job: tuple[int, Piece]) -> None:
            number, piece = job
            part = f"fetch-{number}.part"
            try:
                _download(piece, os.path.join(folder, part), opener, token, stop, on_bytes)
                with hashing:
                    name = files.hash(part)
                os.replace(os.path.join(folder, part), os.path.join(folder, name))
                with lock:
                    index[piece.key] = name
                    _save_index(folder, index)
                    now = done
                if progress is not None:
                    progress(now, total)  # hashing the last ones takes a while: the main PC hears of each
            except BaseException:
                stop.set()  # one failure ends the others
                try:
                    os.remove(os.path.join(folder, part))
                except OSError:
                    pass
                raise

        if progress is not None:
            progress(0, total)
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            errors = [f.exception() for f in [pool.submit(one, job) for job in enumerate(todo)]]
        errors = [e for e in errors if e is not None]
        if errors:
            # The first failure stops the others; say what failed, not that they were stopped.
            raise next((e for e in errors if not isinstance(e, Stopped)), errors[0])
    return {"fetched": len(todo), "cached": len(pieces) - len(todo), "bytes": total}


def parse_request(line: bytes) -> tuple[list[Piece], dict[str, int]]:
    """The pieces a request asks for, and the sizes it gives for their files; ValueError says what is wrong with it."""
    try:
        request = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ValueError("the request is not JSON")
    if not isinstance(request, dict) or request.get(MAGIC) != "fetch" or request.get("v") != VERSION:
        raise ValueError("not a fetch request of this version")
    urls, items = request.get("urls"), request.get("pieces")
    if not isinstance(urls, list) or not isinstance(items, list) or len(items) > MAX_PIECES:
        raise ValueError("the request needs urls and pieces")
    for url in urls:
        if not isinstance(url, str) or not url.startswith(SOURCE) or any(c in url for c in "\r\n \\"):
            raise ValueError(f"only files on {SOURCE} are downloaded")
    given = request.get("sizes")  # an older main PC sends none
    sizes = {}
    if isinstance(given, list) and len(given) == len(urls):
        sizes = {url: n for url, n in zip(urls, given) if isinstance(n, int) and not isinstance(n, bool) and n > 0}
    pieces = []
    for item in items:
        ok = isinstance(item, list) and len(item) == 3 and all(isinstance(n, int) and not isinstance(n, bool)
                                                               for n in item)
        if not ok or not 0 <= item[0] < len(urls) or item[1] < 0 or item[2] <= 0:
            raise ValueError("a piece must be [url index, offset, size]")
        piece = Piece(urls[item[0]], item[1], item[2])
        if piece.offset + piece.size > sizes.get(piece.url, piece.offset + piece.size):
            raise ValueError("a piece reaches past the end of its file")
        pieces.append(piece)
    return pieces, sizes


def request_kind(line: bytes) -> tuple[str, dict]:
    """What a request line asks for ("fetch", "serve", ...; "" if it is not one of ours), and the request."""
    try:
        request = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return "", {}
    if not isinstance(request, dict) or not isinstance(request.get(MAGIC), str):
        return "", {}
    return request[MAGIC], request


class FetchServer:
    """Takes the main PC's requests to fetch weights into the RPC server's cache ``folder``.

    The worker's other requests arrive on the same port: ``others`` maps
    their kind to ``handler(request, client, peer, say)``, which is called in
    the connection's thread and may keep the connection for as long as it
    likes (``serving`` does, while a model runs). With ``fetches`` False this
    PC takes only those, and answers a request to fetch with an error.
    """

    def __init__(self, listen: tuple[str, int], folder: str, is_allowed: Callable[[str], bool],
                 log: Callable[[str], None] = lambda s: None, fetch: Callable = fetch_pieces,
                 others: dict[str, Callable] | None = None, fetches: bool = True) -> None:
        family = socket.AF_INET6 if ":" in listen[0] else socket.AF_INET
        self.sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            exclusive(self.sock)
            self.sock.bind(listen)
            self.sock.listen(4)
        except OSError:
            self.sock.close()
            raise
        self.address = self.sock.getsockname()[:2]
        self.folder = folder
        self.is_allowed = is_allowed
        self.log = log
        self.fetch = fetch
        self.others = dict(others or {})
        self.fetches = fetches
        self.busy = threading.Lock()
        self.stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "FetchServer":
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()
        return self

    def _accept(self) -> None:
        while True:
            try:
                client, peer = self.sock.accept()
            except OSError:
                return  # closed
            if not self.is_allowed(peer[0]):
                client.close()
                continue
            threading.Thread(target=self._serve, args=(client, peer[0]), daemon=True).start()

    def _serve(self, client: socket.socket, peer: str) -> None:
        def say(message: dict) -> None:
            client.sendall(json.dumps(message).encode("utf-8") + b"\n")

        with client:
            try:
                client.settimeout(30)
                line = b""
                while not line.endswith(b"\n") and len(line) <= MAX_REQUEST:
                    data = client.recv(65536)
                    if not data:
                        break
                    line += data
                client.settimeout(60)
                kind, request = request_kind(line)
                if kind in self.others:
                    self.others[kind](request, client, peer, say)
                    return
                if not self.fetches:
                    say({"error": "this PC does not download weights itself"})
                    return
                try:
                    pieces, sizes = parse_request(line)
                except ValueError as e:
                    say({"error": str(e)})
                    return
                if not self.busy.acquire(blocking=False):
                    say({"error": "this PC is busy with another download"})
                    return
                try:
                    self._fetch_for(pieces, sizes, peer, say)
                finally:
                    self.busy.release()
            except OSError:
                pass  # the main PC went away

    def _fetch_for(self, pieces: list[Piece], sizes: dict[str, int], peer: str, say: Callable[[dict], None]) -> None:
        gone = threading.Event()
        last = [0.0]
        telling = threading.Lock()  # the downloads run side by side

        def progress(done: int, total: int) -> None:
            with telling:
                now = time.monotonic()
                if now - last[0] < 1.0 and done < total:
                    return
                last[0] = now
                try:
                    say({"progress": [done, total]})
                except OSError:
                    gone.set()  # the main PC went away: stop downloading for it

        class Either:
            def is_set(_self) -> bool:
                return gone.is_set() or self.stop.is_set()

            def set(_self) -> None:
                gone.set()

        started = time.monotonic()
        size = sum(p.size for p in pieces) / 1024 ** 3
        self.log(f"the PC at {peer} asks this PC to download its share of the weights itself: "
                 f"{len(pieces)} tensors, {size:.1f} GiB, from {SOURCE}")
        try:
            result = self.fetch(pieces, self.folder, progress=progress, stop=Either(), sizes=sizes)
        except Stopped:
            self.log(f"stopped downloading for the PC at {peer}")
            return
        except Exception as e:  # noqa: BLE001 - reported to the main PC, which then sends the weights itself
            self.log(f"could not download the weights ({e}); the PC at {peer} sends them instead")
            try:
                say({"error": str(e)})
            except OSError:
                pass
            return
        took = time.monotonic() - started
        if result["fetched"]:
            self.log(f"downloaded {result['fetched']} tensors, {result['bytes'] / 1024**3:.1f} GiB, in {took:.0f} s"
                     + (f" ({result['cached']} were here already)" if result["cached"] else ""))
        else:
            self.log("every one of them is here already")
        say(dict(result, done=True))

    def close(self) -> None:
        self.stop.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        if self._thread is not None:
            self._thread.join(timeout=5)


# --- the main PC's side ----------------------------------------------------------

def ask_to_fetch(address: str, port: int, pieces: list[Piece], progress: Callable[[int, int], None] | None = None,
                 stall: float = STALL, connect: Callable = socket.create_connection,
                 sizes: dict[str, int] | None = None) -> dict:
    """Ask the worker at ``address`` to fetch the pieces, and wait until it has; OSError if it cannot.

    ``progress(done, total)`` is called as the worker reports. A worker that
    reports nothing for ``stall`` seconds is given up. ``sizes`` gives the
    size of each file here (by URL), for the worker to compare with the
    file on Hugging Face before it downloads from it.
    """
    urls = list(dict.fromkeys(p.url for p in pieces))
    request = {MAGIC: "fetch", "v": VERSION, "urls": urls,
               "pieces": [[urls.index(p.url), p.offset, p.size] for p in pieces]}
    if sizes:
        request["sizes"] = [sizes.get(url) for url in urls]
    with connect((address, port), timeout=10) as s:
        if reached_itself(s):
            raise OSError("nothing listens there")
        s.sendall(json.dumps(request).encode("utf-8") + b"\n")
        s.settimeout(5)
        buffer = b""
        seen = None
        changed = time.monotonic()
        while True:
            if time.monotonic() - changed > stall:
                raise OSError(f"no progress for {stall:.0f} s")
            try:
                data = s.recv(65536)
            except socket.timeout:
                continue
            if not data:
                raise OSError("it closed the connection")
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    message = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    raise OSError("it answered something else")
                if not isinstance(message, dict):
                    raise OSError("it answered something else")
                if message.get("error"):
                    raise OSError(str(message["error"]))
                if message.get("done"):
                    return message
                changed = time.monotonic()  # it reports only as bytes arrive or a tensor is done
                state = message.get("progress")
                if state != seen:
                    seen = state
                    if progress is not None and isinstance(state, list) and len(state) == 2:
                        progress(int(state[0]), int(state[1]))
