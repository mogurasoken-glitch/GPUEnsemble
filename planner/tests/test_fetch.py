import io
import json
import os
import re
import socket
import struct
import tempfile
import threading
import types
import unittest
import urllib.error
from unittest import mock

from gpuensemble_planner import cli, fetch, hub, worker
from gpuensemble_planner.devices import GIB, Device, load_gpus
from gpuensemble_planner.fetch import FetchServer, Piece, ask_to_fetch, fetch_pieces, parse_request, stage_tensors
from gpuensemble_planner.fnv import FileHasher, fnv1a64, name_of
from gpuensemble_planner.gguf import read_model
from gpuensemble_planner.llamabin import Install

ALIGN = 32
URL = "https://huggingface.co/org/model-GGUF/resolve/main/"


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: str, tensors: list[tuple[str, bytes]], ggml_type: int = 24, shape=None) -> None:
    """A GGUF v3 file whose tensors hold the given bytes (I8: one byte per element)."""
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    header += _str("general.architecture") + struct.pack("<I", 8) + _str("llama")
    offset = 0
    for name, data in tensors:
        dims = shape or (len(data),)
        header += _str(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        header += struct.pack("<IQ", ggml_type, offset)
        offset += (len(data) + ALIGN - 1) // ALIGN * ALIGN
    with open(path, "wb") as f:
        f.write(header + b"\0" * ((-len(header)) % ALIGN))
        for i, (_, data) in enumerate(tensors):
            f.write(data)
            if i < len(tensors) - 1:
                f.write(b"\0" * ((-len(data)) % ALIGN))


def blob(seed: int, size: int) -> bytes:
    return bytes((seed * 31 + i * 7) % 251 for i in range(size))


class Response:
    def __init__(self, data: bytes, status: int, headers: dict[str, str] | None = None) -> None:
        self.data, self.status, self.headers = io.BytesIO(data), status, headers or {}

    def read(self, n: int = -1) -> bytes:
        return self.data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


class Site:
    """Stands in for Hugging Face: serves byte ranges of ``files`` (url -> bytes)."""

    def __init__(self, files: dict[str, bytes], status: int = 206, cut_first: int | None = None,
                 error: Exception | None = None, says_size: bool = True) -> None:
        self.files, self.status, self.cut_first, self.error = files, status, cut_first, error
        self.says_size = says_size
        self.asked: list[tuple[str, int, int]] = []
        self.lock = threading.Lock()

    def __call__(self, request, timeout=None):
        if self.error is not None:
            raise self.error
        first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        with self.lock:
            self.asked.append((request.full_url, first, last))
            cut, self.cut_first = self.cut_first, None
        whole = self.files[request.full_url]
        data = whole[first:last + 1]
        headers = {"Content-Range": f"bytes {first}-{last}/{len(whole)}"} if self.says_size else {}
        return Response(data[:cut] if cut else data, self.status, headers)


def python_hasher(folder: str) -> FileHasher:
    return FileHasher(folder, native=False)


class GgufSizesTest(unittest.TestCase):
    def test_exact_sizes_and_places(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            first = os.path.join(d, "m-00001-of-00002.gguf")
            second = os.path.join(d, "m-00002-of-00002.gguf")
            a, b, c = blob(1, 100), blob(2, 33), blob(3, 64)
            write_gguf(first, [("blk.0.a", a), ("blk.0.b", b)])
            write_gguf(second, [("blk.1.a", c)])
            g = read_model(first)
            self.assertEqual(g.parts, [first, second])
            by_name = {t.name: t for t in g.tensors}
            self.assertEqual([(t.name, t.part, t.size) for t in g.tensors],
                             [("blk.0.a", 0, 100), ("blk.0.b", 0, 33), ("blk.1.a", 1, 64)])
            self.assertEqual(by_name["blk.0.a"].nbytes, 128)  # the gap to the next tensor, with its padding
            for t, data in ((by_name["blk.0.a"], a), (by_name["blk.0.b"], b), (by_name["blk.1.a"], c)):
                with open(g.parts[t.part], "rb") as f:
                    f.seek(t.start)
                    self.assertEqual(f.read(t.size), data)

    def test_types(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.gguf")
            # MXFP4: 17 bytes for every 32 weights. 64 x 3 weights are 6 blocks.
            write_gguf(path, [("blk.0.x", blob(1, 102)), ("blk.0.y", blob(2, 102))], ggml_type=39, shape=(64, 3))
            self.assertEqual([t.size for t in read_model(path).tensors], [102, 102])
            # A type this reader does not know, or a size that does not fit the gap: no exact size.
            write_gguf(path, [("blk.0.x", blob(1, 102)), ("blk.0.y", blob(2, 102))], ggml_type=99, shape=(64, 3))
            self.assertEqual([t.size for t in read_model(path).tensors], [None, None])
            write_gguf(path, [("blk.0.x", blob(1, 300)), ("blk.0.y", blob(2, 300))], ggml_type=39, shape=(64, 3))
            self.assertEqual([t.size for t in read_model(path).tensors], [None, None])
            write_gguf(path, [("blk.0.x", blob(1, 102))], ggml_type=39, shape=(65, 3))  # not whole blocks
            self.assertEqual([t.size for t in read_model(path).tensors], [None])


class StageTensorsTest(unittest.TestCase):
    def test_what_goes_to_a_stage(self) -> None:
        with tempfile.TemporaryDirectory() as d, mock.patch.object(fetch, "HASH_THRESHOLD", 100):
            path = os.path.join(d, "m.gguf")
            tensors = [("token_embd.weight", blob(0, 400))]
            for i in range(4):
                tensors += [(f"blk.{i}.attn_q.weight", blob(i, 50)),  # too small to be hashed
                            (f"blk.{i}.attn_v.weight", blob(i, 101)),
                            (f"blk.{i}.ffn_gate_exps.weight", blob(i, 300))]
            tensors += [("output_norm.weight", blob(9, 20)), ("output.weight", blob(9, 400))]
            write_gguf(path, tensors)
            g = read_model(path)
            names = lambda **kw: [t.name for t in stage_tensors(g, **kw)]  # noqa: E731
            self.assertEqual(names(first=1, last=2, cpu_experts=[], head=False),
                             ["blk.1.attn_v.weight", "blk.1.ffn_gate_exps.weight",
                              "blk.2.attn_v.weight", "blk.2.ffn_gate_exps.weight"])
            # Experts the plan leaves in RAM are not sent to the card.
            self.assertEqual(names(first=1, last=2, cpu_experts=[1, 3], head=False),
                             ["blk.1.attn_v.weight", "blk.2.attn_v.weight", "blk.2.ffn_gate_exps.weight"])
            # The last stage's card also gets the output head.
            self.assertEqual(names(first=3, last=3, cpu_experts=[], head=True),
                             ["blk.3.attn_v.weight", "blk.3.ffn_gate_exps.weight", "output.weight"])


class FetchPiecesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = os.path.join(tmp.name, "rpc-cache", "rpc")
        self.file = blob(5, 5000)
        self.site = Site({URL + "m.gguf": self.file})
        self.pieces = [Piece(URL + "m.gguf", 100, 1000), Piece(URL + "m.gguf", 1200, 700),
                       Piece(URL + "m.gguf", 2000, 1000)]

    def fetch(self, pieces=None, **kw):
        kw.setdefault("opener", self.site)
        kw.setdefault("hasher", python_hasher)
        return fetch_pieces(pieces or self.pieces, self.dir, **kw)

    def names(self) -> list[str]:
        return sorted(n for n in os.listdir(self.dir) if n != fetch.INDEX)

    def test_pieces_land_under_their_hash(self) -> None:
        seen = []
        result = self.fetch(progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(result, {"fetched": 3, "cached": 0, "bytes": 2700})
        expected = sorted(name_of(fnv1a64(self.file[p.offset:p.offset + p.size])) for p in self.pieces)
        self.assertEqual(self.names(), expected)  # what llama.cpp's RPC server looks for
        for p in self.pieces:
            with open(os.path.join(self.dir, name_of(fnv1a64(self.file[p.offset:p.offset + p.size]))), "rb") as f:
                self.assertEqual(f.read(), self.file[p.offset:p.offset + p.size])
        self.assertEqual(sorted(self.site.asked), [(URL + "m.gguf", 100, 1099), (URL + "m.gguf", 1200, 1899),
                                                   (URL + "m.gguf", 2000, 2999)])
        self.assertEqual(seen[0], (0, 2700))
        self.assertEqual(seen[-1], (2700, 2700))

    def test_nothing_is_fetched_twice(self) -> None:
        self.fetch()
        self.site.asked.clear()
        self.assertEqual(self.fetch(), {"fetched": 0, "cached": 3, "bytes": 0})
        self.assertEqual(self.site.asked, [])
        # One more piece: only that one is downloaded.
        more = self.pieces + [Piece(URL + "m.gguf", 3000, 512)]
        self.assertEqual(self.fetch(more), {"fetched": 1, "cached": 3, "bytes": 512})
        self.assertEqual(self.site.asked, [(URL + "m.gguf", 3000, 3511)])

    def sent_before(self, piece: Piece, data: bytes | None = None) -> str:
        """Put a piece into the cache as llama.cpp does when the main PC sends it: under its hash, unknown to the index."""
        data = self.file[piece.offset:piece.offset + piece.size] if data is None else data
        os.makedirs(self.dir, exist_ok=True)
        name = name_of(fnv1a64(data))
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(data)
        return name

    def test_what_the_main_pc_sent_before_is_not_fetched(self) -> None:
        names = sorted(self.sent_before(p) for p in self.pieces)
        self.assertEqual(self.fetch(), {"fetched": 0, "cached": 3, "bytes": 0})
        self.assertEqual(self.names(), names)
        # Only the pieces' samples were asked for (here the pieces are smaller than a sample: all of each).
        self.assertEqual(sorted(self.site.asked), [(URL + "m.gguf", 100, 1099), (URL + "m.gguf", 1200, 1899),
                                                   (URL + "m.gguf", 2000, 2999)])
        # They are in the index now: the next time nothing is asked.
        self.site.asked.clear()
        self.assertEqual(self.fetch(), {"fetched": 0, "cached": 3, "bytes": 0})
        self.assertEqual(self.site.asked, [])

    def test_a_file_of_the_same_size_is_not_taken_for_the_piece(self) -> None:
        # The layers of a model often have tensors of one size. The cache holds two of the layers the worker had
        # before (one it downloaded, one the main PC sent); now it is asked for other layers, of the same size.
        self.fetch(self.pieces[:1])
        self.sent_before(self.pieces[2])
        before = self.names()
        others = [Piece(URL + "m.gguf", 3000, 1000), Piece(URL + "m.gguf", 4000, 1000)]
        self.assertEqual(self.fetch(others), {"fetched": 2, "cached": 0, "bytes": 2000})
        expected = [name_of(fnv1a64(self.file[p.offset:p.offset + p.size])) for p in others]
        self.assertEqual(self.names(), sorted(before + expected))
        # And of the first layers again, the one the main PC sent is found by its bytes.
        self.site.asked.clear()
        self.assertEqual(self.fetch(self.pieces[:1] + self.pieces[2:]), {"fetched": 0, "cached": 2, "bytes": 0})
        self.assertEqual(self.site.asked, [(URL + "m.gguf", 2000, 2999)])

    def test_large_pieces_are_told_by_their_ends(self) -> None:
        with mock.patch.object(fetch, "SAMPLE", 100):
            same_ends = self.file[100:200] + bytes(800) + self.file[1000:1100]  # only the middle differs: unlikely
            self.sent_before(self.pieces[0], same_ends)
            self.sent_before(self.pieces[2])
            self.sent_before(self.pieces[1], bytes(700))  # of its size, but something else
            self.assertEqual(self.fetch(), {"fetched": 1, "cached": 2, "bytes": 700})
            self.assertEqual(sorted(self.site.asked), [
                (URL + "m.gguf", 100, 199), (URL + "m.gguf", 1000, 1099),    # the ends of the first piece
                (URL + "m.gguf", 1200, 1299), (URL + "m.gguf", 1200, 1899),  # the second: its ends, then all of it
                (URL + "m.gguf", 1800, 1899),
                (URL + "m.gguf", 2000, 2099), (URL + "m.gguf", 2900, 2999)])
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".part")], [])

    def test_a_dropped_connection_is_resumed(self) -> None:
        self.site.cut_first = 300
        self.assertEqual(self.fetch(self.pieces[:1], parallel=1)["fetched"], 1)
        self.assertEqual(self.site.asked, [(URL + "m.gguf", 100, 1099), (URL + "m.gguf", 400, 1099)])
        self.assertEqual(self.names(), [name_of(fnv1a64(self.file[100:1100]))])

    def test_a_slow_connection_is_replaced(self) -> None:
        # The first connection trickles: 100 bytes a read, 20 s a read. After 30 s it has brought 200 bytes,
        # under the floor, so a new one asks for the rest; that one is fast and brings it all.
        time_now = [0.0]
        first = [True]

        def opener(request, timeout=None):
            r = self.site(request, timeout)
            if first[0]:
                first[0] = False
                read = r.read

                def slow(n=-1):
                    time_now[0] += 20.0
                    return read(min(n, 100))
                r.read = slow
            return r

        dest = os.path.join(self.dir, "x.part")
        os.makedirs(self.dir)
        piece = self.pieces[0]
        with mock.patch.object(fetch, "SLOW_RATE", 10):
            fetch._download(piece, dest, opener, None, threading.Event(), lambda n: None,
                            sleep=lambda s: self.fail("a slow connection is not a failure"), clock=lambda: time_now[0])
        self.assertEqual(self.site.asked, [(URL + "m.gguf", 100, 1099), (URL + "m.gguf", 300, 1099)])
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), self.file[100:1100])

    def test_a_steady_connection_is_kept(self) -> None:
        time_now = [0.0]

        def opener(request, timeout=None):
            r = self.site(request, timeout)
            read = r.read

            def steady(n=-1):
                time_now[0] += 20.0
                return read(min(n, 100))  # 5 bytes a second: over a floor of 4
            r.read = steady
            return r

        dest = os.path.join(self.dir, "x.part")
        os.makedirs(self.dir)
        with mock.patch.object(fetch, "SLOW_RATE", 4):
            fetch._download(self.pieces[0], dest, opener, None, threading.Event(), lambda n: None,
                            clock=lambda: time_now[0])
        self.assertEqual(len(self.site.asked), 1)

    def test_failures(self) -> None:
        with self.assertRaisesRegex(Exception, "did not send the range"):
            self.fetch(opener=Site({URL + "m.gguf": self.file}, status=200))  # would be the whole file
        with self.assertRaises(urllib.error.HTTPError):
            self.fetch(opener=Site({}, error=urllib.error.HTTPError(URL, 404, "Not Found", None, None)))
        with mock.patch.object(fetch.shutil, "disk_usage", return_value=types.SimpleNamespace(free=1000)), \
                self.assertRaisesRegex(OSError, "GiB free"):
            self.fetch()
        self.assertEqual(self.names(), [])  # nothing half done is left behind

    def test_a_file_of_another_size_is_not_downloaded(self) -> None:
        # The main PC's file is not the one at the address (another repository's build of the same model, say):
        # its tensors are elsewhere in it, and every byte downloaded would be for nothing.
        with self.assertRaisesRegex(fetch.FetchError, "m.gguf on Hugging Face has 5000 bytes and the main PC's has "
                                                      "4000: they are not the same file"):
            self.fetch(sizes={URL + "m.gguf": 4000})
        self.assertEqual(self.site.asked, [(URL + "m.gguf", 0, 0)])  # asked its size, and nothing more
        self.assertEqual(self.names(), [])

    def test_a_file_of_the_same_size_is_downloaded(self) -> None:
        self.assertEqual(self.fetch(sizes={URL + "m.gguf": 5000})["fetched"], 3)
        self.assertEqual(self.site.asked[0], (URL + "m.gguf", 0, 0))
        self.assertEqual(len(self.site.asked), 4)
        # Asked again with everything here: the size is not asked for.
        self.site.asked.clear()
        self.assertEqual(self.fetch(sizes={URL + "m.gguf": 4000})["cached"], 3)
        self.assertEqual(self.site.asked, [])

    def test_a_size_that_cannot_be_told_does_not_stop_the_download(self) -> None:
        quiet = Site({URL + "m.gguf": self.file}, says_size=False)
        self.assertEqual(self.fetch(opener=quiet, sizes={URL + "m.gguf": 4000})["fetched"], 3)

        def flaky(request, timeout=None):  # the first request, for the size, fails in a way that passes
            if not flaky.failed:
                flaky.failed = True
                raise urllib.error.HTTPError(URL, 503, "Service Unavailable", None, None)
            return self.site(request, timeout)

        flaky.failed = False
        self.assertEqual(self.fetch(self.pieces + [Piece(URL + "m.gguf", 3000, 512)], opener=flaky,
                                    sizes={URL + "m.gguf": 4000})["fetched"], 1)
        with self.assertRaises(urllib.error.HTTPError):  # but a file that is not there is not tried
            self.fetch([Piece(URL + "m.gguf", 4000, 100)], sizes={URL + "m.gguf": 5000},
                       opener=Site({}, error=urllib.error.HTTPError(URL, 404, "Not Found", None, None)))

    def test_stopped(self) -> None:
        stop = threading.Event()
        stop.set()
        with self.assertRaises(fetch.Stopped):
            self.fetch(stop=stop)
        self.assertEqual(self.names(), [])

    def test_leftovers_of_a_stopped_download_are_removed(self) -> None:
        os.makedirs(self.dir)
        with open(os.path.join(self.dir, "fetch-0.part"), "wb") as f:
            f.write(b"old bytes of something else")
        self.fetch(self.pieces[:1])
        self.assertEqual(self.names(), [name_of(fnv1a64(self.file[100:1100]))])


class RequestTest(unittest.TestCase):
    def request(self, **changes) -> bytes:
        request = {"gpuensemble": "fetch", "v": 1, "urls": [URL + "a.gguf", URL + "b.gguf"],
                   "pieces": [[0, 10, 20], [1, 0, 5]]}
        request.update(changes)
        return json.dumps(request).encode() + b"\n"

    def test_pieces(self) -> None:
        pieces = [Piece(URL + "a.gguf", 10, 20), Piece(URL + "b.gguf", 0, 5)]
        self.assertEqual(parse_request(self.request()), (pieces, {}))  # an older main PC gives no sizes
        self.assertEqual(parse_request(self.request(sizes=[30, 5])), (pieces, {URL + "a.gguf": 30, URL + "b.gguf": 5}))
        # A size that is not known, or a list that does not fit the addresses: those files are not checked.
        self.assertEqual(parse_request(self.request(sizes=[None, 5])), (pieces, {URL + "b.gguf": 5}))
        for sizes in ([30], "x", [True, 5.5], [-1, 0]):
            self.assertEqual(parse_request(self.request(sizes=sizes)), (pieces, {}), sizes)

    def test_refused(self) -> None:
        bad = [b"", b"not json\n", b"[]\n", self.request(v=2), self.request(gpuensemble="find"),
               self.request(urls="x"), self.request(pieces=[[2, 0, 5]]), self.request(pieces=[[0, -1, 5]]),
               self.request(pieces=[[0, 0, 0]]), self.request(pieces=[[0, 0]]), self.request(pieces=[[0, 0, 1.5]]),
               self.request(pieces=[[True, 0, 5]]), self.request(sizes=[29, 5]),  # past the end of its file
               # Only Hugging Face: a worker is not a way to make another PC fetch any address.
               self.request(urls=["http://huggingface.co/x", URL]), self.request(urls=["https://example.com/x", URL]),
               self.request(urls=["https://huggingface.co.example.com/x", URL]),
               self.request(urls=["file:///c:/windows/x", URL]), self.request(urls=[URL + "a\nb", URL])]
        for line in bad:
            with self.assertRaises(ValueError, msg=line):
                parse_request(line)


class ServerTest(unittest.TestCase):
    def serve(self, fetch_, allowed=lambda peer: True):
        logs: list[str] = []
        server = FetchServer(("127.0.0.1", 0), "/cache", allowed, logs.append, fetch=fetch_).start()
        self.addCleanup(server.close)
        return server, logs

    def test_asked_and_answered(self) -> None:
        got = {}

        def fetch_(pieces, folder, progress, stop, sizes):
            got.update(pieces=pieces, folder=folder, sizes=sizes)
            progress(0, 30)
            progress(30, 30)
            return {"fetched": 2, "cached": 1, "bytes": 30}

        server, logs = self.serve(fetch_)
        pieces = [Piece(URL + "a.gguf", 10, 20), Piece(URL + "b.gguf", 0, 5), Piece(URL + "a.gguf", 40, 5)]
        seen = []
        sizes = {URL + "a.gguf": 100, URL + "b.gguf": 5}
        result = ask_to_fetch("127.0.0.1", server.address[1], pieces, progress=lambda d, t: seen.append((d, t)),
                              sizes=sizes)
        self.assertEqual((result["fetched"], result["cached"], result["bytes"]), (2, 1, 30))
        self.assertEqual(got, {"pieces": pieces, "folder": "/cache", "sizes": sizes})
        self.assertEqual(seen, [(0, 30), (30, 30)])
        self.assertTrue(any("asks this PC to download its share of the weights itself: 3 tensors" in line
                            for line in logs), logs)
        self.assertTrue(any("downloaded 2 tensors" in line and "(1 were here already)" in line for line in logs), logs)

    def test_a_failure_is_told(self) -> None:
        def fails(pieces, folder, progress, stop, sizes):
            raise OSError("12.0 GiB are needed but Z: has 3.0 GiB free")

        server, logs = self.serve(fails)
        with self.assertRaisesRegex(OSError, "3.0 GiB free"):
            ask_to_fetch("127.0.0.1", server.address[1], [Piece(URL + "a.gguf", 0, 5)])
        self.assertTrue(any("sends them instead" in line for line in logs), logs)

    def test_only_pcs_let_in(self) -> None:
        called = []
        server, _ = self.serve(lambda *a, **kw: called.append(1), allowed=lambda peer: False)
        with self.assertRaises(OSError):
            ask_to_fetch("127.0.0.1", server.address[1], [Piece(URL + "a.gguf", 0, 5)])
        self.assertEqual(called, [])

    def test_a_bad_request_is_refused(self) -> None:
        called = []
        server, _ = self.serve(lambda *a, **kw: called.append(1))
        with self.assertRaisesRegex(OSError, "only files on https://huggingface.co/"):
            ask_to_fetch("127.0.0.1", server.address[1], [Piece("https://example.com/x.gguf", 0, 5)])
        self.assertEqual(called, [])

    def test_no_progress_is_given_up(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)

        def hangs(pieces, folder, progress, stop, sizes):
            progress(0, 30)
            release.wait(10)
            return {"fetched": 0, "cached": 0, "bytes": 0}

        server, _ = self.serve(hangs)
        with self.assertRaisesRegex(OSError, "no progress"):
            ask_to_fetch("127.0.0.1", server.address[1], [Piece(URL + "a.gguf", 0, 5)], stall=0.3)

    def test_nobody_there(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with self.assertRaises(OSError):
            ask_to_fetch("127.0.0.1", port, [Piece(URL + "a.gguf", 0, 5)])


class MainPcTest(unittest.TestCase):
    """Which workers `up` asks, and for which bytes."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = tmp.name
        self.entry = {"id": "m", "hf_repo": "org/model-GGUF", "hf_quant": "q"}
        folder = hub.local_path(self.entry, self.base)
        os.makedirs(folder)
        self.path = os.path.join(folder, "m.gguf")
        tensors = [("token_embd.weight", blob(0, 400))]
        for i in range(4):
            tensors += [(f"blk.{i}.attn_v.weight", blob(i, 101)), (f"blk.{i}.ffn_gate_exps.weight", blob(i, 300))]
        tensors += [("output.weight", blob(9, 400))]
        write_gguf(self.path, tensors)
        patch = mock.patch.object(fetch, "HASH_THRESHOLD", 100)
        patch.start()
        self.addCleanup(patch.stop)
        self.local = Device("ti", int(11 * GIB), 463e9, llama_device="CUDA0")
        self.far = Device("4090", int(24 * GIB), 372e9, llama_device="CUDA0", rpc="100.64.0.2:50052")
        self.far.rpc_cache = self.far.rpc_fetch = True
        self.far.rpc_rtt = 0.007
        stage = lambda device, first, last: types.SimpleNamespace(  # noqa: E731
            device=device, first_layer=first, last_layer=last, n_layers=last - first + 1)
        self.plan = types.SimpleNamespace(stages=[stage("ti", 0, 1), stage("4090", 2, 3)])
        self.asked: list[tuple] = []
        self.logs: list[str] = []

    def ask(self, address, port, pieces, progress, sizes=None):
        self.asked.append((address, port, pieces))
        self.sizes = sizes
        progress(10, 20)
        return {"fetched": len(pieces), "cached": 0, "bytes": sum(p.size for p in pieces), "done": True}

    def run_it(self, mode="auto", cpu_experts=(), ask=None, path=None, every=0.0) -> None:
        cli.fetch_on_workers(self.plan, path or self.path, self.entry, self.base, [self.local, self.far],
                             list(cpu_experts), mode, self.logs.append, ask=ask or self.ask, every=every)

    def test_a_far_worker_is_asked_for_its_stage(self) -> None:
        self.run_it(cpu_experts=[2])
        [(address, port, pieces)] = self.asked
        self.assertEqual((address, port), ("100.64.0.2", 50053))
        g = read_model(self.path)
        want = [t for t in g.tensors if t.name in ("blk.2.attn_v.weight", "blk.3.attn_v.weight",
                                                    "blk.3.ffn_gate_exps.weight", "output.weight")]
        url = "https://huggingface.co/org/model-GGUF/resolve/main/m.gguf"
        self.assertEqual(pieces, [Piece(url, t.start, t.size) for t in want])
        self.assertEqual(self.sizes, {url: os.path.getsize(self.path)})
        self.assertIn("the PC at 100.64.0.2 downloads its share of the weights itself", self.logs[0])
        self.assertIn("has them now", self.logs[-1])

    def test_a_worker_that_has_them_says_so_briefly(self) -> None:
        def has_them(address, port, pieces, progress, sizes=None):
            progress(0, 0)
            return {"fetched": 0, "cached": len(pieces), "bytes": 0, "done": True}

        self.run_it(ask=has_them)
        self.assertEqual(len(self.logs), 1, self.logs)
        self.assertIn("the PC at 100.64.0.2 has its share of the weights already (5 tensors", self.logs[0])

    def test_a_worker_on_the_lan_is_sent_them(self) -> None:
        self.far.rpc_rtt = 0.0004
        self.run_it()
        self.assertEqual(self.asked, [])
        self.run_it(mode="fetch")  # unless asked to
        self.assertEqual(len(self.asked), 1)

    def test_who_is_not_asked(self) -> None:
        self.run_it(mode="send")
        self.far.rpc_fetch = False  # an older worker
        self.run_it()
        self.far.rpc_fetch, self.far.rpc_cache = True, False  # it would not keep them
        self.run_it()
        self.far.rpc_cache, self.far.rpc_rtt = True, None  # not measured: given by hand in a file
        self.run_it()
        self.far.rpc_rtt = 0.007
        elsewhere = os.path.join(self.base, "copy.gguf")  # not where pull put it: its source is not known
        os.replace(self.path, elsewhere)
        self.run_it(path=elsewhere)
        self.assertEqual(self.asked, [])
        self.assertEqual(self.logs, [])

    def test_a_worker_that_cannot_is_sent_them(self) -> None:
        def cannot(address, port, pieces, progress, sizes=None):
            raise OSError("3.0 GiB free")

        self.run_it(ask=cannot)
        self.assertTrue(any("could not download its share of the weights itself (3.0 GiB free); this PC sends them"
                            in line for line in self.logs), self.logs)

    def test_source_url(self) -> None:
        self.assertEqual(hub.source_url(self.entry, self.path, self.base),
                         "https://huggingface.co/org/model-GGUF/resolve/main/m.gguf")
        sub = os.path.join(os.path.dirname(self.path), "MXFP4", "a b.gguf")
        self.assertEqual(hub.source_url(self.entry, sub, self.base),
                         "https://huggingface.co/org/model-GGUF/resolve/main/MXFP4/a%20b.gguf")
        self.assertIsNone(hub.source_url(self.entry, os.path.join(self.base, "x.gguf"), self.base))
        self.assertIsNone(hub.source_url({"id": "m"}, self.path, self.base))


class FakeInstall:
    def __init__(self, folder: str) -> None:
        self.folder = folder

    def tool(self, name: str) -> str:
        return Install("b1", "cuda", self.folder).tool(name)


class WorkerTest(unittest.TestCase):
    def test_a_caching_worker_can_be_asked(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        open(os.path.join(tmp.name, "ggml-rpc-server.exe"), "w").close()
        port = worker.free_port()
        seen = {}

        def ask(argv, log):
            # The fetch server answers on the discovery port's number; a bad request is refused, not ignored.
            try:
                ask_to_fetch("127.0.0.1", port + 1, [Piece("https://example.com/x", 0, 5)])
            except OSError as e:
                seen["answer"] = str(e)
            return 0

        detected = [Device("GPU0 NVIDIA GeForce RTX 4090", int(24 * GIB), 372e9, llama_device="CUDA0")]
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": tmp.name}):
            os.environ.pop("LLAMA_CACHE", None)
            self.assertEqual(worker.rpc_cache_dir(), os.path.join(tmp.name, "rpc-cache", "rpc"))
            with mock.patch.object(worker.fnv, "fast", return_value=True):
                worker.start("127.0.0.1", port, None, True, detect=lambda: detected,
                             install=lambda: FakeInstall(tmp.name), log=lambda s: None, out=out, run=ask)
            self.assertIn("only files on", seen["answer"])
            [entry] = json.loads(out.getvalue())["gpus"]
            self.assertTrue(entry["fetch"])
            # Where the hash runs in Python, downloading would take longer than being sent the weights.
            slow = io.StringIO()
            seen.clear()
            with mock.patch.object(worker.fnv, "fast", return_value=False):
                worker.start("127.0.0.1", port, None, True, detect=lambda: detected,
                             install=lambda: FakeInstall(tmp.name), log=lambda s: None, out=slow, run=ask)
            self.assertNotIn("fetch", json.loads(slow.getvalue())["gpus"][0])
            self.assertNotIn("only files on", seen["answer"])
        self.assertTrue(load_gpus([dict(entry, rtt_ms=7.5)])[0].rpc_fetch)
        self.assertAlmostEqual(load_gpus([dict(entry, rtt_ms=7.5)])[0].rpc_rtt, 0.0075)
        # Without --cache the server keeps nothing, so there is nothing to fetch into.
        [plain] = worker.rig_entries(detected, "10.0.0.2:50052")
        self.assertNotIn("fetch", plain)


if __name__ == "__main__":
    unittest.main()
