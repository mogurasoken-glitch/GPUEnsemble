import os
import tempfile
import unittest

from gpuensemble_planner.gguf import read_model
from gpuensemble_planner.model import profile_from_gguf
from gpuensemble_planner.remote import RangeFile, cache_path, fill_profiles, profile_of, read_remote

from test_gguf import DENSE_META, write_gguf
from test_hub import FakeHub, FakeResponse


class RangeHub(FakeHub):
    """A FakeHub that sends exactly the bytes a range asks for, and counts them."""

    def __init__(self, files, **kw):
        super().__init__(files, **kw)
        self.sent = 0

    def __call__(self, req, timeout=None):
        headers = {k.lower(): v for k, v in req.header_items()}
        if "/resolve/main/" in req.full_url and "range" in headers:
            self.requests.append((req.full_url, headers))
            path = req.full_url.split("/resolve/main/")[1].replace("%2F", "/")
            start, end = (int(x) for x in headers["range"][len("bytes="):].split("-"))
            data = self.files[path][start:end + 1]
            self.sent += len(data)
            return FakeResponse(data, status=206)
        return super().__call__(req, timeout)


TENSORS = [("token_embd.weight", 4096), ("blk.0.attn_q.weight", 3000000), ("blk.1.attn_q.weight", 3000000),
           ("output.weight", 4096)]


class RemoteProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.local = os.path.join(self.tmp.name, "m.gguf")
        write_gguf(self.local, DENSE_META, TENSORS)
        with open(self.local, "rb") as f:
            self.data = f.read()
        self.hub = RangeHub({"tiny-Q4_K_M.gguf": self.data, "tiny-Q8_0.gguf": b"x"})
        self.entry = {"id": "tiny", "name": "tiny", "hf_repo": "org/tiny-GGUF", "hf_quant": "Q4_K_M"}
        self.cache = os.path.join(self.tmp.name, "profiles")

    def test_range_file_reads_like_a_file(self) -> None:
        f = RangeFile("https://huggingface.co/org/r/resolve/main/tiny-Q4_K_M.gguf", len(self.data),
                      opener=self.hub, read_ahead=100)
        self.assertEqual(f.read(4), b"GGUF")
        self.assertEqual(f.read(250), self.data[4:254])
        self.assertEqual(f.tell(), 254)
        self.assertEqual(f.read(10), self.data[254:264])

    def test_only_the_header_is_downloaded_and_the_profile_matches_the_local_one(self) -> None:
        g = read_remote(self.entry, opener=self.hub)
        self.assertEqual(profile_from_gguf(g).layers, profile_from_gguf(read_model(self.local)).layers)
        self.assertLess(self.hub.sent, 4 * 1024 * 1024)
        self.assertLess(self.hub.sent, len(self.data) // 2)

    def test_profile_is_read_once_and_kept(self) -> None:
        first = profile_of(self.entry, base=self.cache, opener=self.hub)
        self.assertTrue(os.path.exists(cache_path(self.entry, self.cache)))
        asked = len(self.hub.requests)
        again = profile_of(self.entry, base=self.cache, opener=self.hub)
        self.assertEqual(len(self.hub.requests), asked)
        self.assertEqual(first.layers, again.layers)
        # Without a network it still comes from the kept copy.
        self.assertEqual(profile_of(self.entry, fetch=False, base=self.cache).layers, first.layers)

    def test_entries_that_cannot_be_read_are_left_out(self) -> None:
        said = []
        broken = {"id": "gone", "name": "gone", "hf_repo": "org/tiny-GGUF", "hf_quant": "IQ1_S"}
        written = {"id": "w", "name": "w", "profile": {"name": "w", "uniform": {
            "n_layers": 1, "dense_gib_per_layer": 1, "kv_bytes_per_token_per_layer": 1}, "head_gib": 0, "embed_gib": 0}}
        out = fill_profiles([written, self.entry, broken], base=self.cache, say=said.append, opener=self.hub)
        self.assertEqual([e["id"] for e in out], ["w", "tiny"])
        self.assertIn("layers", out[1]["profile"])
        self.assertEqual(len(said), 1)
        self.assertIn("gone", said[0])

    def test_without_network_and_without_a_kept_copy_it_is_skipped(self) -> None:
        self.assertEqual(fill_profiles([self.entry], fetch=False, base=self.cache), [])


if __name__ == "__main__":
    unittest.main()
