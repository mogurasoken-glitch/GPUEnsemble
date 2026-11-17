import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
import urllib.error

from gpuensemble_planner import hub
from gpuensemble_planner.cli import main
from gpuensemble_planner.hub import RemoteFile, download, find_local, list_repo, pull, select_gguf
from gpuensemble_planner.recommend import load_catalog


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200, headers: dict | None = None, fail_after: int | None = None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}
        self.fail_after = fail_after  # raise a dropped connection after this many bytes

    def read(self, n: int = -1) -> bytes:
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise ConnectionResetError("connection reset by peer")
        if self.fail_after is not None and n > 0:
            n = min(n, self.fail_after - self.tell())
        return super().read(n)


class FakeHub:
    """Serves a fake repository tree and file downloads; records each request."""

    def __init__(self, files: dict[str, bytes], page: int = 1000, honour_range: bool = True):
        self.files = files
        self.page = page
        self.honour_range = honour_range
        self.requests: list[tuple[str, dict]] = []
        self.drop_next_at: int | None = None
        self.fail_with: Exception | None = None

    def __call__(self, req, timeout=None):
        url = req.full_url
        headers = {k.lower(): v for k, v in req.header_items()}
        self.requests.append((url, headers))
        if self.fail_with is not None:
            e, self.fail_with = self.fail_with, None
            raise e
        if "/api/models/" in url:
            start = int(url.split("cursor=")[1]) if "cursor=" in url else 0
            names = sorted(self.files)
            entries = [{"type": "directory", "path": "sub"}] if start == 0 else []
            for n in names[start:start + self.page]:
                data = self.files[n]
                entries.append({"type": "file", "path": n, "size": 134,
                                "lfs": {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}})
            link = {}
            if start + self.page < len(names):
                link = {"Link": f'<{url.split("&cursor")[0]}&cursor={start + self.page}>; rel="next"'}
            return FakeResponse(json.dumps(entries).encode(), headers=link)
        path = url.split("/resolve/main/")[1].replace("%2F", "/")
        data = self.files[path]
        status = 200
        if "range" in headers and self.honour_range:
            data = data[int(headers["range"][len("bytes="):-1]):]
            status = 206
        drop, self.drop_next_at = self.drop_next_at, None
        return FakeResponse(data, status=status, fail_after=drop)


def blob(n: int, seed: int = 0) -> bytes:
    return bytes((i * 7 + seed) % 251 for i in range(n))


def remote(path: str) -> RemoteFile:
    return RemoteFile(path, 1)


class SelectTest(unittest.TestCase):
    def test_picks_the_plain_file_over_variants(self) -> None:
        files = [remote(p) for p in ("Qwen3-14B-IQ4_K_M.gguf", "Qwen3-14B-UD-Q4_K_M.gguf",
                                     "Qwen3-14B-Q4_K_M.gguf", "Qwen3-14B-Q4_K_S.gguf", "README.md")]
        self.assertEqual([f.path for f in select_gguf(files, "Q4_K_M")], ["Qwen3-14B-Q4_K_M.gguf"])

    def test_split_parts_come_in_order(self) -> None:
        files = [remote(f"gpt-oss-120b-mxfp4-0000{i}-of-00003.gguf") for i in (3, 1, 2)]
        files.append(remote("gpt-oss-120b-F16.gguf"))
        parts = select_gguf(files, "mxfp4")
        self.assertEqual([p.path[-19:] for p in parts], [f"0000{i}-of-00003.gguf" for i in (1, 2, 3)])

    def test_missing_part_is_an_error(self) -> None:
        files = [remote("m-Q4_K_M-00001-of-00002.gguf")]
        with self.assertRaisesRegex(LookupError, "1 of 2"):
            select_gguf(files, "Q4_K_M")

    def test_quant_folder_and_projector(self) -> None:
        files = [remote("Q4_K_M/Big-00001-of-00002.gguf"), remote("Q4_K_M/Big-00002-of-00002.gguf"),
                 remote("mmproj-Q4_K_M.gguf"), remote("Q8_0/Big-00001-of-00001.gguf")]
        self.assertEqual(len(select_gguf(files, "Q4_K_M")), 2)
        with self.assertRaises(LookupError):
            select_gguf(files, "Q5_K_M")

    def test_draft_and_mtp_files_are_skipped(self) -> None:
        # Some repositories put a small model for faster guessing next to the big one, at the top.
        files = [remote("UD-Q4_K_XL/Big-UD-Q4_K_XL-00001-of-00002.gguf"),
                 remote("UD-Q4_K_XL/Big-UD-Q4_K_XL-00002-of-00002.gguf"),
                 remote("mtp-Big-UD-Q4_K_XL.gguf"), remote("dflash-Big-UD-Q4_K_XL.gguf"),
                 remote("Big-draft-UD-Q4_K_XL.gguf")]
        self.assertEqual([f.path for f in select_gguf(files, "UD-Q4_K_XL")],
                         ["UD-Q4_K_XL/Big-UD-Q4_K_XL-00001-of-00002.gguf",
                          "UD-Q4_K_XL/Big-UD-Q4_K_XL-00002-of-00002.gguf"])


class DownloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = os.path.join(self.tmp.name, "a", "m.gguf")
        self.data = blob(3 * hub.CHUNK + 123)
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.hub = FakeHub({"m.gguf": self.data})
        self.url = f"{hub.HF}/r/resolve/main/m.gguf"

    def get(self, **kw) -> str:
        return download(self.url, self.dest, len(self.data), self.sha, opener=self.hub, sleep=lambda s: None, **kw)

    def test_whole_file(self) -> None:
        self.assertEqual(self.get(), self.dest)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.data)
        self.assertFalse(os.path.exists(self.dest + ".part"))
        self.get()  # already there: no new request
        self.assertEqual(len(self.hub.requests), 1)

    def test_dropped_connection_resumes_with_a_range(self) -> None:
        self.hub.drop_next_at = hub.CHUNK + 5
        with contextlib.redirect_stderr(io.StringIO()):
            self.get()
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.data)
        self.assertEqual(self.hub.requests[1][1]["range"], f"bytes={hub.CHUNK + 5}-")

    def test_resumes_a_part_left_by_an_earlier_run(self) -> None:
        os.makedirs(os.path.dirname(self.dest))
        with open(self.dest + ".part", "wb") as f:
            f.write(self.data[:1000])
        with contextlib.redirect_stderr(io.StringIO()):
            self.get()
        self.assertEqual(self.hub.requests[0][1]["range"], "bytes=1000-")
        self.assertEqual(hub.sha256_of(self.dest), self.sha)

    def test_server_that_ignores_the_range_starts_over(self) -> None:
        self.hub.honour_range = False
        os.makedirs(os.path.dirname(self.dest))
        with open(self.dest + ".part", "wb") as f:
            f.write(b"x" * 1000)  # not even the right bytes
        with contextlib.redirect_stderr(io.StringIO()):
            self.get()
        self.assertEqual(hub.sha256_of(self.dest), self.sha)

    def test_corrupt_download_is_removed(self) -> None:
        os.makedirs(os.path.dirname(self.dest))
        with open(self.dest + ".part", "wb") as f:
            f.write(b"x" * 1000)  # wrong bytes, then a resume
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(OSError, "SHA-256"):
            self.get()
        self.assertFalse(os.path.exists(self.dest + ".part"))
        self.assertFalse(os.path.exists(self.dest))

    def test_gives_up_after_repeated_failures(self) -> None:
        def broken(req, timeout=None):
            raise urllib.error.URLError("no route to host")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(OSError, "run pull again"):
            download(self.url, self.dest, len(self.data), self.sha, opener=broken, sleep=lambda s: None, retries=2)

    def test_client_errors_are_not_retried(self) -> None:
        self.hub.fail_with = urllib.error.HTTPError(self.url, 404, "Not Found", {}, None)
        with self.assertRaises(urllib.error.HTTPError):
            self.get()
        self.assertEqual(len(self.hub.requests), 1)


class PullTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.files = {
            "README.md": b"hello",
            "Big-mxfp4-00001-of-00002.gguf": blob(5000, 1),
            "Big-mxfp4-00002-of-00002.gguf": blob(3000, 2),
            "Big-F16.gguf": blob(9000, 3),
        }
        self.entry = {"id": "big", "name": "Big", "hf_repo": "org/Big-GGUF", "hf_quant": "mxfp4", "file_gib": 0.0}

    def test_lists_every_page(self) -> None:
        fake = FakeHub(self.files, page=1)
        listed = list_repo("org/Big-GGUF", opener=fake)
        self.assertEqual(sorted(f.path for f in listed), sorted(self.files))
        self.assertEqual(len(fake.requests), len(self.files))
        self.assertEqual({f.path: f.size for f in listed}["Big-F16.gguf"], 9000)

    def test_pull_downloads_every_part_once(self) -> None:
        fake = FakeHub(self.files)
        with contextlib.redirect_stderr(io.StringIO()):
            first = pull(self.entry, base=self.tmp.name, opener=fake)
        folder = os.path.join(self.tmp.name, "org__Big-GGUF")
        self.assertEqual(first, os.path.join(folder, "Big-mxfp4-00001-of-00002.gguf"))
        self.assertEqual(sorted(os.listdir(folder)), ["Big-mxfp4-00001-of-00002.gguf", "Big-mxfp4-00002-of-00002.gguf"])
        self.assertEqual(find_local(self.entry, self.tmp.name), first)
        n = len(fake.requests)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(pull(self.entry, base=self.tmp.name, opener=fake), first)
        self.assertEqual(len(fake.requests), n)  # nothing fetched, not even the file list

    def test_half_pulled_model_is_not_found_locally(self) -> None:
        folder = os.path.join(self.tmp.name, "org__Big-GGUF")
        os.makedirs(folder)
        with open(os.path.join(folder, "Big-mxfp4-00001-of-00002.gguf"), "wb") as f:
            f.write(self.files["Big-mxfp4-00001-of-00002.gguf"])
        with open(os.path.join(folder, "Big-mxfp4-00002-of-00002.gguf.part"), "wb") as f:
            f.write(b"12")
        self.assertIsNone(find_local(self.entry, self.tmp.name))

    def test_token_is_sent_to_hugging_face_only(self) -> None:
        fake = FakeHub(self.files)
        with contextlib.redirect_stderr(io.StringIO()):
            pull(self.entry, base=self.tmp.name, token="hf_secret", opener=fake)
        self.assertTrue(all(h.get("authorization") == "Bearer hf_secret" for _, h in fake.requests))
        req = hub._request("https://huggingface.co/x", "hf_secret")
        self.assertNotIn("Authorization", req.headers)  # unredirected: not copied on a redirect
        self.assertEqual(req.unredirected_hdrs["Authorization"], "Bearer hf_secret")

    def test_unknown_repository(self) -> None:
        fake = FakeHub(self.files)
        fake.fail_with = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        with self.assertRaisesRegex(LookupError, "HF_TOKEN"):
            pull(self.entry, base=self.tmp.name, opener=fake)


class CatalogSourceTest(unittest.TestCase):
    def test_every_catalog_model_can_be_pulled(self) -> None:
        for e in load_catalog():
            self.assertRegex(e["hf_repo"], r"^[\w.-]+/[\w.-]+$", e["id"])
            self.assertTrue(e["hf_quant"], e["id"])

    def test_cli_models_and_unknown_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["models", "--dir", tmp]), 0)
            self.assertIn("Qwen/Qwen3-32B-GGUF Q4_K_M", out.getvalue())
            with contextlib.redirect_stderr(err):
                self.assertEqual(main(["pull", "no-such-model", "--dir", tmp]), 1)
            self.assertIn("qwen3-32b", err.getvalue())


if __name__ == "__main__":
    unittest.main()
