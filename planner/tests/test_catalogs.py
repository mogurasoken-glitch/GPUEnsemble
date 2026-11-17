import contextlib
import datetime
import io
import json
import os
import tempfile
import unittest
import urllib.error
from unittest import mock

from gpuensemble_planner import catalogs, cli
from gpuensemble_planner.recommend import evaluate, load_catalog
from gpuensemble_planner.devices import load_rig

from test_gguf import DENSE_META, write_gguf
from test_hub import FakeResponse
from test_remote import RangeHub

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
TODAY = datetime.date(2026, 11, 16)


def model(repo, downloads=50000, created="2026-10-01", task="text-generation"):
    return {"id": repo, "downloads": downloads, "likes": 10, "createdAt": created + "T00:00:00.000Z",
            "pipeline_tag": task}


class SearchHub(RangeHub):
    """Hugging Face's model search, file lists and files, and the repository's list."""

    def __init__(self, files, searches, checked=None):
        super().__init__(files)
        self.searches = searches
        self.checked = checked

    def __call__(self, req, timeout=None):
        url = req.full_url
        if "raw.githubusercontent.com" in url:
            self.requests.append((url, {}))
            if self.checked is None:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return FakeResponse(json.dumps(self.checked).encode())
        if "/api/models?" in url:
            self.requests.append((url, {}))
            author = url.split("author=")[1].split("&")[0]
            return FakeResponse(json.dumps(self.searches.get(author, [])).encode())
        if "/api/models/" in url:
            repo = url.split("/api/models/")[1].split("/tree/")[0]
            self.requests.append((url, {}))
            entries = [{"type": "file", "path": p.split("/", 2)[2], "size": len(d)}
                       for p, d in self.files.items() if p.startswith(repo + "/")]
            return FakeResponse(json.dumps(entries).encode())
        return super().__call__(req, timeout)


class RepoFiles(dict):
    """Files keyed "org/repo/path"; the hub's download URL gives only the path, so look it up by suffix."""

    def __getitem__(self, path):
        for k, v in self.items():
            if k.endswith("/" + path):
                return v
        raise KeyError(path)


class CatalogsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = self.tmp.name
        p = os.path.join(self.home, "m.gguf")
        write_gguf(p, DENSE_META, [("token_embd.weight", 4096), ("blk.0.w.weight", 4096),
                                   ("blk.1.w.weight", 4096), ("output.weight", 4096)])
        with open(p, "rb") as f:
            self.gguf = f.read()

    def test_same_model_from_different_quantizers_has_one_key(self) -> None:
        keys = {catalogs.model_key(r) for r in ("unsloth/gemma-4-31B-it-GGUF", "bartowski/google_gemma-4-31B-it-GGUF",
                                                "ggml-org/gemma-4-31b-it-GGUF")}
        self.assertEqual(keys, {"gemma-4-31b"})
        self.assertNotEqual(catalogs.model_key("unsloth/gemma-4-12b-it-GGUF"), "gemma-4-31b")

    def test_the_bundled_list_has_a_date_and_is_used_until_a_newer_one_is_fetched(self) -> None:
        bundled = catalogs.checked_list(self.home)
        self.assertRegex(bundled["updated"], r"^\d{4}-\d{2}-\d{2}$")
        older = {"updated": "2000-01-01", "models": [{"id": "old", "rank": 1}]}
        hub = SearchHub({}, {}, checked=older)
        catalogs.fetch_checked(self.home, opener=hub)
        self.assertEqual(catalogs.checked_list(self.home)["models"], bundled["models"])
        newer = {"updated": "2099-01-01", "models": [{"id": "x", "name": "X", "rank": 1,
                                                     "hf_repo": "org/X-GGUF", "hf_quant": "Q4_K_M"}]}
        said = catalogs.fetch_checked(self.home, opener=SearchHub({}, {}, checked=newer))
        self.assertIn("1 checked models", said)
        self.assertEqual([e["id"] for e in catalogs.load_all(self.home)], ["x"])

    def test_a_repository_that_cannot_be_read_keeps_the_installed_list(self) -> None:
        said = catalogs.fetch_checked(self.home, opener=SearchHub({}, {}, checked=None))
        self.assertIn("HTTP 404", said)
        self.assertFalse(os.path.exists(catalogs.fetched_path(self.home)))
        bad = SearchHub({}, {}, checked={"models": [{"no id": 1}]})
        self.assertIn("not a model list", catalogs.fetch_checked(self.home, opener=bad))

    def test_new_models_are_found_and_listed_after_the_checked_ones(self) -> None:
        files = RepoFiles({
            "unsloth/NewModel-9B-GGUF/NewModel-9B-Q4_K_M.gguf": self.gguf,
            "unsloth/NewModel-9B-GGUF/NewModel-9B-Q8_0.gguf": b"x",
            "bartowski/maker_NewModel-9B-GGUF/maker_NewModel-9B-Q4_K_M.gguf": self.gguf,
            "unsloth/OnlyQ8-GGUF/OnlyQ8-Q8_0.gguf": b"x",
            "unsloth/Picture-Maker-GGUF/Picture-Maker-Q4_K_M.gguf": self.gguf,
            "unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf": self.gguf,
        })
        searches = {
            "unsloth": [model("unsloth/NewModel-9B-GGUF", 9000), model("unsloth/OnlyQ8-GGUF"),
                        model("unsloth/gemma-4-31B-it-GGUF"),  # already checked
                        model("unsloth/Old-GGUF", created="2024-01-01"),
                        model("unsloth/Few-GGUF", downloads=10),
                        model("unsloth/Some-Embedding-GGUF"),
                        model("unsloth/Picture-Maker-GGUF", task="text-to-image"),
                        # tagged as text models on Hugging Face, but they make pictures and videos
                        model("unsloth/flux-3-image-klein-pup-GGUF"), model("unsloth/SenseNova-6.8-Code2Video-GGUF")],
            "bartowski": [model("bartowski/maker_NewModel-9B-GGUF", 90000)],  # the same model, later quantizer
        }
        hub = SearchHub(files, searches)
        found, notes = catalogs.find_new(catalogs.checked_list(self.home)["models"], opener=hub, today=TODAY)
        self.assertEqual(notes, [])
        self.assertEqual([e["hf_repo"] for e in found], ["unsloth/NewModel-9B-GGUF"])
        self.assertEqual(found[0]["hf_quant"], "Q4_K_M")
        self.assertGreaterEqual(found[0]["rank"], catalogs.NEW_RANK)

        catalogs._write(catalogs.found_path(self.home), {"models": found})
        entries = catalogs.load_all(self.home)
        self.assertEqual(entries[-1]["hf_repo"], "unsloth/NewModel-9B-GGUF")
        self.assertTrue(entries[-1]["new"])
        self.assertFalse(any(e.get("new") for e in entries[:-1]))

    def test_update_keeps_only_chat_models_and_goes_on_past_a_file_it_cannot_read(self) -> None:
        image = os.path.join(self.home, "image.gguf")
        write_gguf(image, {"general.architecture": "flux"}, [("double_blocks.0.w", 4096), ("final.w", 4096)])
        odd = os.path.join(self.home, "odd.gguf")
        write_gguf(odd, dict(DENSE_META, **{"llama.block_count": [1, 2]}), [("blk.0.w", 4096)])
        with open(image, "rb") as f:
            image_bytes = f.read()
        with open(odd, "rb") as f:
            odd_bytes = f.read()
        files = RepoFiles({
            "unsloth/Chat-9B-GGUF/Chat-9B-Q4_K_M.gguf": self.gguf,
            "unsloth/Pixel-1-GGUF/Pixel-1-Q4_K_M.gguf": image_bytes,  # says nothing of what it does
            "unsloth/Odd-1-GGUF/Odd-1-Q4_K_M.gguf": odd_bytes,
        })
        newtype = os.path.join(self.home, "newtype.gguf")
        write_gguf(newtype, {k.replace("llama.", "brandnew."): v for k, v in DENSE_META.items()
                             if k != "general.architecture"} | {"general.architecture": "brandnew"},
                   [("blk.0.w", 4096), ("blk.1.w", 4096)])
        with open(newtype, "rb") as f:
            files["unsloth/Brand-New-7B-GGUF/Brand-New-7B-Q4_K_M.gguf"] = f.read()
        searches = {"unsloth": [model("unsloth/Chat-9B-GGUF", 5000), model("unsloth/Pixel-1-GGUF", 90000, task=None),
                                model("unsloth/Odd-1-GGUF", 80000), model("unsloth/Brand-New-7B-GGUF", 70000)]}
        checked = {"updated": "2099-01-01", "models": [
            {"id": "odd", "name": "Odd", "rank": 1, "hf_repo": "unsloth/Odd-1-GGUF", "hf_quant": "Q4_K_M"}]}
        said = []
        code = catalogs.update(self.home, opener=SearchHub(files, searches, checked=checked), say=said.append)
        self.assertEqual(code, 1)  # the checked model whose file could not be read
        self.assertTrue(any("pixel-1" in s and "left out" in s for s in said), said)
        self.assertTrue(any("brand-new-7b" in s and "not checked yet (brandnew)" in s for s in said), said)
        found = catalogs._read(catalogs.found_path(self.home))["models"]
        self.assertEqual([e["hf_repo"] for e in found], ["unsloth/Chat-9B-GGUF"])
        self.assertEqual(found[0]["rank"], catalogs.NEW_RANK)

    def test_update_reads_profiles_so_planning_needs_no_network(self) -> None:
        checked = {"updated": "2099-01-01", "models": [
            {"id": "a", "name": "A", "rank": 1, "hf_repo": "org/A-GGUF", "hf_quant": "Q4_K_M"},
            {"id": "gone", "name": "Gone", "rank": 2, "hf_repo": "org/Gone-GGUF", "hf_quant": "Q4_K_M"}]}
        hub = SearchHub(RepoFiles({"org/A-GGUF/A-Q4_K_M.gguf": self.gguf}), {}, checked=checked)
        said = []
        code = catalogs.update(self.home, opener=hub, discover=False, say=said.append)
        self.assertEqual(code, 1)  # one model could not be read
        self.assertTrue(any("gone" in s for s in said), said)
        self.assertEqual([s for s in said if s.startswith("read ")], ["read 0/2", "read 1/2", "read 2/2"])
        old = os.environ.get("GPUENSEMBLE_HOME")
        os.environ["GPUENSEMBLE_HOME"] = self.home
        try:
            devices, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
            cands = evaluate(load_catalog(), devices, host, 4096)
        finally:
            if old is None:
                del os.environ["GPUENSEMBLE_HOME"]
            else:
                os.environ["GPUENSEMBLE_HOME"] = old
        self.assertEqual([c.entry["id"] for c in cands], ["a"])
        self.assertIn("file_gib", cands[0].entry)


class GatedHub(SearchHub):
    """Lists the files but asks for a login to download them, as a gated repository does."""

    def __call__(self, req, timeout=None):
        if "/resolve/main/" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
        return super().__call__(req, timeout)


class MineTest(unittest.TestCase):
    """Models the user adds to this PC's own list (catalog --add)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = self.tmp.name
        p = os.path.join(self.home, "m.gguf")
        write_gguf(p, DENSE_META, [("token_embd.weight", 4096), ("blk.0.w.weight", 4096),
                                   ("blk.1.w.weight", 4096), ("output.weight", 4096)])
        with open(p, "rb") as f:
            self.gguf = f.read()
        self.files = RepoFiles({
            "someone/Chat-9B-Uncensored-GGUF/Chat-9B-Uncensored-Q4_K_M.gguf": self.gguf,
            "someone/Chat-9B-Uncensored-GGUF/Chat-9B-Uncensored-noMTP-Q4_K_M.gguf": b"x",
            "someone/Chat-9B-Uncensored-GGUF/Chat-9B-Uncensored-draft-Q8_0.gguf": b"x",
            "someone/Chat-9B-Uncensored-GGUF/Chat-9B-Uncensored-Q8_0.gguf": self.gguf + b"pad",
            "bartowski/maker_Chat-9B-Uncensored-GGUF/maker_Chat-9B-Uncensored-Q4_K_M.gguf": self.gguf,
        })

    def test_what_the_user_gives_names_a_repository_and_a_quantization(self) -> None:
        repo = "someone/Chat-9B-Uncensored-GGUF"
        cases = {
            repo: (repo, "Q4_K_M"),
            f"  https://huggingface.co/{repo}  ": (repo, "Q4_K_M"),
            f"https://huggingface.co/{repo}/tree/main": (repo, "Q4_K_M"),
            f"https://huggingface.co/{repo}/blob/main/Chat-9B-Uncensored-Q8_0.gguf": (repo, "Q8_0"),
            f"hf.co/{repo}/resolve/main/UD-Q4_K_XL/Chat-UD-Q4_K_XL-00001-of-00002.gguf?download=true":
                (repo, "UD-Q4_K_XL"),
            f"https://huggingface.co/{repo}/blob/main/Qwen3.8-27B-IQ4_XS.gguf": (repo, "IQ4_XS"),
        }
        for given, want in cases.items():
            self.assertEqual(catalogs.parse_source(given), want, given)
        self.assertEqual(catalogs.parse_source(f"{repo}/blob/main/x-Q8_0.gguf", "q6_k"), (repo, "Q6_K"))
        for bad in ("", "just-a-name", "https://example.com/a/b", "a/b c"):
            with self.assertRaises(ValueError, msg=bad):
                catalogs.parse_source(bad)

    def test_an_added_model_is_listed_after_the_checked_ones_and_planned_without_network(self) -> None:
        hub = SearchHub(self.files, {})
        entry, notes = catalogs.add_mine("https://huggingface.co/someone/Chat-9B-Uncensored-GGUF",
                                         home=self.home, opener=hub)
        self.assertEqual(notes, [])
        self.assertEqual(entry["id"], "chat-9b-uncensored")
        self.assertEqual(entry["name"], "Chat-9B-Uncensored Q4_K_M (someone)")
        self.assertEqual((entry["hf_repo"], entry["hf_quant"]), ("someone/Chat-9B-Uncensored-GGUF", "Q4_K_M"))
        self.assertEqual(entry["rank"], catalogs.MINE_RANK)
        # found models come after it
        catalogs._write(catalogs.found_path(self.home), {"models": [
            {"id": "f", "name": "F", "rank": catalogs.NEW_RANK, "hf_repo": "unsloth/F-GGUF", "hf_quant": "Q4_K_M"}]})
        entries = catalogs.load_all(self.home)
        ids = [e["id"] for e in entries]
        self.assertEqual(ids[-2:], ["chat-9b-uncensored", "f"])
        self.assertTrue(entries[-2]["mine"])
        self.assertFalse(any(e.get("mine") for e in entries[:-2] + entries[-1:]))
        old = os.environ.get("GPUENSEMBLE_HOME")
        os.environ["GPUENSEMBLE_HOME"] = self.home
        try:
            devices, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
            cands = evaluate([e for e in load_catalog() if e["id"] == "chat-9b-uncensored"], devices, host, 4096)
        finally:
            if old is None:
                del os.environ["GPUENSEMBLE_HOME"]
            else:
                os.environ["GPUENSEMBLE_HOME"] = old
        self.assertEqual(len(cands), 1)
        self.assertIsNotNone(cands[0].plan)

    def test_adding_again_replaces_and_another_maker_gets_its_own_id(self) -> None:
        hub = SearchHub(self.files, {})
        catalogs.add_mine("someone/Chat-9B-Uncensored-GGUF", home=self.home, opener=hub)
        catalogs.add_mine("someone/Chat-9B-Uncensored-GGUF", name="Mine", home=self.home, opener=hub)
        other, _ = catalogs.add_mine("bartowski/maker_Chat-9B-Uncensored-GGUF", home=self.home, opener=hub)
        q8, _ = catalogs.add_mine("someone/Chat-9B-Uncensored-GGUF/blob/main/Chat-9B-Uncensored-Q8_0.gguf",
                                  home=self.home, opener=hub)
        added = catalogs.mine(self.home)
        self.assertEqual([e["id"] for e in added],
                         ["chat-9b-uncensored", "chat-9b-uncensored-maker", "chat-9b-uncensored-someone"])
        self.assertEqual(added[0]["name"], "Mine")
        self.assertEqual((other["id"], q8["hf_quant"]), ("chat-9b-uncensored-maker", "Q8_0"))
        self.assertEqual([e["rank"] for e in added], [catalogs.MINE_RANK + i for i in range(3)])
        self.assertTrue(catalogs.remove_mine("chat-9b-uncensored-maker", self.home))
        self.assertFalse(catalogs.remove_mine("chat-9b-uncensored-maker", self.home))
        self.assertEqual([e["id"] for e in catalogs.mine(self.home)],
                         ["chat-9b-uncensored", "chat-9b-uncensored-someone"])

    def test_models_that_cannot_be_added(self) -> None:
        checked = catalogs.checked_list(self.home)["models"][0]
        with self.assertRaisesRegex(ValueError, "already in the list"):
            catalogs.add_mine(checked["hf_repo"], checked["hf_quant"], home=self.home, opener=SearchHub({}, {}))
        with self.assertRaisesRegex(LookupError, "IQ1_S"):
            catalogs.add_mine("someone/Chat-9B-Uncensored-GGUF", "IQ1_S", home=self.home,
                              opener=SearchHub(self.files, {}))
        with self.assertRaisesRegex(LookupError, "login"):
            catalogs.add_mine("someone/Chat-9B-Uncensored-GGUF", home=self.home, opener=GatedHub(self.files, {}))
        image = os.path.join(self.home, "image.gguf")
        write_gguf(image, {"general.architecture": "flux"}, [("double_blocks.0.w", 4096), ("final.w", 4096)])
        with open(image, "rb") as f:
            files = RepoFiles({"someone/Picture-GGUF/Picture-Q4_K_M.gguf": f.read()})
        with self.assertRaisesRegex(ValueError, "not a chat model"):
            catalogs.add_mine("someone/Picture-GGUF", home=self.home, opener=SearchHub(files, {}))
        self.assertEqual(catalogs.mine(self.home), [])

    def test_a_model_of_an_untried_type_is_added_with_a_note(self) -> None:
        newtype = os.path.join(self.home, "newtype.gguf")
        write_gguf(newtype, {k.replace("llama.", "brandnew."): v for k, v in DENSE_META.items()
                             if k != "general.architecture"} | {"general.architecture": "brandnew"},
                   [("blk.0.w", 4096), ("blk.1.w", 4096)])
        with open(newtype, "rb") as f:
            files = RepoFiles({"someone/Brand-New-GGUF/Brand-New-Q4_K_M.gguf": f.read()})
        entry, notes = catalogs.add_mine("someone/Brand-New-GGUF", home=self.home, opener=SearchHub(files, {}))
        self.assertEqual(entry["id"], "brand-new")
        self.assertEqual(len(notes), 1)
        self.assertIn("brandnew", notes[0])

    def test_update_reads_the_added_models_too(self) -> None:
        catalogs._write(catalogs.mine_path(self.home), {"models": [
            {"id": "c", "name": "C", "rank": catalogs.MINE_RANK, "hf_repo": "someone/Chat-9B-Uncensored-GGUF",
             "hf_quant": "Q4_K_M"}]})
        checked = {"updated": "2099-01-01", "models": []}
        said = []
        code = catalogs.update(self.home, opener=SearchHub(self.files, {}, checked=checked), discover=False,
                               say=said.append)
        self.assertEqual(code, 0)
        self.assertEqual([s for s in said if s.startswith("read ")], ["read 0/1", "read 1/1"])

    def test_the_command_adds_lists_and_removes(self) -> None:
        hub = SearchHub(self.files, {})
        real = catalogs.add_mine
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": self.home}), \
                mock.patch.object(catalogs, "add_mine", lambda *a, **k: real(*a, opener=hub, **k)), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(["catalog", "--add", "someone/Chat-9B-Uncensored-GGUF"]), 0)
            self.assertEqual(cli.main(["catalog", "--add", "someone/Chat-9B-Uncensored-GGUF", "--quant", "IQ1_S"]), 1)
            self.assertEqual(cli.main(["catalog"]), 0)
            self.assertEqual(cli.main(["catalog", "--remove", "chat-9b-uncensored"]), 0)
            self.assertEqual(cli.main(["catalog", "--remove", "chat-9b-uncensored"]), 1)
        self.assertIn("added chat-9b-uncensored: Chat-9B-Uncensored Q4_K_M (someone)", out.getvalue())
        self.assertIn("up --model chat-9b-uncensored", out.getvalue())
        self.assertIn("added on this PC: 1 models", out.getvalue())
        self.assertIn("chat-9b-uncensored: someone/Chat-9B-Uncensored-GGUF Q4_K_M", out.getvalue())
        self.assertIn("no IQ1_S .gguf file found", err.getvalue())
        self.assertEqual(catalogs.mine(self.home), [])


if __name__ == "__main__":
    unittest.main()
