import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from gpuensemble_planner import cli, devices, hub
from gpuensemble_planner.discovery import Found

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
RIG = os.path.join(EXAMPLES, "rig-e5-turing.json")
WORKER_GPUS = [{"name": "RTX 4090", "vram_gib": 24, "llama_device": "CUDA0", "bandwidth_gbs": 820}]


def state(*argv: str) -> tuple[int, dict | None, str]:
    out, err = io.StringIO(), io.StringIO()
    out.buffer = io.BytesIO()  # the app's JSON is written as UTF-8 bytes
    with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": home}), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["state", *argv])
    raw = out.buffer.getvalue()
    return code, json.loads(raw.decode("utf-8")) if raw else None, err.getvalue()


class StateTest(unittest.TestCase):
    def test_rig_gives_gpus_models_and_the_best_one(self) -> None:
        code, s, _ = state("--rig", RIG)
        self.assertEqual(code, 0)
        self.assertEqual(s["version"], 1)
        self.assertEqual([g["name"] for g in s["gpus"]], ["RTX 2080 Ti 11GB", "RTX 2080 SUPER 8GB"])
        self.assertTrue(all(g["here"] and g["pc"] == "this PC" for g in s["gpus"]))
        self.assertEqual(s["best"], "gpt-oss-120b")
        self.assertEqual(s["workers"], [])
        self.assertFalse(s["llama_cpp"])
        ids = [m["id"] for m in s["models"]]
        self.assertIn("gpt-oss-120b", ids)
        for m in s["models"]:
            self.assertEqual(set(m), {"id", "name", "rank", "new", "mine", "hf_repo", "file_gib", "downloaded", "path", "fits", "speed",
                                      "speed_empty", "there", "placement", "layout", "error", "cpu_layers", "stages"})
            self.assertFalse(m["downloaded"])
            if m["fits"]:
                self.assertGreater(m["speed"], 0)
                self.assertTrue(m["stages"])
                self.assertEqual(m["layout"]["gpus"], sum(1 for st in m["stages"] if st["first"] is not None))
            else:
                self.assertIsNone(m["layout"])
                gpus = {g["name"] for g in s["gpus"]}
                for st in m["stages"]:
                    self.assertIn(st["gpu"], gpus)
                    self.assertLessEqual(st["used_gib"], st["capacity_gib"] + 0.05)
            if not m["fits"]:
                self.assertTrue(m["error"])

    def test_without_any_gpu_it_still_answers(self) -> None:
        # A PC with 64 GiB of RAM: what fits in RAM depends on it, and the CI's Windows runner has 16.
        with mock.patch.object(devices, "detect_nvidia", side_effect=RuntimeError("nvidia-smi was not found")), \
                mock.patch.object(devices, "other_cards", return_value=[]), \
                mock.patch.object(devices, "physical_ram_bytes", return_value=64 * devices.GIB), \
                mock.patch.object(cli, "installed", return_value=None):
            code, s, _ = state()
        self.assertEqual(code, 0)
        self.assertEqual(s["gpus"], [])
        self.assertEqual(len(s["notes"]), 1)
        self.assertIn("no GPU found on this PC", s["notes"][0])
        # Models that fit in RAM run on the CPU alone.
        cpu_only = [m for m in s["models"] if m["fits"]]
        self.assertTrue(cpu_only)
        for m in cpu_only:
            self.assertEqual(m["layout"]["gpus"], 0)
            self.assertEqual(m["placement"].split(",")[0], "CPU only")

    def test_workers_are_listed_and_their_gpus_named_after_their_pc(self) -> None:
        lends = Found("192.168.10.2", 50052, "pc4090", True, False, WORKER_GPUS)
        waits = Found("192.168.10.3", 50052, "i7", False, asking=True)
        with mock.patch.object(cli, "_find", return_value=[lends, waits]), \
                mock.patch.object(cli, "measure_rtt", return_value=0.0005), \
                mock.patch.object(cli, "is_this_pc", return_value=False):
            code, s, _ = state("--rig", RIG, "--remote", "auto")
        self.assertEqual(code, 0)
        self.assertEqual([(w["name"], w["allowed"]) for w in s["workers"]], [("pc4090", True), ("i7", False)])
        far = [g for g in s["gpus"] if not g["here"]]
        self.assertEqual(len(far), 1)
        self.assertEqual(far[0]["pc"], "pc4090")
        self.assertEqual(far[0]["endpoint"], "192.168.10.2:50052")
        names = {g["name"] for g in s["gpus"]}
        for m in s["models"]:
            for st in m["stages"]:
                self.assertIn(st["gpu"], names, m["id"])

    def test_downloaded_model_is_marked(self) -> None:
        with mock.patch.object(cli, "find_local", side_effect=lambda e, base=None:
                               "C:/m/x.gguf" if e["id"] == "qwen3-32b" else None):
            _, s, _ = state("--rig", RIG)
        [m] = [m for m in s["models"] if m["id"] == "qwen3-32b"]
        self.assertTrue(m["downloaded"])
        self.assertEqual(m["path"], "C:/m/x.gguf")

    def test_models_folder_from_the_app(self) -> None:
        # The app passes the folder the user chose as GPUENSEMBLE_MODELS: every command uses it.
        with tempfile.TemporaryDirectory() as chosen:
            entry = {"id": "x", "hf_repo": "org/X-GGUF", "files": ["x.gguf"]}
            os.makedirs(os.path.join(chosen, "org__X-GGUF"))
            with open(os.path.join(chosen, "org__X-GGUF", "x.gguf"), "wb") as f:
                f.write(b"GGUF")
            with mock.patch.dict(os.environ, {"GPUENSEMBLE_MODELS": chosen}):
                _, s, _ = state("--rig", RIG)
                self.assertEqual(hub.models_dir(), chosen)
                self.assertEqual(hub.local_path(entry), os.path.join(chosen, "org__X-GGUF"))
            self.assertEqual(s["models_dir"], chosen)
        with mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": "/h"}):
            os.environ.pop("GPUENSEMBLE_MODELS", None)
            self.assertEqual(hub.models_dir(), os.path.join("/h", "models"))


if __name__ == "__main__":
    unittest.main()
