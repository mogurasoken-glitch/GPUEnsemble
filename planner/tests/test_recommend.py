import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from gpuensemble_planner.cli import main
from gpuensemble_planner.devices import GIB, Device, load_rig
from gpuensemble_planner.recommend import evaluate, format_table, load_catalog, pick

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
RIG = os.path.join(EXAMPLES, "rig-e5-turing.json")


def e5():
    devices, host = load_rig(RIG)
    host.expert_path = "cpu"
    return devices, host


class CatalogTest(unittest.TestCase):
    def test_every_entry_can_be_downloaded(self) -> None:
        for e in load_catalog():
            self.assertTrue(e.get("hf_repo") and e.get("hf_quant") and e.get("name"), e["id"])
            self.assertEqual(e["hf_repo"].count("/"), 1, e["id"])

    def test_entries_build_profiles_of_the_stated_size(self) -> None:
        entries = load_catalog()
        self.assertEqual([e["rank"] for e in entries], sorted(e["rank"] for e in entries))
        self.assertEqual(len({e["id"] for e in entries}), len(entries))
        devices, host = e5()
        for c in evaluate(entries, devices, host, 4096):
            size = c.model.total_weight_bytes / GIB
            self.assertAlmostEqual(size, c.entry["file_gib"], delta=0.03 * size, msg=c.entry["id"])
            # The size of the vocabulary: what a head on another PC sends back with every token.
            self.assertGreater(c.model.n_vocab, 30000, c.entry["id"])


class RecommendTest(unittest.TestCase):
    def test_e5_rig_gets_gpt_oss_120b(self) -> None:
        # Measured on the E5 rig: 7.8 tokens/s with the planner's command.
        devices, host = e5()
        candidates = evaluate(load_catalog(), devices, host, 8192)
        best = pick(candidates)
        self.assertEqual(best.entry["id"], "gpt-oss-120b")
        self.assertGreater(best.speed, 5)
        table = format_table(candidates, best, 8192, 5)
        self.assertIn("* gpt-oss-120b", table)
        self.assertIn("recommended: gpt-oss-120b", table)

    def test_higher_speed_floor_picks_a_smaller_model(self) -> None:
        devices, host = e5()
        candidates = evaluate(load_catalog(), devices, host, 8192)
        # The best-ranked model that reaches the floor: Qwen3.8-27B fits on the two cards...
        best = pick(candidates, min_speed=20)
        self.assertEqual(best.entry["id"], "qwen3.8-27b")
        self.assertEqual(best.plan.cpu_layers, 0)
        # ...and above its speed, a model that reads few of its weights for a token.
        self.assertEqual(pick(candidates, min_speed=30).entry["id"], "gpt-oss-20b")

    def test_unreachable_floor_falls_back_to_the_fastest(self) -> None:
        devices, host = e5()
        candidates = evaluate(load_catalog(), devices, host, 8192)
        best = pick(candidates, min_speed=1000)
        self.assertEqual(best.speed, max(c.speed for c in candidates))
        self.assertIn("nothing reaches 1000", format_table(candidates, best, 8192, 1000))

    def test_a_model_that_fits_on_another_pc_alone_shows_that_pcs_own_speed(self) -> None:
        # A faster card at another site: what fits on it would run there from end to end.
        devices, host = e5()
        far = Device("RTX 4090", 24 * GIB, 750e9, llama_device="CUDA0", rpc="100.64.0.2:50052", rpc_overhead=9.5e-3)
        candidates = evaluate(load_catalog(), devices + [far], host, 8192)
        by_id = {c.entry["id"]: c for c in candidates}
        plan = by_id["qwen3-32b"].plan
        self.assertEqual([s.device for s in plan.on_other_pc.stages if s.n_layers], ["RTX 4090"])
        self.assertGreater(plan.tokens_per_second_on_other_pc, plan.tokens_per_second * 1.2)
        rows = format_table(candidates, pick(candidates), 8192, 5).splitlines()
        [row] = [line for line in rows if "Qwen3-32B" in line]
        self.assertIn(f"{plan.tokens_per_second_on_other_pc:.1f} if started on the other PC", row)
        # A small model is faster there too, though this PC's own card holds it.
        small = by_id["qwen3-8b"].plan
        self.assertEqual([s.device for s in small.stages if s.n_layers], ["RTX 2080 Ti 11GB"])
        self.assertIn("if started on the other PC", [line for line in rows if "Qwen3-8B" in line][0])
        # A model that needs both PCs has no such speed.
        self.assertIsNone(by_id["gpt-oss-120b"].plan.on_other_pc)
        self.assertNotIn("other PC", [line for line in rows if "gpt-oss-120b" in line][0])

    def test_a_pc_that_runs_models_itself_is_counted_at_its_own_speed(self) -> None:
        # The other PC's worker was started with --serve: `up` starts there what that PC holds alone
        # and is faster at, so the table shows what will run, and the choice is made by those speeds.
        devices, host = e5()
        far = Device("RTX 4090", 24 * GIB, 750e9, llama_device="CUDA0", rpc="100.64.0.2:50052", rpc_overhead=9.5e-3,
                     rpc_serve=True)
        candidates = evaluate(load_catalog(), devices + [far], host, 8192)
        by_id = {c.entry["id"]: c for c in candidates}
        qwen = by_id["qwen3-32b"]
        self.assertTrue(qwen.there)
        there = qwen.plan.on_other_pc
        self.assertAlmostEqual(qwen.speed, there.tokens_per_second)
        self.assertGreater(qwen.speed, qwen.plan.tokens_per_second * 1.2)
        best = pick(candidates, min_speed=30)
        rows = format_table(candidates, best, 8192, 30).splitlines()
        # The best-ranked model that reaches 30 tokens/s does so only there: chosen for what it does there.
        self.assertEqual(best.entry["id"], "qwen3.8-27b")
        self.assertTrue(best.there)
        self.assertLess(best.plan.tokens_per_second, 30)
        self.assertTrue(next(line for line in rows if "Qwen3.8-27B" in line).startswith(" * "), rows)
        row = next(line for line in rows if "Qwen3-32B" in line)
        self.assertIn(f"{there.tokens_per_second_empty:4.1f} /{there.tokens_per_second:5.1f} /"
                      f"{there.tokens_per_second_full:5.1f}  1 GPU, runs on the other PC", row)
        # A model that needs both PCs runs here, and so does one that the other PC is not faster at.
        self.assertFalse(by_id["gpt-oss-120b"].there)
        self.assertAlmostEqual(by_id["gpt-oss-120b"].speed, by_id["gpt-oss-120b"].plan.tokens_per_second)
        slow = Device("GTX 1080 Ti", 24 * GIB, 100e9, llama_device="CUDA0", rpc="100.64.0.2:50052",
                      rpc_overhead=9.5e-3, rpc_serve=True)
        small = {c.entry["id"]: c for c in evaluate(load_catalog(), devices + [slow], host, 8192)}["qwen3-8b"]
        self.assertIsNotNone(small.plan.on_other_pc)
        self.assertFalse(small.there)

    def test_cli_says_where_the_recommended_model_runs(self) -> None:
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": home}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            remote = os.path.join(home, "remote.json")
            with open(remote, "w", encoding="utf-8") as f:
                f.write('{"gpus": [{"name": "RTX 4090", "vram_gib": 24, "rpc": "100.64.0.2:50052", '
                        '"rpc_overhead_ms": 9.5, "serve": true}]}')
            code = main(["recommend", "--rig", RIG, "--remote", remote, "--min-speed", "30"])
        self.assertEqual(code, 0)
        self.assertIn("recommended: Qwen3.8-27B", out.getvalue())
        self.assertIn("to start it (it runs on the other PC, which downloads 16 GiB first if it does not have it):",
                      out.getvalue())

    def test_small_pc_skips_what_does_not_fit(self) -> None:
        _, host = e5()
        host.ram_bytes = 12 * GIB
        devices = [Device("RTX 3060 12GB", 12 * GIB, 270e9, llama_device="CUDA0")]
        candidates = evaluate(load_catalog(), devices, host, 8192)
        by_id = {c.entry["id"]: c for c in candidates}
        self.assertIsNone(by_id["gpt-oss-120b"].plan)
        self.assertIn("does not fit", format_table(candidates, pick(candidates), 8192, 5))
        best = pick(candidates)
        self.assertIsNotNone(best.plan)
        self.assertGreaterEqual(best.speed, 5)

    def test_cli(self) -> None:
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": home}), \
                contextlib.redirect_stdout(out):
            code = main(["recommend", "--rig", RIG, "--ctx", "4096"])
        self.assertEqual(code, 0)
        self.assertIn("recommended: gpt-oss-120b", out.getvalue())
        self.assertIn(f"gpuensemble-plan up --model gpt-oss-120b --rig {RIG} --ctx 4096", out.getvalue())
        self.assertIn("(downloads 59 GiB first, and llama.cpp if needed)", out.getvalue())
        # A fresh install has read only the models whose files come with it: it says the others are left out.
        left = len(load_catalog()) - len(evaluate(load_catalog(), *e5(), 4096))
        self.assertGreater(left, 0)
        self.assertIn(f"{left} more model(s) in the list are not read yet and were left out; "
                      "`gpuensemble-plan catalog --update` reads them", out.getvalue())

    def test_version(self) -> None:
        from gpuensemble_planner import __version__
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as done:
            main(["--version"])
        self.assertEqual(done.exception.code, 0)
        self.assertEqual(out.getvalue().strip(), f"GPUEnsemble {__version__}")

    def test_cli_keeps_the_other_pcs(self) -> None:
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": home}), \
                mock.patch("gpuensemble_planner.cli.found_remote", return_value=[]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["recommend", "--rig", RIG, "--ctx", "4096", "--remote", "auto:192.168.10.2"])
        self.assertEqual(code, 0)
        # The suggested command plans with the same GPUs as this table.
        self.assertIn(f"--rig {RIG} --remote auto:192.168.10.2 --ctx 4096", out.getvalue())


if __name__ == "__main__":
    unittest.main()
