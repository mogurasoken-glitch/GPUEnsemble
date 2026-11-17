import json
import os
import tempfile
import unittest

from gpuensemble_planner import monitor
from gpuensemble_planner.devices import GIB, Device
from gpuensemble_planner.plan import Plan, Stage

# nvidia-smi --query-gpu=<GPU_FIELDS> --format=csv,noheader,nounits on the E5 rig, while a model answers.
SMI = """0, NVIDIA GeForce RTX 2080 Ti, 9810, 11264, 71, 64, 1905, 2100, 180.52, P2
1, NVIDIA GeForce RTX 2080 SUPER, 7302, 8192, 58, 31, 390, 2100, 41.10, P5
"""

# llama-server --metrics (b12375), after the warm-up and one answer.
METRICS = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 120
llamacpp:prompt_seconds_total 0.6
llamacpp:tokens_predicted_total 100
llamacpp:tokens_predicted_seconds_total 10
llamacpp:predicted_tokens_seconds 10
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 3
"""


def plan() -> Plan:
    stages = [
        Stage("RTX 2080 SUPER", 0, 17, 6 * GIB, 0, 0, 7 * GIB, 0.030),
        Stage("RTX 2080 Ti", 18, 63, 9 * GIB, 0, GIB // 2, 10 * GIB, 0.050),
        Stage("RTX 4090 (other PC)", 64, 79, 8 * GIB, 0, 0, 22 * GIB, 0.010),
    ]
    return Plan("test", "whole", 8192, stages, cpu_layers=0, cpu_bytes=0, seconds_per_token=0.100)


DEVICES = [
    Device("RTX 2080 Ti", 11 * GIB, 463e9, llama_device="CUDA0"),
    Device("RTX 2080 SUPER", 8 * GIB, 362e9, llama_device="CUDA1"),
    Device("RTX 4090 (other PC)", 24 * GIB, 820e9, llama_device="CUDA0", rpc="100.64.0.2:50052"),
]


class ParseTest(unittest.TestCase):
    def test_cards(self) -> None:
        ti, sup = monitor.parse_gpus(SMI + "garbage line\n")
        self.assertEqual((ti["index"], ti["name"], ti["used_gib"], ti["total_gib"]),
                         (0, "NVIDIA GeForce RTX 2080 Ti", 9.6, 11.0))
        self.assertEqual((ti["temp_c"], ti["load"], ti["clock_mhz"], ti["max_clock_mhz"], ti["pstate"]),
                         (71, 64, 1905, 2100, "P2"))
        self.assertEqual(sup["clock_mhz"], 390)

    def test_values_a_card_does_not_report(self) -> None:
        (g,) = monitor.parse_gpus("0, Tesla P40, 100, 24576, 40, 0, 544, 1531, [N/A], P8\n")
        self.assertIsNone(g["power_w"])

    def test_metrics(self) -> None:
        m = monitor.parse_metrics(METRICS)
        self.assertEqual(m[monitor.TOKENS], 100)
        self.assertEqual(m["llamacpp:spec_decode_num_accepted_tokens_per_pos_total"], 3)
        self.assertNotIn("#", "".join(m))


WARM = {monitor.TOKENS: 14, monitor.TOKEN_SECONDS: 2.0, monitor.PROMPT: 20, monitor.PROMPT_SECONDS: 0.1}


def ready(info: dict, warm: dict | None = None) -> dict:
    """The run file as `up` leaves it once the model is ready."""
    return dict(info, ready=1700000000.0, warm=dict(WARM if warm is None else warm))


RUN = ready({"pid": 1, "started": 1.0})


class RatesTest(unittest.TestCase):
    def test_speed_from_the_counters(self) -> None:
        r = monitor.Rates()
        r.update(None, RUN)
        self.assertIsNone(r.speed)
        r.update(monitor.parse_metrics(METRICS), RUN)  # 100 tokens in 10 s, 14 in 2 s of them the warm-up's
        self.assertAlmostEqual(r.speed, 86 / 8)
        self.assertAlmostEqual(r.prompt_speed, 100 / 0.5)
        r.update({monitor.TOKENS: 300, monitor.TOKEN_SECONDS: 25, monitor.PROMPT: 120, monitor.PROMPT_SECONDS: 0.6}, RUN)
        self.assertAlmostEqual(r.speed, 200 / 15)
        self.assertAlmostEqual(r.prompt_speed, 200.0)  # no new prompt: the last speed stays
        r.update({monitor.TOKENS: 300, monitor.TOKEN_SECONDS: 25}, RUN)
        self.assertAlmostEqual(r.speed, 200 / 15)  # idle: the last speed stays

    def test_the_warm_up_is_not_an_answer(self) -> None:
        r = monitor.Rates()
        loading = {"pid": 1, "started": 1.0}
        r.update({monitor.TOKENS: 6, monitor.TOKEN_SECONDS: 1.0}, loading)  # warming up: not ready yet
        self.assertIsNone(r.speed)
        r.update(dict(WARM), RUN)  # ready, nothing asked yet
        self.assertIsNone(r.speed)
        r.update({monitor.TOKENS: 114, monitor.TOKEN_SECONDS: 75.0}, RUN)
        self.assertAlmostEqual(r.speed, 100 / 73)

    def test_another_model_starts_from_nothing(self) -> None:
        r = monitor.Rates()
        r.update({monitor.TOKENS: 114, monitor.TOKEN_SECONDS: 75.0}, RUN)
        self.assertIsNotNone(r.speed)
        r.update(None, None)  # stopped
        self.assertIsNone(r.speed)
        r.update({monitor.TOKENS: 5, monitor.TOKEN_SECONDS: 1.0}, ready({"pid": 2, "started": 9.0}, warm={}))
        self.assertAlmostEqual(r.speed, 5.0)

    def test_speeds_as_shown(self) -> None:
        self.assertEqual(monitor._speed(1.3749), 1.37)
        self.assertEqual(monitor._speed(13.349), 13.3)
        self.assertIsNone(monitor._speed(None))


class SnapshotTest(unittest.TestCase):
    def run_info(self) -> dict:
        return ready(monitor.run_info(plan(), DEVICES, "llama-3.3-70b", "0.0.0.0", 8080), warm={})

    def snap(self, smi: str = SMI, metrics: dict | None = None, run: dict | None = None) -> dict:
        rates = monitor.Rates()
        metrics = monitor.parse_metrics(METRICS) if metrics is None else metrics
        run = self.run_info() if run is None else run
        rates.update(metrics, run)
        return monitor.snapshot(run, monitor.parse_gpus(smi), metrics, rates)

    def test_run_info(self) -> None:
        info = self.run_info()
        self.assertEqual((info["host"], info["port"], info["model"]), ("127.0.0.1", 8080, "llama-3.3-70b"))
        self.assertEqual([s["index"] for s in info["stages"]], [1, 0, None])
        self.assertEqual([s["here"] for s in info["stages"]], [True, True, False])

    def test_a_running_model(self) -> None:
        snap = self.snap()
        self.assertTrue(snap["running"] and snap["busy"] and snap["ready"])
        self.assertEqual(snap["api"], "http://127.0.0.1:8080/v1")
        self.assertEqual((snap["speed"], snap["estimate"], snap["tokens"]), (10.0, 10.0, 100))
        self.assertEqual(snap["slowest"], "RTX 2080 Ti")
        self.assertEqual([s["ms"] for s in snap["stages"]], [30.0, 50.0, 10.0])
        self.assertIsNone(snap["cpu"])  # the 10 ms the stages leave is not the CPU's
        ti, sup = snap["gpus"]
        self.assertEqual((ti["layers"], sup["layers"]), ("18-63", "0-17"))
        self.assertFalse(ti["clock_down"])
        self.assertTrue(sup["clock_down"])  # P5 while answering
        self.assertTrue(any("2080 SUPER is in P5 at 390 of 2100 MHz" in n for n in snap["notes"]))
        json.dumps(snap)

    def test_the_time_per_stage_follows_the_measured_speed(self) -> None:
        snap = self.snap("", {monitor.TOKENS: 50, monitor.TOKEN_SECONDS: 10})  # 5 tokens/s: half the estimate
        self.assertEqual(snap["speed"], 5.0)
        self.assertEqual([s["ms"] for s in snap["stages"]], [60.0, 100.0, 20.0])
        self.assertTrue(any("generating at 5 tokens/s, below the estimate of 10" in n for n in snap["notes"]))

    def test_an_idle_card_is_not_said_to_be_slowed_down(self) -> None:
        snap = self.snap(SMI.replace(", 31, 390,", ", 0, 390,"), {monitor.PROCESSING: 0})
        self.assertFalse(snap["gpus"][1]["clock_down"])

    def test_a_working_state_is_not_slowed_down_whatever_its_clock(self) -> None:
        # The E5's 2080 Ti holds P2 at 1350 of 2190 MHz under "prefer maximum performance".
        snap = self.snap("0, NVIDIA GeForce RTX 2080 Ti, 9810, 11264, 59, 20, 1350, 2190, 81.0, P2\n")
        self.assertFalse(snap["gpus"][0]["clock_down"])

    def test_a_card_that_hardly_counts_is_not_said_to_be_slowed_down(self) -> None:
        # 49 layers on the CPU: the cards wait 97% of each token, and their clocks then cost nothing.
        p = plan()
        p.cpu_layers, p.seconds_per_token = 49, 0.700
        for st, sec in zip(p.stages, (0.009, 0.010, 0.0)):
            st.seconds_per_token = sec
        info = ready(monitor.run_info(p, DEVICES, "m", "127.0.0.1", 8080), warm={})
        snap = self.snap(SMI.replace("P2", "P3"), run=info)
        self.assertEqual(snap["slowest"], "CPU")
        self.assertFalse(any(g["clock_down"] for g in snap["gpus"]))
        self.assertFalse(any("lowered its clocks" in n for n in snap["notes"]))

    def test_a_card_that_does_not_tell_its_state_is_judged_by_its_clock(self) -> None:
        snap = self.snap(SMI.replace("P5", "[N/A]"))
        self.assertTrue(snap["gpus"][1]["clock_down"])  # 390 of 2100 MHz
        snap = self.snap(SMI.replace("P5", "[N/A]").replace(", 390,", ", 1650,"))
        self.assertFalse(snap["gpus"][1]["clock_down"])

    def test_cpu_layers_that_take_the_longest(self) -> None:
        p = plan()
        p.cpu_layers, p.seconds_per_token = 5, 0.300
        info = monitor.run_info(p, DEVICES, "m", "127.0.0.1", 8080)
        snap = monitor.snapshot(info, [], None, monitor.Rates())
        self.assertEqual(snap["slowest"], "CPU")
        self.assertEqual((snap["cpu"]["layers"], snap["cpu"]["ms"]), (5, 210.0))

    def test_no_model(self) -> None:
        snap = monitor.snapshot(None, monitor.parse_gpus(SMI), None, monitor.Rates())
        self.assertFalse(snap["running"] or snap["ready"])
        self.assertIsNone(snap["speed"])
        self.assertFalse(any(g["clock_down"] for g in snap["gpus"]))
        self.assertIn("no model is running", monitor.format_text(snap))

    def test_text(self) -> None:
        text = monitor.format_text(self.snap())
        self.assertIn("answering; the answers before this one ran at 10 tokens/s (estimate 10)", text)
        self.assertIn("slowest part: RTX 2080 Ti", text)
        self.assertIn("RTX 4090 (other PC) (other PC): layers 64-79", text)
        self.assertIn("CLOCKED DOWN", text)

    def test_experts_the_cpu_computes_are_named_with_their_stage(self) -> None:
        # gpt-oss-120b on the E5 rig: the 2080 Ti's stage is the slowest because the CPU runs its layers' experts.
        p = plan()
        p.stages[1].expert_path = "cpu"
        run = ready(monitor.run_info(p, DEVICES, "gpt-oss-120b", "127.0.0.1", 8080))
        self.assertEqual([s["cpu_experts"] for s in run["stages"]], [False, True, False])
        snap = self.snap(run=run)
        self.assertTrue(snap["slowest_cpu_experts"])
        text = monitor.format_text(snap)
        self.assertIn("slowest part: RTX 2080 Ti (with the experts of its layers that the CPU computes)", text)
        self.assertIn("RTX 2080 Ti: layers 18-63, experts on the CPU, ~", text)
        self.assertFalse(self.snap()["slowest_cpu_experts"])
        old = self.run_info()  # a run file written before the key existed
        for s in old["stages"]:
            del s["cpu_experts"]
        self.assertFalse(self.snap(run=old)["slowest_cpu_experts"])

    def test_text_says_what_the_model_is_doing(self) -> None:
        idle = dict(monitor.parse_metrics(METRICS), **{monitor.PROCESSING: 0})
        self.assertIn("idle; the last answers ran at 10 tokens/s", monitor.format_text(self.snap(metrics=idle)))
        fresh = {monitor.PROCESSING: 0}
        self.assertIn("ready, no answer yet", monitor.format_text(self.snap(metrics=fresh)))
        self.assertIn("answering its first request", monitor.format_text(self.snap(metrics={monitor.PROCESSING: 1})))
        loading = monitor.run_info(plan(), DEVICES, "m", "127.0.0.1", 8080)
        self.assertIn("loading or warming up", monitor.format_text(self.snap(metrics={}, run=loading)))


class RunFileTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "run", "up.json")

    def test_written_read_and_removed(self) -> None:
        info = monitor.run_info(plan(), DEVICES, "m", "127.0.0.1", 8080)
        monitor.write_run(info, self.path)
        self.assertEqual(monitor.read_run(self.path, alive=lambda pid: True)["model"], "m")
        monitor.clear_run(self.path)
        self.assertIsNone(monitor.read_run(self.path))
        monitor.clear_run(self.path)  # twice is fine

    def test_left_by_an_up_that_is_gone(self) -> None:
        # Killed (the app's Stop button): it could not remove its file. Whoever reads it next does.
        monitor.write_run({"pid": 12345, "model": "m"}, self.path)
        self.assertIsNone(monitor.read_run(self.path, alive=lambda pid: False))
        self.assertFalse(os.path.exists(self.path))

    def test_this_process_is_alive(self) -> None:
        monitor.write_run({"pid": os.getpid(), "model": "m"}, self.path)
        self.assertEqual(monitor.read_run(self.path)["model"], "m")

    def test_ready_notes_the_counters_of_the_warm_up(self) -> None:
        monitor.write_run(monitor.run_info(plan(), DEVICES, "m", "127.0.0.1", 8080), self.path)
        monitor.mark_ready(monitor.parse_metrics(METRICS), self.path)
        info = monitor.read_run(self.path, alive=lambda pid: True)
        self.assertGreater(info["ready"], 0)
        self.assertEqual(info["warm"], {monitor.TOKENS: 100, monitor.TOKEN_SECONDS: 10,
                                        monitor.PROMPT: 120, monitor.PROMPT_SECONDS: 0.6})
        self.assertEqual(info["model"], "m")

    def test_ready_without_counters_or_file(self) -> None:
        monitor.mark_ready(None, self.path)  # no file: nothing to do
        self.assertFalse(os.path.exists(self.path))
        monitor.write_run({"pid": 1}, self.path)
        monitor.mark_ready(None, self.path)  # llama-server did not answer /metrics: count from zero
        self.assertEqual(monitor.read_run(self.path, alive=lambda pid: True)["warm"][monitor.TOKENS], 0.0)


class CommandTest(unittest.TestCase):
    def test_json_once(self) -> None:
        import contextlib
        import io
        from unittest import mock

        from gpuensemble_planner import cli

        out = io.BytesIO()
        fake = mock.Mock(buffer=out)
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": home}), \
                mock.patch.object(monitor, "read_gpus", return_value=monitor.parse_gpus(SMI)), \
                mock.patch("sys.stdout", fake), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["monitor", "--json"]), 0)
        snap = json.loads(out.getvalue().decode("utf-8"))
        self.assertFalse(snap["running"])
        self.assertEqual(len(snap["gpus"]), 2)


class WatchTest(unittest.TestCase):
    def test_readings(self) -> None:
        out, slept = [], []
        info = ready(monitor.run_info(plan(), DEVICES, "m", "127.0.0.1", 8080), warm={})
        counters = iter([{monitor.TOKENS: 10, monitor.TOKEN_SECONDS: 1},
                         {monitor.TOKENS: 30, monitor.TOKEN_SECONDS: 3}])
        asked = []

        def metrics(host, port):
            asked.append((host, port))
            return next(counters)

        def no_smi():
            raise RuntimeError("nvidia-smi was not found")

        monitor.watch(2.0, 2, out.append, read_run_=lambda: info, read_gpus_=no_smi,
                      read_metrics_=metrics, sleep=slept.append)
        self.assertEqual(len(out), 2)
        self.assertEqual(slept, [2.0])
        self.assertEqual(asked, [("127.0.0.1", 8080)] * 2)
        self.assertEqual(out[1]["speed"], 10.0)
        self.assertIn("could not read the GPUs: nvidia-smi was not found", out[0]["notes"])

    def test_no_model_asks_no_server(self) -> None:
        out = []
        monitor.watch(1.0, 1, out.append, read_run_=lambda: None, read_gpus_=lambda: [],
                      read_metrics_=lambda h, p: self.fail("asked"))
        self.assertFalse(out[0]["running"])


if __name__ == "__main__":
    unittest.main()
