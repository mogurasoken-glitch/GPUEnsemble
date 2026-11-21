import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from gpuensemble_planner import boost
from gpuensemble_planner.devices import GIB, Device
from gpuensemble_planner.plan import Plan, Stage


def _dev(name, llama_device=None, rpc=None):
    return Device(name, 8 * GIB, 400e9, llama_device=llama_device, rpc=rpc)


def _stage(device, first, last):
    return Stage(device, first, last, 0, 0, 0, 8 * GIB, 0.01)


def _plan(stages, cpu_layers=0):
    return Plan("m", "dense", 8192, stages, cpu_layers, 0, 0.05)


DEVICES = [_dev("Ti", "CUDA0"), _dev("SUPER", "CUDA1"), _dev("4090", rpc="100.1.2.3:50052"), _dev("vk", "Vulkan0")]


class Cards(unittest.TestCase):
    def test_local_cuda_cards_with_layers(self):
        plan = _plan([_stage("SUPER", 0, 3), _stage("4090", 4, 16), _stage("Ti", 17, 35), _stage("vk", 0, -1)])
        self.assertEqual(boost.local_cards(plan, DEVICES), [1, 0])

    def test_one_card_alone_is_boosted_too(self):
        self.assertEqual(boost.local_cards(_plan([_stage("Ti", 0, 35)]), DEVICES), [0])

    def test_only_another_pcs_card(self):
        self.assertEqual(boost.local_cards(_plan([_stage("4090", 0, 35)]), DEVICES), [])


class Clocks(unittest.TestCase):
    def test_floor_is_a_share_of_the_highest_clock(self):
        self.assertEqual(boost.lock_for(0, 2190), boost.Lock(0, 1350, 2190))
        self.assertEqual(boost.lock_for(1, 2130), boost.Lock(1, 1320, 2130))

    def test_parse_max_clocks(self):
        self.assertEqual(boost.parse_max_clocks("0, 2190\n1, 2130\n2, [N/A]\n"), {0: 2190, 1: 2130})

    def test_lock_text_round_trip(self):
        lock = boost.Lock(1, 1320, 2130)
        self.assertEqual(lock.arg(), "1:1320:2130")
        self.assertEqual(boost.Lock.parse(lock.arg()), lock)

    def test_helper_command_runs_the_helper_with_the_locks(self):
        program, params = boost.helper_command([boost.Lock(0, 1350, 2190)], r"C:\nv\nvidia-smi.exe", 42, r"C:\t\s.json")
        self.assertEqual(program, sys.executable)
        self.assertIn("helper_main", params)
        self.assertIn("--parent 42", params)
        self.assertTrue(params.endswith("0:1350:2190"))


class Helper(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.status = os.path.join(self.dir.name, "status.json")
        self.calls = []

    def run_fake(self, fail_lock=False):
        def run(argv, **kwargs):
            self.calls.append(argv[1:])
            code = 1 if fail_lock and "-lgc" in argv else 0
            return subprocess.CompletedProcess(argv, code, "Not supported" if code else "All done.", "")
        return run

    def status_now(self):
        with open(self.status, encoding="utf-8") as f:
            return json.load(f)

    def test_locks_waits_for_the_parent_and_resets(self):
        beats = iter([True, True, False])
        code = boost.helper_main(["--smi", "smi", "--parent", "1", "--status", self.status, "0:1350:2190", "1:1320:2130"],
                                 run=self.run_fake(), sleep=lambda s: None, alive=lambda: next(beats))
        self.assertEqual(code, 0)
        self.assertEqual(self.calls, [["-i", "0", "-lgc", "1350,2190"], ["-i", "1", "-lgc", "1320,2130"],
                                      ["-i", "0", "-rgc"], ["-i", "1", "-rgc"]])
        self.assertEqual(self.status_now(), {"locks": ["0:1350:2190", "1:1320:2130"], "locked": True, "reset": True})

    def test_stop_file_ends_it(self):
        open(self.status + ".stop", "w").close()
        boost.helper_main(["--smi", "smi", "--parent", "1", "--status", self.status, "0:1350:2190"],
                          run=self.run_fake(), sleep=lambda s: self.fail("waited"), alive=lambda: True)
        self.assertEqual(self.calls[-1], ["-i", "0", "-rgc"])

    def test_a_refused_lock_is_reported_and_undone(self):
        code = boost.helper_main(["--smi", "smi", "--parent", "1", "--status", self.status, "0:1350:2190"],
                                 run=self.run_fake(fail_lock=True), sleep=lambda s: self.fail("waited"),
                                 alive=lambda: True)
        self.assertEqual(code, 0)
        state = self.status_now()
        self.assertFalse(state["locked"])
        self.assertIn("Not supported", state["error"])
        self.assertEqual(self.calls, [["-i", "0", "-lgc", "1350,2190"]])  # nothing locked, nothing to reset


class StartFor(unittest.TestCase):
    def test_nothing_to_lock(self):
        lines = []
        self.assertIsNone(boost.start_for(_plan([_stage("4090", 0, 35)]), DEVICES, lines.append))
        self.assertIn("nothing to lock", lines[0])

    def test_declined_dialog_is_a_note(self):
        lines = []
        with mock.patch.object(boost, "_nvidia_smi", return_value="smi"), \
                mock.patch.object(boost, "locks_for", return_value=[boost.Lock(0, 1350, 2190)]), \
                mock.patch.object(boost.Boost, "start", side_effect=OSError("the administrator dialog was declined")):
            self.assertIsNone(boost.start_for(_plan([_stage("Ti", 0, 35)]), DEVICES, lines.append))
        self.assertIn("CUDA0 (1350-2190 MHz)", lines[0])
        self.assertIn("declined", lines[1])

    def test_started(self):
        lines = []
        started = object()
        with mock.patch.object(boost, "_nvidia_smi", return_value="smi"), \
                mock.patch.object(boost, "locks_for", return_value=[boost.Lock(0, 1350, 2190)]), \
                mock.patch.object(boost.Boost, "start", return_value=started):
            self.assertIs(boost.start_for(_plan([_stage("Ti", 0, 35)]), DEVICES, lines.append), started)
        self.assertIn("boost: on", lines[-1])

    @unittest.skipIf(sys.platform == "win32", "runs on Windows")
    def test_not_on_other_systems(self):
        with self.assertRaises(OSError):
            boost.Boost.start([boost.Lock(0, 1350, 2190)], "smi")


if __name__ == "__main__":
    unittest.main()
