import contextlib
import io
import os
import sys
import threading
import unittest
from unittest import mock

from gpuensemble_planner import cli, launch


class LaunchTest(unittest.TestCase):
    def start(self, *args: str, main=None, ask=None) -> tuple[int, str, mock.Mock]:
        ask = ask or mock.Mock(return_value="")
        out = io.StringIO()
        with mock.patch.object(cli, "main", main or mock.Mock(return_value=0)) as fake, \
                contextlib.redirect_stdout(out):
            code = launch.main(list(args), ask=ask)
        self.calls = fake.call_args_list
        return code, out.getvalue(), ask

    def test_runs_up_with_the_arguments_and_waits(self) -> None:
        code, out, ask = self.start("--yes", "--model", "qwen3-8b")
        self.assertEqual(code, 0)
        self.assertEqual(self.calls[0].args[0], ["up", "--open", "--remote", "auto", "--yes", "--model", "qwen3-8b"])
        self.assertIn("GPUEnsemble stopped (exit code 0)", out)
        ask.assert_called_once()

    def test_ctrl_c_still_waits_for_enter(self) -> None:
        code, out, ask = self.start(main=mock.Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(code, 130)
        self.assertIn("exit code 130", out)
        ask.assert_called_once()

    def test_bad_arguments_and_closed_input(self) -> None:
        code, out, _ = self.start("--bogus", main=mock.Mock(side_effect=SystemExit(2)),
                                  ask=mock.Mock(side_effect=EOFError))
        self.assertEqual(code, 2)
        self.assertIn("exit code 2", out)


class WorkerLaunchTest(unittest.TestCase):
    def pipe(self):
        read_end, write_end = os.pipe()
        stdin, typed = os.fdopen(read_end, "r"), os.fdopen(write_end, "w")
        self.addCleanup(stdin.close)
        return stdin, typed

    def test_lends_the_gpus_and_enter_closes_the_window(self) -> None:
        stdin, typed = self.pipe()
        before = sys.stdin
        seen = {}

        def fake_worker(argv):
            seen["argv"] = argv
            got = []
            # Like the worker's approval prompt: reads in a thread of its own, which is still waiting at the end.
            reader = threading.Thread(target=lambda: got.extend(iter(sys.stdin.readline, "")), daemon=True)
            reader.start()
            typed.write("y\n")
            typed.flush()
            for _ in range(100):
                if got:
                    break
                threading.Event().wait(0.02)
            seen["approvals"] = list(got)
            return 130  # Ctrl+C

        closed = threading.Event()
        out = io.StringIO()

        def run():
            with mock.patch.object(cli, "main", fake_worker), contextlib.redirect_stdout(out):
                seen["code"] = launch.main(["worker", "--device", "CUDA1"], stdin=stdin)
            closed.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        for _ in range(100):
            if "Press Enter" in out.getvalue():
                break
            threading.Event().wait(0.02)
        self.assertFalse(closed.is_set())
        typed.write("\n")  # one Enter closes the window
        typed.flush()
        self.assertTrue(closed.wait(5), out.getvalue())
        typed.close()
        self.assertEqual(seen["argv"], ["worker", "--lan", "--cache", "--serve", "--trust-lan", "--device", "CUDA1"])
        self.assertEqual(seen["approvals"], ["y\n"])
        self.assertEqual(seen["code"], 130)
        self.assertIn("GPUEnsemble worker stopped (exit code 130)", out.getvalue())
        self.assertIs(sys.stdin, before)

    def test_closed_input(self) -> None:
        stdin, typed = self.pipe()
        typed.close()
        with mock.patch.object(cli, "main", lambda argv: sys.stdin.readline() or 0), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.main(["worker"], stdin=stdin), 0)


if __name__ == "__main__":
    unittest.main()
