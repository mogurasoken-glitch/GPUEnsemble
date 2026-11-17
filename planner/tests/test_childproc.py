import os
import subprocess
import sys
import tempfile
import time
import unittest

from gpuensemble_planner.childproc import pid_alive, spawn
from gpuensemble_planner.llamacpp import LlamaCommand, run

SRC = os.path.join(os.path.dirname(__file__), "..", "src")

# A launcher that starts a long-running child the way `run` does, prints the
# child's pid and then waits to be killed.
PARENT = """
import sys, time
sys.path.insert(0, sys.argv[1])
from gpuensemble_planner.childproc import spawn
proc, guard = spawn([sys.executable, "-c", "import time; time.sleep(120)"])
print(proc.pid, guard.active, flush=True)
time.sleep(120)
"""

GUARDED = sys.platform == "win32" or sys.platform.startswith("linux")


def wait_until_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    return False


class ChildProcessTest(unittest.TestCase):
    @unittest.skipUnless(GUARDED, "no OS mechanism on this platform")
    def test_child_dies_when_the_launcher_is_killed(self) -> None:
        parent = subprocess.Popen([sys.executable, "-c", PARENT, os.path.abspath(SRC)],
                                  stdout=subprocess.PIPE, text=True)
        try:
            pid_text, active = parent.stdout.readline().split()
            child = int(pid_text)
            self.assertEqual(active, "True")
            self.assertTrue(pid_alive(child))
            parent.kill()  # SIGKILL / TerminateProcess: no Python cleanup runs
            parent.wait()
            self.assertTrue(wait_until_gone(child), f"child {child} survived its killed parent")
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.stdout.close()

    def test_spawned_child_runs_and_exits_normally(self) -> None:
        proc, guard = spawn([sys.executable, "-c", "raise SystemExit(3)"])
        self.assertEqual(proc.wait(timeout=30), 3)
        self.assertEqual(guard.active, GUARDED)
        self.assertFalse(pid_alive(proc.pid))

    def test_run_returns_the_exit_code_of_a_non_server_program(self) -> None:
        # `python -m compileall -q <empty dir>` stands in for llama-bench: run()
        # builds [exe, "-m", model, *args], and Python reads "-m compileall" as a
        # module. It prints nothing and exits with 0.
        with tempfile.TemporaryDirectory() as empty:
            command = LlamaCommand(args=["-q", empty], env={})
            self.assertEqual(run(sys.executable, command, "compileall"), 0)


if __name__ == "__main__":
    unittest.main()
