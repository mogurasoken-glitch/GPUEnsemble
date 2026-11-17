import os
import sys
import tempfile
import unittest

from gpuensemble_planner.fnv import FileHasher, fnv1a64, hash_file_python, name_of


class FakePowerShell:
    """Stands in for the PowerShell process: answers each name with what ``answer`` gives."""

    def __init__(self, answer, ready="ready\n") -> None:
        self.answer = answer
        self.lines = [ready]
        self.asked: list[str] = []
        self.stdin = self
        self.stdout = self
        self.closed = False

    def write(self, text: str) -> None:
        self.asked.append(text.strip())
        self.lines.append(self.answer(text.strip()))

    def flush(self) -> None:
        pass

    def readline(self) -> str:
        return self.lines.pop(0) if self.lines else ""

    def close(self) -> None:
        self.closed = True

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        pass


class FnvTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        with open(os.path.join(self.dir, "a.part"), "wb") as f:
            f.write(b"foobar")

    def test_known_values(self) -> None:
        # FNV-1a 64 test vectors; llama.cpp prints the hash as %016x.
        self.assertEqual(name_of(fnv1a64(b"")), "cbf29ce484222325")
        self.assertEqual(name_of(fnv1a64(b"a")), "af63dc4c8601ec8c")
        self.assertEqual(name_of(fnv1a64(b"foobar")), "85944171f73967e8")
        self.assertEqual(name_of(7), "0000000000000007")
        self.assertEqual(fnv1a64(b"bar", fnv1a64(b"foo")), fnv1a64(b"foobar"))  # in pieces
        self.assertEqual(hash_file_python(os.path.join(self.dir, "a.part")), "85944171f73967e8")

    def test_python_where_there_is_no_powershell(self) -> None:
        with FileHasher(self.dir, native=False) as h:
            self.assertFalse(h.native)
            self.assertEqual(h.hash("a.part"), "85944171f73967e8")

    def test_powershell_is_asked_by_name(self) -> None:
        started = []

        def popen(argv, **kw):
            started.append((argv, kw))
            return FakePowerShell(lambda name: "ok 85944171f73967e8\n")

        with FileHasher(self.dir, native=True, popen=popen) as h:
            self.assertTrue(h.native)
            self.assertEqual(h.hash("a.part"), "85944171f73967e8")
            proc = h.proc
        self.assertEqual(proc.asked, ["a.part"])
        self.assertTrue(proc.closed)
        argv, kw = started[0]
        self.assertEqual(argv[0], "powershell.exe")
        self.assertIn("-EncodedCommand", argv)
        self.assertEqual(kw["cwd"], self.dir)  # the folder may hold any characters; the names sent are ASCII

    @unittest.skipUnless(sys.platform == "win32", "PowerShell compiles the hash on Windows only")
    def test_the_real_powershell_agrees(self) -> None:
        data = os.urandom(5 * 1024 * 1024 + 3)  # more than one of its buffers
        with open(os.path.join(self.dir, "b.part"), "wb") as f:
            f.write(data)
        notes: list[str] = []
        with FileHasher(self.dir, log=notes.append) as h:
            self.assertTrue(h.native, notes)
            self.assertEqual(h.hash("a.part"), "85944171f73967e8")
            self.assertEqual(h.hash("b.part"), name_of(fnv1a64(data)))
        self.assertEqual(notes, [])

    def test_falls_back_to_python(self) -> None:
        notes: list[str] = []

        def missing(argv, **kw):
            raise FileNotFoundError("powershell.exe")

        cases = (missing,
                 lambda argv, **kw: FakePowerShell(lambda name: "", ready="At line:1 char:1\n"),  # did not compile
                 lambda argv, **kw: FakePowerShell(lambda name: "error Access denied\n"),
                 lambda argv, **kw: FakePowerShell(lambda name: ""))  # it died
        for popen in cases:
            with FileHasher(self.dir, native=True, popen=popen, log=notes.append) as h:
                self.assertEqual(h.hash("a.part"), "85944171f73967e8")
                self.assertFalse(h.native)
        self.assertEqual(len(notes), 4)
        self.assertTrue(all("hashing in Python" in n for n in notes), notes)


if __name__ == "__main__":
    unittest.main()
