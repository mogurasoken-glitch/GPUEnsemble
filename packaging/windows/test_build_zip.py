import contextlib
import io
import os
import tempfile
import unittest
import zipfile
from unittest import mock

import build_zip


def fake_python_zip(path: str) -> None:
    """Same layout as python.org's embeddable package, without the binaries."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("python.exe", b"MZ")
        z.writestr("python312.dll", b"MZ")
        z.writestr("python312.zip", b"PK\x05\x06" + b"\0" * 18)
        z.writestr("python312._pth", "python312.zip\n.\n\n# Uncomment to run site.main() automatically\n#import site\n")
        z.writestr("LICENSE.txt", "PSF")


class BuildZipTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.python_zip = os.path.join(self.tmp, "python-embed.zip")
        fake_python_zip(self.python_zip)
        # A stand-in for the built app: the tests never run the .NET SDK (nor need one).
        self.app_exe = os.path.join(self.tmp, "GPUEnsemble.exe")
        with open(self.app_exe, "wb") as f:
            f.write(b"MZapp")
        with open(self.app_exe + ".config", "w", encoding="utf-8") as f:
            f.write("<configuration/>")

    def build(self, *extra: str, kind: str = "app") -> zipfile.ZipFile:
        """Builds the zips; returns the one of ``kind`` ("app" or "cli")."""
        out = os.path.join(self.tmp, "dist")
        if "--app-exe" not in extra and "--no-app" not in extra:
            extra = ("--app-exe", self.app_exe) + extra
        extra = tuple(a for a in extra if a != "--no-app")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(build_zip.main(["--python-zip", self.python_zip, "--out", out, *extra]), 0)
        self.made = sorted(os.listdir(out))
        name = f"GPUEnsemble-{build_zip.package_version()}-windows-x64-{kind}.zip"
        self.assertIn(name, self.made)
        z = zipfile.ZipFile(os.path.join(out, name))
        self.addCleanup(z.close)
        return z

    def test_layout(self) -> None:
        z = self.build()
        top = f"GPUEnsemble-{build_zip.package_version()}/"
        names = set(z.namelist())
        for n in ("GPUEnsemble.exe", "GPUEnsemble.exe.config", "Start GPUEnsemble.cmd", "Start GPUEnsemble worker.cmd", "gpuensemble.cmd", "README.txt", "LICENSE",
                  "THIRD_PARTY_NOTICES.md",
                  "python/python.exe",
                  "python/LICENSE.txt", "app/gpuensemble_planner/__main__.py",
                  "app/gpuensemble_planner/catalog.json", "examples/rig-e5-turing.json"):
            self.assertIn(top + n, names)
        self.assertFalse([n for n in names if "__pycache__" in n or n.endswith(".pyc")])
        self.assertTrue(all(n.startswith(top) for n in names))
        # Windows file names ignore case: no two entries may differ only by it.
        self.assertEqual(len({n.lower() for n in names}), len(names))

    def test_two_zips_same_but_for_the_app(self) -> None:
        app = self.build()
        cli = self.build(kind="cli")
        v = build_zip.package_version()
        self.assertEqual(self.made, [f"GPUEnsemble-{v}-windows-x64-app.zip", f"GPUEnsemble-{v}-windows-x64-cli.zip"])
        top = f"GPUEnsemble-{v}/"
        extra = {top + "GPUEnsemble.exe", top + "GPUEnsemble.exe.config"}
        self.assertEqual(set(app.namelist()) - set(cli.namelist()), extra)
        self.assertEqual(set(cli.namelist()) - set(app.namelist()), set())
        self.assertIn("GPUEnsemble.exe".encode(), app.read(top + "README.txt"))
        self.assertNotIn("GPUEnsemble.exe".encode(), cli.read(top + "README.txt"))
        self.assertEqual(app.read(top + "Start GPUEnsemble.cmd"), cli.read(top + "Start GPUEnsemble.cmd"))

    def test_app_is_left_out_without_dotnet(self) -> None:
        with mock.patch.object(build_zip.shutil, "which", return_value=None):
            z = self.build("--no-app", kind="cli")
        self.assertEqual(self.made, [f"GPUEnsemble-{build_zip.package_version()}-windows-x64-cli.zip"])
        top = f"GPUEnsemble-{build_zip.package_version()}/"
        self.assertNotIn(top + "GPUEnsemble.exe", z.namelist())
        self.assertIn(top + "Start GPUEnsemble.cmd", z.namelist())

    def test_release_build_needs_the_app(self) -> None:
        out = os.path.join(self.tmp, "dist")
        with mock.patch.object(build_zip.shutil, "which", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            build_zip.main(["--python-zip", self.python_zip, "--out", out, "--require-app"])

    def test_app_is_built_with_dotnet_when_there(self) -> None:
        calls = []

        def fake_run(argv, check):
            calls.append(argv)
            out = argv[argv.index("-o") + 1]
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "GPUEnsemble.exe"), "wb") as f:
                f.write(b"MZbuilt")

        with mock.patch.object(build_zip.shutil, "which", return_value="dotnet"), \
                mock.patch.object(build_zip.subprocess, "run", side_effect=fake_run):
            z = self.build("--no-app")
        top = f"GPUEnsemble-{build_zip.package_version()}/"
        self.assertEqual(z.read(top + "GPUEnsemble.exe"), b"MZbuilt")
        [argv] = calls
        self.assertEqual(argv[:3], ["dotnet", "build", build_zip.APP_PROJECT])

    def test_search_path_includes_the_app(self) -> None:
        z = self.build()
        top = f"GPUEnsemble-{build_zip.package_version()}/"
        pth = z.read(top + "python/python312._pth").decode("ascii")
        self.assertEqual(pth.split("\r\n"), ["python312.zip", ".", "..\\app", ""])

    def test_launchers(self) -> None:
        z = self.build()
        top = f"GPUEnsemble-{build_zip.package_version()}/"
        start = z.read(top + "Start GPUEnsemble.cmd")
        lend = z.read(top + "Start GPUEnsemble worker.cmd")
        cli = z.read(top + "gpuensemble.cmd")
        for text in (start, lend, cli):
            text.decode("ascii")  # cmd.exe reads them in the console code page
            self.assertIn(b"\r\n", text)
            self.assertNotIn(b"\n", text.replace(b"\r\n", b""))
            self.assertIn(b'set "GPUENSEMBLE_HOME=%~dp0data"', text)
            self.assertIn(b'set "GPUENSEMBLE_PROG=gpuensemble.cmd"', text)
            self.assertIn(b'"%~dp0python\\python.exe" -m gpuensemble_planner', text)
        # Its own window without a batch file in it: no "Terminate batch job (Y/N)?" on Ctrl+C.
        self.assertIn(b'start "GPUEnsemble" "%~dp0python\\python.exe" -m gpuensemble_planner.launch %*', start)
        self.assertIn(b'start "GPUEnsemble worker" "%~dp0python\\python.exe" -m gpuensemble_planner.launch worker %*',
                      lend)
        self.assertIn(b"gpuensemble_planner %*", cli)
        self.assertTrue(z.read(top + "README.txt").startswith(b"\xef\xbb\xbf"))
        readme = z.read(top + "README.txt")
        # Paths keep their backslashes: "data\rpc-cache" once lost its "\r" to a carriage return.
        self.assertIn(b"data\\rpc-cache", readme)
        self.assertNotIn(b"\r", readme.replace(b"\r\n", b""))

    def test_rejects_a_zip_without_python(self) -> None:
        bad = os.path.join(self.tmp, "bad.zip")
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("readme.txt", "no python here")
        with self.assertRaises(SystemExit):
            build_zip.main(["--python-zip", bad, "--out", os.path.join(self.tmp, "dist")])

    def test_download_is_checked(self) -> None:
        cache = os.path.join(self.tmp, "cache")
        os.makedirs(cache)
        planted = os.path.join(cache, os.path.basename(build_zip.PYTHON_URL))
        with open(planted, "wb") as f:
            f.write(b"not the real file")
        with self.assertRaises(SystemExit) as e:
            build_zip.fetch_python(cache)
        self.assertIn("does not match", str(e.exception))
        self.assertFalse(os.path.exists(planted))


if __name__ == "__main__":
    unittest.main()
