import contextlib
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

from gpuensemble_planner import llamabin
from gpuensemble_planner.llamabin import (Asset, add_local_runtime, choose_backend, installed, pick_assets,
                                          setup)

TAG = "b12375"
LIST_DEVICES = """Available devices:
  CUDA0: NVIDIA GeForce RTX 2080 Ti (11263 MiB, 10215 MiB free)
  CUDA1: NVIDIA GeForce RTX 2080 SUPER (8191 MiB, 7987 MiB free)
"""


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeResponse(io.BytesIO):
    status = 200
    headers: dict = {}


class FakeGitHub:
    def __init__(self, archives: dict[str, bytes], tag: str = TAG):
        self.archives = archives
        self.tag = tag
        self.urls: list[str] = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.urls.append(url)
        if "/releases/" in url and "api.github.com" in url:
            assets = [{"name": n, "size": len(d), "browser_download_url": f"https://github.com/dl/{n}",
                       "digest": "sha256:" + hashlib.sha256(d).hexdigest()} for n, d in self.archives.items()]
            assets.append({"name": "llama-b12375-xcframework.zip", "size": 1, "browser_download_url": "x"})
            return FakeResponse(json.dumps({"tag_name": self.tag, "assets": assets}).encode())
        return FakeResponse(self.archives[url.rsplit("/", 1)[1]])


class FakeRun:
    """Stands in for subprocess.run: nvidia-smi and llama-server --list-devices."""

    def __init__(self, driver: str | None = "560.94", codes: list[int] | None = None, devices: str = LIST_DEVICES):
        self.driver = driver
        self.codes = codes or [0]
        self.devices = devices
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[0] == "nvidia-smi":
            if self.driver is None:
                raise FileNotFoundError("nvidia-smi")
            return subprocess.CompletedProcess(argv, 0, self.driver + "\n", "")
        code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        return subprocess.CompletedProcess(argv, code, self.devices if code == 0 else "", "")


def asset(name: str) -> Asset:
    return Asset(name, "u", 1)


RELEASE = [asset(n) for n in (
    f"llama-{TAG}-bin-win-cuda-12.4-x64.zip", f"llama-{TAG}-bin-win-cuda-13.1-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-13.1-x64.zip",
    f"llama-{TAG}-bin-win-vulkan-x64.zip", f"llama-{TAG}-bin-win-cpu-x64.zip",
    f"llama-{TAG}-bin-ubuntu-x64.tar.gz", f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz",
    f"llama-{TAG}-bin-macos-arm64.tar.gz")]


class ChoiceTest(unittest.TestCase):
    def test_backend(self) -> None:
        self.assertEqual(choose_backend("win32", (560, 94))[0], "cuda")
        self.assertEqual(choose_backend("win32", (551, 61))[0], "cuda")
        self.assertEqual(choose_backend("win32", (528, 49))[0], "vulkan")  # the E5 before Windows Update
        self.assertIn("update the driver", choose_backend("win32", (528, 49))[1])
        self.assertEqual(choose_backend("linux", (560, 35, 3))[0], "vulkan")
        self.assertEqual(choose_backend("win32", None)[0], "cpu")

    def test_assets(self) -> None:
        names = lambda *a: [x.name for x in pick_assets(RELEASE, TAG, *a)]  # noqa: E731
        self.assertEqual(names("win32", "cuda"), [f"llama-{TAG}-bin-win-cuda-12.4-x64.zip",
                                                  "cudart-llama-bin-win-cuda-12.4-x64.zip"])
        self.assertEqual(names("win32", "vulkan"), [f"llama-{TAG}-bin-win-vulkan-x64.zip"])
        self.assertEqual(names("win32", "cpu"), [f"llama-{TAG}-bin-win-cpu-x64.zip"])
        self.assertEqual(names("linux", "vulkan"), [f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz"])
        self.assertEqual(names("linux", "cpu"), [f"llama-{TAG}-bin-ubuntu-x64.tar.gz"])
        with self.assertRaises(LookupError):
            pick_assets(RELEASE[:1], TAG, "win32", "cuda")  # no cudart

    def test_driver_version(self) -> None:
        self.assertEqual(llamabin.nvidia_driver(FakeRun("560.94")), (560, 94))
        self.assertIsNone(llamabin.nvidia_driver(FakeRun(None)))


class SetupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "llama.cpp")
        self.quiet = []
        quiet = contextlib.ExitStack()
        quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))  # download progress
        self.addCleanup(quiet.close)

    def log(self, s: str) -> None:
        self.quiet.append(s)

    def test_linux_vulkan_install_and_reuse(self) -> None:
        name = f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz"
        gh = FakeGitHub({name: tar_bytes({"build/bin/llama-server": b"#!/bin/sh\n", "build/bin/libggml.so": b"x"})})
        run = FakeRun()
        inst = setup(root=self.root, opener=gh, run=run, platform="linux", log=self.log)
        self.assertEqual(inst.backend, "vulkan")
        self.assertTrue(inst.folder.endswith(os.path.join(f"{TAG}-vulkan", "build", "bin")))
        self.assertTrue(os.access(inst.server, os.X_OK))
        self.assertEqual(inst.devices[0], "CUDA0: NVIDIA GeForce RTX 2080 Ti (11263 MiB, 10215 MiB free)")
        self.assertEqual(run.calls[-1], [inst.server, "--list-devices"])
        self.assertFalse(os.path.exists(os.path.join(self.root, "downloads", name)))  # archive removed
        self.assertEqual(installed(self.root), inst)
        n = len(gh.urls)
        again = setup(root=self.root, opener=gh, run=run, platform="linux", log=self.log)
        self.assertEqual(again.folder, inst.folder)
        self.assertEqual(len(gh.urls), n)  # nothing downloaded, no API call

    def test_windows_cuda_puts_the_runtime_next_to_the_server(self) -> None:
        exe = "llama-server.exe"
        gh = FakeGitHub({
            f"llama-{TAG}-bin-win-cuda-12.4-x64.zip": zip_bytes({exe: b"MZ", "ggml-cuda.dll": b"d"}),
            "cudart-llama-bin-win-cuda-12.4-x64.zip": zip_bytes({"cudart64_12.dll": b"c", "cublas64_12.dll": b"b"}),
        })
        inst = setup(root=self.root, opener=gh, run=FakeRun("560.94"), platform="win32", log=self.log)
        self.assertEqual(inst.backend, "cuda")
        self.assertEqual(inst.server, os.path.join(inst.folder, "llama-server.exe"))
        self.assertNotIn(os.sep + "." + os.sep, inst.server)
        self.assertEqual(sorted(os.listdir(inst.folder)), sorted([exe, "ggml-cuda.dll", "cudart64_12.dll", "cublas64_12.dll"]))

    def test_old_visual_cpp_runtime_is_fixed_next_to_llama_cpp(self) -> None:
        exe = "llama-server.exe"
        gh = FakeGitHub({f"llama-{TAG}-bin-win-vulkan-x64.zip": zip_bytes({exe: b"MZ"})})
        run = FakeRun("528.49", codes=[0xC0000005, 0])
        copied = []
        with mock.patch.object(llamabin, "add_local_runtime", lambda folder: copied.append(folder) or r"C:\Edge\131"), \
                mock.patch.object(llamabin, "system_runtime_version", lambda: (14, 28, 29914, 0)):
            inst = setup(root=self.root, opener=gh, run=run, platform="win32", log=self.log)
        self.assertEqual(copied, [inst.folder])
        self.assertIn("Visual C++ runtime is 14.28.29914.0", "\n".join(self.quiet))
        self.assertEqual(len(inst.devices), 2)

    def test_crash_without_a_newer_runtime_explains_what_to_install(self) -> None:
        exe = "llama-server.exe"
        gh = FakeGitHub({f"llama-{TAG}-bin-win-cpu-x64.zip": zip_bytes({exe: b"MZ"})})
        with mock.patch.object(llamabin, "add_local_runtime", lambda folder: None), \
                mock.patch.object(llamabin, "system_runtime_version", lambda: None), \
                self.assertRaisesRegex(OSError, "vc_redist"):
            setup(root=self.root, opener=gh, run=FakeRun(None, codes=[-1073741819]), platform="win32", log=self.log)
        self.assertIsNone(installed(self.root))

    def test_gpu_build_that_sees_no_gpu_is_an_error(self) -> None:
        name = f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz"
        gh = FakeGitHub({name: tar_bytes({"llama-server": b"x"})})
        with self.assertRaisesRegex(OSError, "sees no GPU"):
            setup(root=self.root, opener=gh, run=FakeRun(devices="Available devices:\n"), platform="linux", log=self.log)


class HostIndependenceTest(SetupTest):
    """The same installs, run as if this computer were the other OS."""

    def run(self, result=None):
        with mock.patch.object(llamabin, "_windows", lambda: not llamabin.sys.platform == "win32"):
            return super().run(result)


class RuntimeTest(unittest.TestCase):
    def test_copies_the_newest_complete_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            versions = {}
            for folder, v in (("edge_old", (14, 28, 0, 0)), ("edge_new", (14, 51, 36000, 0)), ("partial", (14, 60, 0, 0))):
                os.makedirs(os.path.join(tmp, folder))
                names = llamabin.VC_RUNTIME if folder != "partial" else llamabin.VC_RUNTIME[:1]
                for n in names:
                    path = os.path.join(tmp, folder, n)
                    with open(path, "w") as f:
                        f.write(folder)
                    versions[path] = v
            dest = os.path.join(tmp, "llama")
            os.makedirs(dest)
            candidates = [os.path.join(tmp, f) for f in ("edge_old", "edge_new", "partial")]
            source = add_local_runtime(dest, candidates, version=versions.get)
            self.assertEqual(source, os.path.join(tmp, "edge_new"))
            for n in llamabin.VC_RUNTIME:
                with open(os.path.join(dest, n)) as f:
                    self.assertEqual(f.read(), "edge_new")
            self.assertIsNone(add_local_runtime(dest, candidates[:1], version=versions.get))


class RunDefaultTest(unittest.TestCase):
    def test_run_without_exe_and_without_setup(self) -> None:
        from gpuensemble_planner.cli import main
        rig = os.path.join(os.path.dirname(__file__), "..", "examples", "rig-e5-turing.json")
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": home}):
            gguf = os.path.join(home, "m.gguf")
            err = io.StringIO()
            with mock.patch("gpuensemble_planner.cli._load_model") as load, contextlib.redirect_stderr(err):
                from gpuensemble_planner.model import ModelProfile
                load.return_value = ModelProfile.from_json_file(
                    os.path.join(os.path.dirname(rig), "qwen3-32b-q4_k_m.json"))
                self.assertEqual(main(["run", gguf, "--rig", rig]), 1)
            self.assertIn("gpuensemble-plan setup", err.getvalue())


if __name__ == "__main__":
    unittest.main()
