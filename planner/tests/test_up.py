import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from gpuensemble_planner import cli
from gpuensemble_planner.llamabin import Install
from gpuensemble_planner.model import ModelProfile

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
RIG = os.path.join(EXAMPLES, "rig-e5-turing.json")
QWEN = ModelProfile.from_json_file(os.path.join(EXAMPLES, "qwen3-32b-q4_k_m.json"))


class UpCase(unittest.TestCase):
    """`up` with stand-ins for the download, the llama.cpp build and llama-server."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = tmp.name
        patches = [
            mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": self.home}),
            mock.patch.object(cli, "pull", side_effect=self.fake_pull),
            mock.patch.object(cli, "installed", return_value=None),
            mock.patch.object(cli, "setup", return_value=Install("b1", "cuda", "/opt/llama")),
            mock.patch.object(cli, "_load_model", return_value=QWEN),
            mock.patch.object(cli, "run", return_value=0),
        ]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.pulled: list[str] = []

    def fake_pull(self, entry, base=None):
        self.pulled.append(entry["id"])
        return os.path.join(self.home, "models", entry["id"] + ".gguf")

    def up(self, *args: str) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(["up", "--rig", RIG, "--yes", *args])
        return code, err.getvalue()


class UpTest(UpCase):
    def test_recommends_downloads_installs_and_starts(self) -> None:
        code, err = self.up("--ctx", "4096")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.pulled, ["gpt-oss-120b"])  # the recommendation for this rig
        self.assertIn("recommended: gpt-oss-120b", err)
        run = cli.run
        exe, command, path, extra = run.call_args.args
        self.assertTrue(run.call_args.kwargs["warmup"])
        self.assertEqual(exe, os.path.join("/opt/llama", "llama-server"))  # no .exe there: the build decides
        self.assertTrue(path.endswith("gpt-oss-120b.gguf"))
        self.assertIn("-c", command.args)
        self.assertEqual(extra, ["--host", "127.0.0.1", "--port", "8080", "-a", "gpt-oss-120b", "--metrics"])
        self.assertIn("chat at http://127.0.0.1:8080/", err)
        # The printed command is the one started, with the model and llama-server's own options.
        started = next(line for line in err.splitlines() if line.startswith("starting: "))
        self.assertIn(" -m ", started)
        self.assertTrue(started.endswith("--host 127.0.0.1 --port 8080 -a gpt-oss-120b --metrics"), started)

    def test_named_model_and_server_options(self) -> None:
        code, err = self.up("--model", "qwen3-32b", "--port", "9000", "--host", "0.0.0.0", "--", "--jinja")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.pulled, ["qwen3-32b"])
        extra = cli.run.call_args.args[3]
        self.assertEqual(extra, ["--host", "0.0.0.0", "--port", "9000", "-a", "qwen3-32b", "--metrics", "--jinja"])
        self.assertIn("-a qwen3-32b --metrics --jinja\n", err)
        self.assertIn("http://127.0.0.1:9000/", err)

    def test_tells_monitor_what_runs_while_it_runs(self) -> None:
        seen = []

        def fake_run(*args, **kwargs):
            with open(os.path.join(self.home, "run", "up.json"), encoding="utf-8") as f:
                seen.append(json.load(f))
            return 0

        cli.run.side_effect = fake_run
        code, _ = self.up("--model", "qwen3-32b", "--port", "9000")
        self.assertEqual(code, 0)
        info = seen[0]
        self.assertEqual((info["model"], info["host"], info["port"]), ("qwen3-32b", "127.0.0.1", 9000))
        self.assertEqual(info["pid"], os.getpid())
        self.assertTrue(info["stages"])
        self.assertGreater(info["seconds_per_token"], 0)
        self.assertFalse(os.path.exists(os.path.join(self.home, "run", "up.json")))

    def test_metrics_asked_for_by_hand_are_not_doubled(self) -> None:
        self.up("--model", "qwen3-32b", "--", "--metrics")
        self.assertEqual(cli.run.call_args.args[3].count("--metrics"), 1)

    def test_open_shows_the_chat_page_when_ready(self) -> None:
        code, err = self.up("--ctx", "4096")
        with mock.patch.object(cli.webbrowser, "open") as browser, \
                mock.patch.object(cli.monitor, "read_metrics", return_value=None):
            cli.run.call_args.kwargs["on_ready"]("127.0.0.1", 8080)
        browser.assert_not_called()
        code, err = self.up("--ctx", "4096", "--open", "--host", "0.0.0.0")
        on_ready = cli.run.call_args.kwargs["on_ready"]
        with mock.patch.object(cli.webbrowser, "open") as browser, contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(cli.monitor, "read_metrics", return_value=None):
            on_ready("0.0.0.0", 8080)
        browser.assert_called_once_with("http://127.0.0.1:8080/")

    def test_ready_tells_monitor_where_the_warm_up_ended(self) -> None:
        seen = []

        def fake_run(*args, **kwargs):
            with mock.patch.object(cli.monitor, "read_metrics",
                                   return_value={cli.monitor.TOKENS: 14.0, cli.monitor.TOKEN_SECONDS: 2.0}) as read:
                kwargs["on_ready"]("0.0.0.0", 9000)
            read.assert_called_once_with("127.0.0.1", 9000)
            with open(os.path.join(self.home, "run", "up.json"), encoding="utf-8") as f:
                seen.append(json.load(f))
            return 0

        cli.run.side_effect = fake_run
        self.up("--model", "qwen3-32b", "--host", "0.0.0.0", "--port", "9000")
        self.assertEqual(seen[0]["warm"][cli.monitor.TOKENS], 14.0)
        self.assertGreater(seen[0]["ready"], 0)

    def test_checks_the_rpc_server_before_starting(self) -> None:
        rig = os.path.join(EXAMPLES, "rig-e5-rpc-loopback.json")
        with mock.patch.object(cli.worker, "unreachable_servers", return_value=["127.0.0.1:50052"]) as check:
            code, err = self.up("--model", "qwen3-32b", "--rig", rig)
        self.assertEqual(code, 1)
        check.assert_called_once_with(["127.0.0.1:50052"])
        cli.run.assert_not_called()
        self.assertIn("no RPC server answers at 127.0.0.1:50052", err)
        self.assertIn("worker --host", err)
        with mock.patch.object(cli.worker, "unreachable_servers", return_value=[]):
            code, err = self.up("--model", "qwen3-32b", "--rig", rig)
        self.assertEqual(code, 0, err)
        command = cli.run.call_args.args[1]
        self.assertEqual(command.rpc_servers, ["127.0.0.1:50052"])
        self.assertIn("RPC0", command.args[command.args.index("-dev") + 1])

    def test_chats_are_not_kept_in_ram_when_layers_are_on_another_pc(self) -> None:
        # The server would read each finished chat's KV cache back from the cards: over a network, a download.
        remote = os.path.join(self.home, "remote.json")
        with open(remote, "w", encoding="utf-8") as f:
            f.write('{"gpus": [{"name": "RTX 4090", "vram_gib": 24, "rpc": "100.64.0.2:50052", "rtt_ms": 7.6}]}')
        with mock.patch.object(cli.worker, "unreachable_servers", return_value=[]):
            code, err = self.up("--model", "qwen3-32b", "--remote", remote)
            self.assertEqual(code, 0, err)
            command, _, extra = cli.run.call_args.args[1:]
            self.assertEqual(command.rpc_servers, ["100.64.0.2:50052"])
            self.assertEqual(extra, ["--host", "127.0.0.1", "--port", "8080", "-a", "qwen3-32b", "--cache-ram", "0", "--metrics"])
            self.assertIn("note: finished chats are not kept in RAM (--cache-ram 0)", err)
            # The user's own choice stands.
            code, err = self.up("--model", "qwen3-32b", "--remote", remote, "--", "--cache-ram", "4096")
            self.assertEqual(cli.run.call_args.args[3][-2:], ["--cache-ram", "4096"])
            self.assertNotIn("--cache-ram 0", err)
            # A server on this PC is read from as fast as a card of its own.
            code, err = self.up("--model", "qwen3-32b", "--rig", os.path.join(EXAMPLES, "rig-e5-rpc-loopback.json"))
            self.assertEqual(code, 0, err)
            self.assertNotIn("--cache-ram", cli.run.call_args.args[3])
        # Nor without another PC.
        code, err = self.up("--model", "qwen3-32b")
        self.assertNotIn("--cache-ram", cli.run.call_args.args[3])

    def test_waits_for_a_worker_only_at_a_console(self) -> None:
        local = ["--model", "C:/models/Qwen3-32B-Q4_K_M.gguf", "--exe", "llama-server", "--remote", "auto"]
        for yes, tty, wait in ((False, True, cli.APPROVAL_WAIT), (True, True, 0.0), (False, False, 0.0)):
            stdin = mock.Mock(isatty=mock.Mock(return_value=tty))
            with mock.patch.object(cli, "found_remote", return_value=[]) as found, \
                    mock.patch.object(cli.sys, "stdin", stdin), contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["up", "--rig", RIG, *local] + (["--yes"] if yes else []))
            self.assertEqual(code, 0)
            found.assert_called_once_with("auto", wait)

    def test_local_file_needs_no_download(self) -> None:
        code, err = self.up("--model", "C:/models/Qwen3-32B-Q4_K_M.gguf", "--exe", "llama-server")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.pulled, [])
        cli.setup.assert_not_called()
        self.assertEqual(cli.run.call_args.args[3][-2], "Qwen3-32B-Q4_K_M")

    def test_unknown_model(self) -> None:
        code, err = self.up("--model", "no-such-model")
        self.assertEqual(code, 1)
        self.assertIn("not in the catalog", err)
        cli.run.assert_not_called()


class FakeRelay:
    """Stands where the API is expected, without listening."""

    def __init__(self, listen, target, log) -> None:
        self.listen, self.target = listen, target
        self.started = self.closed = False

    def start(self):
        self.started = True
        return self

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, port: int = 50054, code: int = 0, wait=None) -> None:
        self.port, self.code, self.closed = port, code, False
        self._wait = wait

    def wait(self, on_log) -> int:
        on_log("srv  update_slots: all slots are idle")
        if self._wait is not None:
            self._wait()
        return self.code

    def close(self) -> None:
        self.closed = True


class ThereTest(UpCase):
    """A model that fits on one other PC is started there, and its API passed on (``serving``)."""

    def setUp(self) -> None:
        super().setUp()
        self.session = FakeSession()
        self.relays: list[FakeRelay] = []
        self.refusal: OSError | None = None
        self.asked: list[tuple] = []

        def relay(listen, target, log):
            self.relays.append(FakeRelay(listen, target, log))
            return self.relays[-1]

        def ask(address, port, model_id, ctx, on_log):
            self.asked.append((address, port, model_id, ctx))
            on_log("loaded in 14 s")
            if self.refusal is not None:
                raise self.refusal
            return self.session

        for p in (mock.patch.object(cli.serving, "Relay", side_effect=relay),
                  mock.patch.object(cli.serving, "ask", side_effect=ask),
                  mock.patch.object(cli.worker, "unreachable_servers", return_value=[])):
            p.start()
            self.addCleanup(p.stop)
        self.serves = self.remote(serve=True)
        self.lends = self.remote(serve=False)

    def remote(self, serve: bool) -> str:
        path = os.path.join(self.home, f"remote-{serve}.json")
        gpu = {"name": "RTX 4090", "vram_gib": 24, "rpc": "100.64.0.2:50052", "rtt_ms": 7.6, "gate": True}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"gpus": [dict(gpu, serve=True) if serve else gpu]}, f)
        return path

    def test_starts_on_the_pc_that_holds_all_of_it(self) -> None:
        code, err = self.up("--model", "qwen3-32b", "--remote", self.serves)
        self.assertEqual(code, 0, err)
        # Asked on the worker's request port, for the catalog id and the context and nothing else.
        self.assertEqual(self.asked, [("100.64.0.2", 50053, "qwen3-32b", 8192)])
        # This PC downloads nothing, installs nothing and starts nothing.
        self.assertEqual(self.pulled, [])
        cli.setup.assert_not_called()
        cli.run.assert_not_called()
        # The API is where the user expects it, passed on to the other PC's gate.
        [relay] = self.relays
        self.assertEqual((relay.listen, relay.target), (("127.0.0.1", 8080), ("100.64.0.2", 50054)))
        self.assertTrue(relay.started and relay.closed and self.session.closed)
        self.assertIn("starting it there", err)
        self.assertIn("[100.64.0.2] loaded in 14 s", err)
        self.assertIn("[100.64.0.2] srv  update_slots", err)
        ready = [line for line in err.splitlines() if line.startswith("ready: ")]
        self.assertEqual(len(ready), 1, err)
        self.assertIn("http://127.0.0.1:8080/v1 (the model runs on the PC at 100.64.0.2", ready[0])

    def test_the_port_and_address_asked_for(self) -> None:
        self.session = FakeSession(port=50064, code=3)
        with mock.patch.object(cli.webbrowser, "open") as browser:
            code, err = self.up("--model", "qwen3-32b", "--remote", self.serves, "--host", "0.0.0.0",
                                "--port", "9000", "--ctx", "16384", "--open")
        self.assertEqual(code, 3)  # llama-server's exit code on the other PC
        self.assertEqual(self.asked, [("100.64.0.2", 50053, "qwen3-32b", 16384)])
        [relay] = self.relays
        self.assertEqual((relay.listen, relay.target), (("0.0.0.0", 9000), ("100.64.0.2", 50064)))
        browser.assert_called_once_with("http://127.0.0.1:9000/")

    def test_a_pc_that_only_lends_its_cards_is_used_as_before(self) -> None:
        code, err = self.up("--model", "qwen3-32b", "--remote", self.lends)
        self.assertEqual(code, 0, err)
        self.assertEqual((self.asked, self.relays), ([], []))
        self.assertEqual(cli.run.call_args.args[1].rpc_servers, ["100.64.0.2:50052"])
        self.assertIn("started on that PC itself", err)  # the hint stays

    def test_a_model_spread_over_both_pcs_starts_here(self) -> None:
        code, err = self.up("--model", "gpt-oss-120b", "--remote", self.serves)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.asked, [])
        self.assertEqual(cli.run.call_args.args[1].rpc_servers, ["100.64.0.2:50052"])

    def test_a_pc_that_will_not_is_used_as_before(self) -> None:
        self.refusal = OSError("this PC is already running qwen3-14b for the PC at 100.64.0.9")
        code, err = self.up("--model", "qwen3-32b", "--remote", self.serves)
        self.assertEqual(code, 0, err)
        self.assertIn("note: the other PC did not start it (this PC is already running qwen3-14b", err)
        [relay] = self.relays
        self.assertTrue(relay.closed and not relay.started)  # the port is free again for llama-server
        self.assertEqual(self.pulled, ["qwen3-32b"])
        self.assertEqual(cli.run.call_args.args[1].rpc_servers, ["100.64.0.2:50052"])

    def test_run_here(self) -> None:
        code, err = self.up("--model", "qwen3-32b", "--remote", self.serves, "--run", "here")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.asked, [])
        self.assertEqual(cli.run.call_args.args[1].rpc_servers, ["100.64.0.2:50052"])

    def test_what_cannot_be_passed_on_starts_it_here(self) -> None:
        for args, kept in ((["--", "--jinja"], "arguments for llama-server"), (["--kv", "q8_0"], "--kv"),
                           (["--exe", "llama-server"], "--exe")):
            with self.subTest(kept=kept):
                code, err = self.up("--model", "qwen3-32b", "--remote", self.serves, *args)
                self.assertEqual(code, 0, err)
                self.assertEqual(self.asked, [])
                self.assertIn(f"but {kept} cannot be passed on to it; starting it on this PC", err)
                cli.run.assert_called()
                code, err = self.up("--model", "qwen3-32b", "--remote", self.serves, "--run", "there", *args)
                self.assertEqual(code, 1)
                self.assertIn(f"error: --run there cannot pass {kept} on to the other PC", err)

    def test_run_there_does_not_fall_back(self) -> None:
        code, err = self.up("--model", "qwen3-32b", "--remote", self.lends, "--run", "there")
        self.assertEqual(code, 1)
        self.assertIn("error: --run there: Qwen3-32B", err)
        code, err = self.up("--model", "gpt-oss-120b", "--remote", self.serves, "--run", "there")
        self.assertEqual(code, 1)
        self.refusal = OSError("it said nothing for 300 s")
        code, err = self.up("--model", "qwen3-32b", "--remote", self.serves, "--run", "there")
        self.assertEqual(code, 1)
        self.assertIn("error: the other PC did not start it (it said nothing for 300 s)", err)
        cli.run.assert_not_called()
        self.assertEqual(self.pulled, [])

    def test_a_port_in_use_is_said_before_the_other_pc_is_asked(self) -> None:
        with mock.patch.object(cli.serving, "Relay", side_effect=OSError("address already in use")):
            code, err = self.up("--model", "qwen3-32b", "--remote", self.serves)
        self.assertEqual(code, 1)
        self.assertIn("cannot listen on 127.0.0.1:8080", err)
        self.assertEqual(self.asked, [])

    def test_ctrl_c_stops_it_there(self) -> None:
        def interrupted():
            raise KeyboardInterrupt

        self.session = FakeSession(wait=interrupted)
        code, err = self.up("--model", "qwen3-32b", "--remote", self.serves)
        self.assertEqual(code, 130)
        self.assertTrue(self.relays[0].closed and self.session.closed)  # closing the session stops the model there
        self.refusal = KeyboardInterrupt()  # while the other PC is still loading
        code, err = self.up("--model", "qwen3-32b", "--remote", self.serves)
        self.assertEqual(code, 130)
        self.assertTrue(self.relays[1].closed)

    def test_the_recommended_model_may_be_one_that_runs_there(self) -> None:
        code, err = self.up("--remote", self.serves, "--min-speed", "25")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.asked, [("100.64.0.2", 50053, "qwen3.8-27b", 8192)])
        self.assertIn("recommended: Qwen3.8-27B", err)
        [row] = [line for line in err.splitlines() if line.lstrip().startswith("* ")]
        self.assertIn("runs on the other PC", row)


class CatalogIdTest(unittest.TestCase):
    """plan, args, describe and run take a catalog id where they take a file."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = tmp.name
        patch = mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": self.home})
        patch.start()
        self.addCleanup(patch.stop)

    def cli(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_plan_with_a_model_that_is_not_downloaded(self) -> None:
        code, out, err = self.cli("plan", "qwen3-32b", "--rig", RIG)
        self.assertEqual(code, 0, err)
        self.assertIn("estimated generation speed", out)
        self.assertIn("not downloaded; using the catalog's description of it", err)
        self.assertIn("Qwen3-32B", out)
        code, out, err = self.cli("args", "qwen3-32b", "--rig", RIG)
        self.assertEqual(code, 0, err)
        self.assertIn("llama-server", out)

    def test_plan_shows_what_up_runs(self) -> None:
        # gpt-oss-120b: llama.cpp keeps a layer's experts on the GPU or in RAM as a whole. `plan` once
        # split them inside a layer, and so showed another placement and speed than `args`, `up` and the app.
        model = os.path.join(EXAMPLES, "gpt-oss-120b.json")

        def stages(text: str) -> list[str]:
            return [line for line in text.splitlines() if "experts on GPU" in line]

        code, plan, err = self.cli("plan", model, "--rig", RIG)
        self.assertEqual(code, 0, err)
        code, args, err = self.cli("args", model, "--rig", RIG)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(stages(plan)), 2)
        self.assertEqual(stages(plan), stages(args))
        code, split, err = self.cli("plan", model, "--rig", RIG, "--split-experts")
        self.assertEqual(code, 0, err)
        self.assertNotEqual(stages(split), stages(plan))

    def test_plan_with_the_downloaded_file(self) -> None:
        for path in (os.path.join(self.home, "models", "Qwen3-32B-Q4_K_M.gguf"),
                     "Z:\\AI\\GPUEnsemble\\data\\models\\Qwen3-32B-Q4_K_M.gguf"):
            with self.subTest(path=path):
                self.plan_and_run(path)

    def plan_and_run(self, path: str) -> None:
        with mock.patch.object(cli, "find_local", return_value=path), \
                mock.patch.object(cli, "_load_model", return_value=QWEN) as load:
            code, out, err = self.cli("plan", "qwen3-32b", "--rig", RIG)
            self.assertEqual(code, 0, err)
            load.assert_called_once_with(path)
            self.assertNotIn("not downloaded", err)
            with mock.patch.object(cli, "run", return_value=0) as run:
                code, _, err = self.cli("run", "qwen3-32b", "--rig", RIG, "--exe", "llama-server")
            self.assertEqual(code, 0, err)
            self.assertEqual(run.call_args.args[2], path)
            # The printed command names the file (in quotes where the shell needs them, as for a Windows path).
            started = next(line for line in err.splitlines() if line.startswith("starting: "))
            self.assertIn(" -m ", started)
            self.assertIn(path, started)

    def test_run_needs_the_file(self) -> None:
        with mock.patch.object(cli, "run", return_value=0) as run:
            code, _, err = self.cli("run", "qwen3-32b", "--rig", RIG, "--exe", "llama-server")
        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertIn("qwen3-32b is not downloaded yet", err)
        self.assertIn("pull qwen3-32b", err)

    def test_neither_a_file_nor_in_the_catalog(self) -> None:
        code, _, err = self.cli("plan", "no-such-model", "--rig", RIG)
        self.assertEqual(code, 1)
        self.assertIn("there is no file no-such-model, and no-such-model is not in the catalog; choose one of:", err)
        # A file name that does not exist is still reported as a missing file.
        code, _, err = self.cli("plan", "missing.gguf", "--rig", RIG)
        self.assertEqual(code, 1)
        self.assertNotIn("catalog", err)


if __name__ == "__main__":
    unittest.main()
