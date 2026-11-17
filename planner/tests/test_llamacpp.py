import contextlib
import io
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, HTTPServer

from gpuensemble_planner.devices import GIB, Device, Host, load_rig
from gpuensemble_planner.llamacpp import (LineLog, LlamaCommand, Stopped, _server_address, interruptible,
                                          llama_command, run, wait_ready, warm_up)
from gpuensemble_planner.model import LayerProfile, ModelProfile
from gpuensemble_planner.plan import make_plan

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def devices() -> list[Device]:
    return [
        Device("ti", int(11 * GIB), 463e9, llama_device="CUDA0"),
        Device("super", int(8 * GIB), 362e9, llama_device="CUDA1"),
    ]


HOST = Host(int(52 * GIB), 34e9, 13e9, expert_path="cpu", cpu_overhead=0.075)


def arg(cmd, flag):
    return cmd.args[cmd.args.index(flag) + 1]


class LlamaCommandTest(unittest.TestCase):
    def test_dense_that_fits_offloads_everything(self) -> None:
        model = ModelProfile("d", "llama", [LayerProfile(int(0.25 * GIB), 0, 4096)] * 40, int(0.5 * GIB), int(0.3 * GIB))
        plan = make_plan(model, devices(), HOST, context=4096)
        cmd = llama_command(plan, model, devices())
        self.assertEqual(arg(cmd, "-ngl"), "999")
        stages = [s for s in plan.stages if s.n_layers]
        self.assertEqual(arg(cmd, "-dev").split(","), [{"ti": "CUDA0", "super": "CUDA1"}[s.device] for s in stages])
        if len(stages) > 1:
            # The output head counts as one more layer on the last device.
            self.assertEqual(sum(int(x) for x in arg(cmd, "-ts").split(",")), 41)
        self.assertNotIn("-ot", cmd.args)
        self.assertEqual(cmd.env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_cpu_prefix_sets_ngl_and_exact_split(self) -> None:
        model = ModelProfile("d", "llama", [LayerProfile(int(0.3 * GIB), 0, 4096)] * 64, int(0.6 * GIB), int(0.4 * GIB))
        plan = make_plan(model, devices(), HOST, context=640)
        self.assertGreater(plan.cpu_layers, 0)
        cmd = llama_command(plan, model, devices(), tool="bench")
        self.assertEqual(int(arg(cmd, "-ngl")), 64 - plan.cpu_layers + 1)
        ts = [int(x) for x in arg(cmd, "-ts").split("/")]
        self.assertEqual(sum(ts), 64 - plan.cpu_layers + 1)
        self.assertEqual(ts[-1], plan.stages[-1].n_layers + 1)
        self.assertIn("/", arg(cmd, "-dev"))

    def test_moe_pins_whole_expert_layers_to_cpu(self) -> None:
        model = ModelProfile(
            "m", "moe", [LayerProfile(int(0.05 * GIB), int(1.577 * GIB), 2048) for _ in range(36)],
            int(1.1 * GIB), int(1.1 * GIB), n_expert=128, n_expert_used=4,
        )
        plan = make_plan(model, devices(), HOST, context=640)
        self.assertEqual(plan.mode, "moe")
        cmd = llama_command(plan, model, devices())
        self.assertEqual(arg(cmd, "-ngl"), "999")
        pattern, target = arg(cmd, "-ot").rsplit("=", 1)
        self.assertEqual(target, "CPU")
        on_cpu = {int(x) for x in re.search(r"\(([\d|]+)\)", pattern).group(1).split("|")}
        self.assertEqual(sorted(on_cpu), cmd.cpu_expert_layers)
        # Whatever stays on a GPU fits that GPU's budget from the plan.
        for s in plan.stages:
            kept = sum(model.layers[i].expert_bytes for i in range(s.first_layer, s.last_layer + 1) if i not in on_cpu)
            total = sum(model.layers[i].expert_bytes for i in range(s.first_layer, s.last_layer + 1))
            self.assertLessEqual(kept, s.expert_fraction_on_gpu * total + 1)
        self.assertTrue(re.fullmatch(pattern, "blk.%d.ffn_up_exps." % min(on_cpu)) or re.match(pattern, "blk.%d.ffn_up_exps.weight" % min(on_cpu)))

    def test_e5_gpt_oss_matches_hand_tuned_best(self) -> None:
        # Phase 0: the best llama.cpp run kept experts of 28 layers on the CPU (-ncmoe 28).
        devs, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
        host.expert_path = "cpu"
        model = ModelProfile.from_json_file(os.path.join(EXAMPLES, "gpt-oss-120b.json"))
        plan = make_plan(model, devs, host, context=640)
        cmd = llama_command(plan, model, devs, tool="bench")
        self.assertIn(len(cmd.cpu_expert_layers), range(26, 31))

    def test_e5_qwen3_32b_puts_fast_link_first(self) -> None:
        # Phase 1a on the E5 rig: with layers on the CPU, prompt batches copy
        # their weights over the first GPU's link. Ti first (Gen2 x8) gave pp
        # 278 vs 199 with the SUPER first (Gen2 x4), at about the same tg.
        devs, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
        host.expert_path = "cpu"
        model = ModelProfile.from_json_file(os.path.join(EXAMPLES, "qwen3-32b-q4_k_m.json"))
        plan = make_plan(model, devs, host, context=640)
        cmd = llama_command(plan, model, devs, tool="bench")
        # Verified on the rig: -ngl 60 -dev CUDA0/CUDA1 -ts 36/24 fits at ctx 640, pp 295 / tg 7.89
        # (-t 32). It runs out of memory on the SUPER at ctx 1152, where the plan changes.
        self.assertEqual(cmd.args[:8], ["-ngl", "60", "-sm", "layer", "-dev", "CUDA0/CUDA1", "-ts", "36/24"])
        longer = llama_command(make_plan(model, devs, host, context=1152), model, devs, tool="bench")
        self.assertEqual(longer.args[6:8], ["-ts", "36/23"])  # SUPER: 22 layers + head, verified to fit
        self.assertNotIn("-nopo", cmd.args)  # op offload helps dense models

    def test_moe_with_cpu_experts_turns_op_offload_off(self) -> None:
        # Phase 1a: gpt-oss-120b pp 16.3 -> 42.2 and tg 7.15 -> 8.08 with -nopo 1.
        devs, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
        host.expert_path = "cpu"
        model = ModelProfile.from_json_file(os.path.join(EXAMPLES, "gpt-oss-120b.json"))
        plan = make_plan(model, devs, host, context=640)
        self.assertEqual(llama_command(plan, model, devs, tool="bench").args[-2:], ["-nopo", "1"])
        server = llama_command(plan, model, devs, threads=host.threads)
        self.assertIn("--no-op-offload", server.args)
        # llama.cpp's default of 16 threads halved the server's speed on the rig.
        self.assertEqual(arg(server, "-t"), "32")

    def test_shell_lines(self) -> None:
        model = ModelProfile("d", "llama", [LayerProfile(int(0.25 * GIB), 0, 4096)] * 8, int(0.5 * GIB), int(0.3 * GIB))
        cmd = llama_command(make_plan(model, devices(), HOST, context=1024), model, devices())
        self.assertTrue(cmd.shell("llama-server").startswith("CUDA_DEVICE_ORDER=PCI_BUS_ID llama-server -ngl 999"))
        self.assertIn("$env:CUDA_DEVICE_ORDER='PCI_BUS_ID'; & ", cmd.shell(r"C:\llm\bin\cuda\llama-server.exe", windows=True))


class KvCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.devs, self.host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
        self.host.expert_path = "cpu"
        self.model = ModelProfile.from_json_file(os.path.join(EXAMPLES, "qwen3-32b-q4_k_m.json"))

    def test_short_context_keeps_f16(self) -> None:
        plan = make_plan(self.model, self.devs, self.host, context=640)
        self.assertEqual(plan.kv_type, "f16")
        self.assertNotIn("-ctk", llama_command(plan, self.model, self.devs).args)

    def test_long_context_quantizes_kv_to_keep_layers_on_gpu(self) -> None:
        # E5 rig: at 8192 tokens an f16 cache pushed 12 layers of Qwen3-32B to the CPU (4.3 tok/s).
        f16 = make_plan(self.model, self.devs, self.host, context=8192, kv_type="f16")
        auto = make_plan(self.model, self.devs, self.host, context=8192)
        self.assertEqual(auto.kv_type, "q8_0")
        self.assertLess(auto.cpu_layers, f16.cpu_layers)
        cmd = llama_command(auto, self.model, self.devs)
        self.assertEqual(arg(cmd, "-ctk"), "q8_0")
        self.assertEqual(arg(cmd, "-ctv"), "q8_0")

    def test_short_context_keeps_f16_even_if_a_layer_squeezes_in(self) -> None:
        # At ctx 640 q8_0 saves ~80 MB, a quarter of a layer. Find a little more
        # VRAM where that is just enough for one more layer: the plan keeps f16.
        from dataclasses import replace
        for step in range(1, 61):
            devs = [replace(d, vram_bytes=d.vram_bytes + step * 5 * 1024**2) for d in self.devs]
            f16 = make_plan(self.model, devs, self.host, context=640, kv_type="f16")
            q8 = make_plan(self.model, devs, self.host, context=640, kv_type="q8_0")
            if q8.cpu_layers < f16.cpu_layers:
                break
        else:
            self.fail("no VRAM size where q8_0 squeezes in a layer")
        plan = make_plan(self.model, devs, self.host, context=640)
        self.assertEqual(plan.kv_type, "f16")

    def test_forced_type(self) -> None:
        plan = make_plan(self.model, self.devs, self.host, context=8192, kv_type="f16")
        self.assertEqual(plan.kv_type, "f16")


class FakeServer(BaseHTTPRequestHandler):
    ready_after = 2  # /health answers 503 this many times first
    calls: list[str] = []

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        FakeServer.calls.append("health")
        code = 503 if FakeServer.calls.count("health") <= FakeServer.ready_after else 200
        self.send_response(code)
        self.end_headers()

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeServer.calls.append(f"{self.path} {len(body['prompt'].split())} {body['n_predict']}")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"content": "hi"}')


# Stands in for llama-server: answers /health and /completion, writes a log line
# in two pieces, and exits after the eight warm-up requests.
FAKE_LLAMA_SERVER = """#!PYTHON
import os, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
served = []
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
        served.append(1)
        print("served /completion", len(served), flush=True)
        if len(served) == 8:
            threading.Timer(0.3, lambda: os._exit(0)).start()
sys.stdout.write("0.00.000.001 I "); sys.stdout.flush(); time.sleep(0.05)
print("server starting on a line written in two pieces", flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


class FakeProc:
    def __init__(self, code):
        self.returncode = code

    def poll(self):
        return self.returncode


class InterruptibleTest(unittest.TestCase):
    def test_returns_the_value(self) -> None:
        self.assertEqual(interruptible(lambda a, b: a + b, 2, 3), 5)

    def test_raises_the_error(self) -> None:
        def fail() -> None:
            raise OSError("refused")
        with self.assertRaisesRegex(OSError, "refused"):
            interruptible(fail)

    def test_waits_in_short_steps(self) -> None:
        # The caller's thread polls instead of blocking on the call, which is
        # what lets Ctrl+C reach it on Windows.
        release = threading.Event()
        steps: list[float] = []
        real_join = threading.Thread.join

        def join(thread, timeout=None):
            steps.append(timeout)
            if len(steps) >= 3:
                release.set()
            return real_join(thread, timeout)

        with unittest.mock.patch.object(threading.Thread, "join", join):
            interruptible(release.wait)
        self.assertTrue(all(t is not None and t <= 0.5 for t in steps))
        self.assertGreaterEqual(len(steps), 3)


class ServerLifecycleTest(unittest.TestCase):
    def test_server_address(self) -> None:
        self.assertEqual(_server_address(["-ngl", "999"]), ("127.0.0.1", 8080))
        self.assertEqual(_server_address(["--host", "0.0.0.0", "--port", "9000"]), ("127.0.0.1", 9000))

    def test_waits_for_health(self) -> None:
        FakeServer.calls = []
        httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)  # cleanups run last-in first-out: shutdown, then close
        self.addCleanup(httpd.shutdown)
        wait_ready("127.0.0.1", httpd.server_address[1], FakeProc(None), timeout=10, poll=0.01)
        self.assertEqual(FakeServer.calls, ["health", "health", "health"])

    def test_warm_up_sends_each_prompt_length(self) -> None:
        FakeServer.calls = []
        httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        warm_up("127.0.0.1", httpd.server_address[1], sizes=(1, 12, 512), timeout=10)
        self.assertEqual(FakeServer.calls, ["/completion 1 2", "/completion 12 2", "/completion 512 2"])

    def test_server_lines_and_messages_stay_whole(self) -> None:
        # A server line written in two pieces, with a launcher message in between.
        out = io.StringIO()
        log = LineLog(out)
        r, w = os.pipe()
        reader = threading.Thread(target=log.forward, args=(os.fdopen(r, "rb"),))
        reader.start()
        with os.fdopen(w, "wb", buffering=0) as pipe:
            pipe.write(b"3.16.287.620 I ")
            time.sleep(0.05)
            log.say("ready: http://127.0.0.1:8080/v1 (loaded in 9 s)")
            pipe.write(b"slot print_timing: id 0\r\nlast line without newline")
        reader.join(timeout=5)
        self.assertEqual(out.getvalue().splitlines(), [
            "ready: http://127.0.0.1:8080/v1 (loaded in 9 s)",
            "3.16.287.620 I slot print_timing: id 0",
            "last line without newline",
        ])

    @unittest.skipIf(os.name == "nt", "the stand-in server is a script with a #! line")
    def test_run_forwards_server_output_and_warms_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "llama-server")
            with open(exe, "w") as f:
                f.write(FAKE_LLAMA_SERVER.replace("PYTHON", sys.executable))
            os.chmod(exe, 0o755)
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                ready_at: list = []
                code = run(exe, LlamaCommand(args=[], env={}), "m.gguf", ["--port", str(port)], warmup=True,
                           on_ready=lambda h, p: ready_at.append((h, p)))
        self.assertEqual(code, 0)
        self.assertEqual(ready_at, [("127.0.0.1", port)])
        lines = err.getvalue().splitlines()
        self.assertIn("0.00.000.001 I server starting on a line written in two pieces", lines)
        self.assertEqual(sum(1 for l in lines if l.startswith("served /completion")), 8)
        self.assertTrue(any(re.fullmatch(rf"ready: http://127.0.0.1:{port}/v1 \(loaded in \d+ s, warmed up in \d+ s\)", l)
                            for l in lines), lines)

    @unittest.skipIf(os.name == "nt", "the stand-in server is a script with a #! line")
    def test_another_thread_can_stop_it(self) -> None:
        # A worker that runs a model for another PC stops it when that PC goes away (``serving``).
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "llama-server")
            with open(exe, "w") as f:
                f.write(FAKE_LLAMA_SERVER.replace("PYTHON", sys.executable))
            os.chmod(exe, 0o755)
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            out, err, stop = io.StringIO(), io.StringIO(), threading.Event()
            started = time.monotonic()
            with contextlib.redirect_stderr(err):
                code = run(exe, LlamaCommand(args=[], env={}), "m.gguf", ["--port", str(port)],
                           on_ready=lambda h, p: threading.Timer(0.1, stop.set).start(), log=LineLog(out), stop=stop)
            self.assertEqual(code, 130)  # as on Ctrl+C
            self.assertLess(time.monotonic() - started, 20)
            # Its output and the launcher's messages went to the log that was given, not to stderr.
            self.assertIn("server starting on a line written in two pieces", out.getvalue())
            self.assertIn(f"ready: http://127.0.0.1:{port}/v1", out.getvalue())
            self.assertNotIn("server starting", err.getvalue())
            self.assertNotIn("ready:", err.getvalue())
            with self.assertRaises(OSError):  # the server is gone
                socket.create_connection(("127.0.0.1", port), timeout=2).close()
            # While it is still loading, too.
            with open(exe, "w") as f:
                f.write(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
            stop = threading.Event()
            threading.Timer(0.2, stop.set).start()
            started = time.monotonic()
            code = run(exe, LlamaCommand(args=[], env={}), "m.gguf", ["--port", str(port)], log=LineLog(out),
                       stop=stop)
            self.assertEqual(code, 130)
            self.assertLess(time.monotonic() - started, 20)

    def test_the_wait_ends_when_told_to_stop(self) -> None:
        stop = threading.Event()
        threading.Timer(0.05, stop.set).start()
        started = time.monotonic()
        refused = unittest.mock.patch("urllib.request.urlopen", side_effect=ConnectionRefusedError())
        with refused, self.assertRaises(Stopped):
            wait_ready("127.0.0.1", 9, FakeProc(None), timeout=60, poll=30, stop=stop)  # not after the 30 s
        self.assertLess(time.monotonic() - started, 10)

    def test_stops_waiting_when_server_exits(self) -> None:
        with self.assertRaises(RuntimeError):
            wait_ready("127.0.0.1", 9, FakeProc(1), timeout=60, poll=0.01)

    def test_timeout(self) -> None:
        with self.assertRaises(TimeoutError):
            wait_ready("127.0.0.1", 9, timeout=0.05, poll=0.01)

    def test_says_it_is_still_loading(self) -> None:
        said: list[str] = []
        # Refused at once: on Windows a real refused connection takes about 2 s, longer than this test waits.
        refused = unittest.mock.patch("urllib.request.urlopen", side_effect=ConnectionRefusedError())
        with refused, self.assertRaises(TimeoutError):
            wait_ready("127.0.0.1", 9, timeout=0.3, poll=0.01, say=said.append, every=0.1)
        self.assertGreaterEqual(len(said), 2)
        self.assertTrue(all(re.fullmatch(r"still loading \(\d+ min so far; Ctrl\+C stops it\)", s) for s in said), said)

    def test_big_model_is_not_memory_mapped(self) -> None:
        # E5 rig: gpt-oss-120b (59 GiB) with 52 GiB of RAM for weights.
        devs, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
        host.expert_path = "cpu"
        for name, expect in (("gpt-oss-120b.json", True), ("qwen3-32b-q4_k_m.json", False)):
            model = ModelProfile.from_json_file(os.path.join(EXAMPLES, name))
            plan = make_plan(model, devs, host, context=8192)
            server = llama_command(plan, model, devs, threads=32, ram_bytes=host.ram_bytes)
            self.assertEqual("--no-host" in server.args, expect, name)
            bench = llama_command(plan, model, devs, tool="bench", ram_bytes=host.ram_bytes)
            self.assertNotIn("--no-host", bench.args)


if __name__ == "__main__":
    unittest.main()
