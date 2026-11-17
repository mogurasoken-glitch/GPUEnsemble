import io
import ipaddress
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

from gpuensemble_planner import serving, worker
from gpuensemble_planner.devices import GIB, Device, Host, load_gpus
from gpuensemble_planner.discovery import find_workers
from gpuensemble_planner.fetch import FetchServer, Piece, ask_to_fetch
from gpuensemble_planner.llamabin import Install
from gpuensemble_planner.llamacpp import LineLog
from gpuensemble_planner.model import ModelProfile

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
QWEN = ModelProfile.from_json_file(os.path.join(EXAMPLES, "qwen3-32b-q4_k_m.json"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def until(condition, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class Timed(unittest.TestCase):
    """These tests wait for sockets and for other threads: one that waits out a timeout fails, by its name."""

    LIMIT = 60.0  # seconds; each takes a few at most

    def setUp(self) -> None:
        super().setUp()
        started = time.monotonic()
        # Added first, so it runs last: what the test leaves to be closed is timed too.
        self.addCleanup(lambda: self.assertLess(time.monotonic() - started, self.LIMIT, "the test waited for something"))


def to_itself(address, timeout=None) -> socket.socket:
    """A connection whose two ends are one socket, as the system makes when it hands a connection
    the very port it connects to and nothing listens there."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        s.settimeout(timeout)
        s.connect(s.getsockname())
    except OSError:
        s.close()
        raise unittest.SkipTest("this system does not connect a socket to itself")
    return s


class RequestTest(unittest.TestCase):
    def test_a_model_id_and_a_context_size_and_nothing_else(self) -> None:
        ok = {"gpuensemble": "serve", "v": 1, "model": "qwen3-32b", "ctx": 8192}
        self.assertEqual(serving.parse_request(ok), ("qwen3-32b", 8192))
        # Whatever else a request carries is not read: no file, address or argument can be named.
        self.assertEqual(serving.parse_request(dict(ok, file="C:\\x.gguf", extra=["--host", "0.0.0.0"])),
                         ("qwen3-32b", 8192))
        for bad in (dict(ok, v=2), dict(ok, gpuensemble="fetch"), dict(ok, model="../x"), dict(ok, model="a b"),
                    dict(ok, model="C:\\models\\x.gguf"), dict(ok, model=""), dict(ok, model=7),
                    dict(ok, model="-m"), dict(ok, ctx="8192"), dict(ok, ctx=True), dict(ok, ctx=8),
                    dict(ok, ctx=1 << 30), {"gpuensemble": "serve", "v": 1, "model": "qwen3-32b"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                serving.parse_request(bad)

    def test_ports(self) -> None:
        self.assertEqual(serving.api_port(50052), 50054)


class Worker:
    """The worker's side on 127.0.0.1: the request port, and a stand-in for llama-server."""

    def __init__(self, test: unittest.TestCase, launch=None) -> None:
        self.logs: list[str] = []
        self.launched: list[tuple] = []
        self.stopped = threading.Event()
        self.api = free_port()
        self.serving = serving.Serving(("127.0.0.1", self.api), launch or self.launch, self.logs.append,
                                       out=io.StringIO())
        self.requests = FetchServer(("127.0.0.1", 0), "unused", lambda peer: True, self.logs.append,
                                    others={"serve": self.serving.handle}, fetches=False).start()
        self.port = self.requests.address[1]
        test.addCleanup(self.requests.close)
        test.addCleanup(self.serving.stop.set)

    def launch(self, model_id, ctx, port, log, stop, on_ready) -> int:
        """Like llama-server: listens on 127.0.0.1:port, answers each connection once, until told to stop."""
        self.launched.append((model_id, ctx, port))
        log.say(f"downloading {model_id}")
        with socket.socket() as srv:
            srv.bind(("127.0.0.1", port))
            srv.listen(4)
            srv.settimeout(0.05)
            log.say("ready: http://127.0.0.1/v1 (loaded in 0 s)")
            on_ready("127.0.0.1", port)
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.sendall(b"answer to " + conn.recv(100))
        self.stopped.set()
        return 130


class ServingTest(Timed):
    def ask(self, worker: Worker, model: str = "qwen3-32b", **kw) -> tuple[serving.Session, list[str]]:
        said: list[str] = []
        session = serving.ask("127.0.0.1", worker.port, model, 8192, said.append, **kw)
        self.addCleanup(session.close)
        return session, said

    def test_a_model_runs_there_and_answers_here(self) -> None:
        worker = Worker(self)
        session, said = self.ask(worker)
        self.assertEqual(session.port, worker.api)
        [(model_id, ctx, inner)] = worker.launched
        self.assertEqual((model_id, ctx), ("qwen3-32b", 8192))
        self.assertNotIn(inner, (worker.api, worker.port))  # llama-server has a port of its own on 127.0.0.1
        self.assertEqual(said, ["downloading qwen3-32b", "ready: http://127.0.0.1/v1 (loaded in 0 s)"])
        self.assertTrue(any("asks this PC to run qwen3-32b itself (context 8192)" in line for line in worker.logs))

        # The main PC listens where the user expects the API and passes connections on.
        relay = serving.Relay(("127.0.0.1", 0), ("127.0.0.1", session.port), worker.logs.append).start()
        self.addCleanup(relay.close)
        for _ in range(3):
            with socket.create_connection(relay.address, timeout=5) as c:
                c.sendall(b"GET /health")
                self.assertEqual(c.recv(100), b"answer to GET /health")

        # Closing the session (`up` stopped) stops llama-server there and takes its gate away.
        session.close()
        self.assertTrue(worker.stopped.wait(5))
        self.assertTrue(until(lambda: any("the GPUs are free again" in line for line in worker.logs)))
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", worker.api), timeout=2).close()
        # ... and the next request is taken.
        session, _ = self.ask(worker)
        self.assertEqual(len(worker.launched), 2)

    def test_the_api_is_for_the_pc_that_asked_only(self) -> None:
        worker = Worker(self)
        with mock.patch.object(serving, "Gate", wraps=serving.Gate) as gate:
            self.ask(worker)
        listen, target, allowed = gate.call_args.args[:3]
        self.assertEqual(listen, ("127.0.0.1", worker.api))
        self.assertEqual(target[0], "127.0.0.1")
        self.assertEqual(allowed, [ipaddress.ip_network("127.0.0.1/32")])

    def test_one_model_at_a_time(self) -> None:
        worker = Worker(self)
        self.ask(worker)
        with self.assertRaisesRegex(OSError, "already running qwen3-32b for the PC at 127.0.0.1"):
            serving.ask("127.0.0.1", worker.port, "llama-3.3-70b", 8192, lambda line: None)
        self.assertEqual(len(worker.launched), 1)

    def test_what_goes_wrong_there_is_said_here(self) -> None:
        def launch(model_id, ctx, port, log, stop, on_ready) -> int:
            log.say("looking")
            raise LookupError(f"{model_id} is not in this PC's catalog")

        worker = Worker(self, launch)
        said: list[str] = []
        with self.assertRaisesRegex(OSError, "nope is not in this PC's catalog"):
            serving.ask("127.0.0.1", worker.port, "nope", 8192, said.append)
        self.assertEqual(said, ["looking"])
        self.assertTrue(until(lambda: not worker.serving.running))
        # A request that asks for something else gets its reason too, and nothing is started.
        with socket.create_connection(("127.0.0.1", worker.port), timeout=5) as c:
            c.sendall(b'{"gpuensemble": "serve", "v": 1, "model": "../../x", "ctx": 8192}\n')
            self.assertIn(b"needs a model id from the catalog", c.recv(1000))
        # This PC takes requests to run models only.
        with socket.create_connection(("127.0.0.1", worker.port), timeout=5) as c:
            c.sendall(b'{"gpuensemble": "fetch", "v": 1, "urls": [], "pieces": []}\n')
            self.assertIn(b"does not download weights itself", c.recv(1000))

    def test_llama_server_ending_there_ends_the_wait_here(self) -> None:
        go = threading.Event()

        def launch(model_id, ctx, port, log, stop, on_ready) -> int:
            on_ready("127.0.0.1", port)
            go.wait(5)
            log.say("llama-server crashed")
            return 3

        worker = Worker(self, launch)
        session, _ = self.ask(worker)
        said: list[str] = []
        go.set()
        self.assertEqual(session.wait(said.append), 3)
        self.assertEqual(said, ["llama-server crashed"])

    def test_a_worker_that_closes_stops_the_model(self) -> None:
        worker = Worker(self)
        session, _ = self.ask(worker)
        worker.serving.close()
        self.assertTrue(worker.stopped.is_set())
        self.assertEqual(session.wait(lambda line: None), 130)

    def test_a_pc_that_says_nothing_is_given_up(self) -> None:
        def launch(model_id, ctx, port, log, stop, on_ready) -> int:
            stop.wait(5)
            return 130

        worker = Worker(self, launch)
        with self.assertRaisesRegex(OSError, "said nothing for"):
            serving.ask("127.0.0.1", worker.port, "qwen3-32b", 8192, lambda line: None, stall=0.3)
        # Giving up closed the connection, which stops the work there.
        self.assertTrue(until(lambda: not worker.serving.running))

    def test_the_last_word_arrives_when_the_main_pc_said_nothing_for_long(self) -> None:
        # The request port gives a connection a time to send its request in; the wait for the main PC
        # to leave has none, and what the worker says last is read at the other end.
        worker = Worker(self)
        here, there = socket.socketpair()
        self.addCleanup(here.close)
        there.settimeout(0.1)
        request = {"gpuensemble": "serve", "v": 1, "model": "qwen3-32b", "ctx": 8192}

        def handle() -> None:
            with there:
                worker.serving.handle(request, there, "127.0.0.1",
                                      lambda message: there.sendall(json.dumps(message).encode("utf-8") + b"\n"))

        thread = threading.Thread(target=handle, daemon=True)
        thread.start()
        self.assertTrue(until(lambda: worker.launched))
        time.sleep(0.5)
        self.assertFalse(worker.stopped.is_set())
        worker.serving.stop.set()  # the worker closes
        self.assertEqual(serving.Session(here, "127.0.0.1", 0).wait(lambda line: None), 130)
        thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_no_worker_there(self) -> None:
        started = time.monotonic()
        with socket.socket() as held:  # a port that is taken and does not listen: no connection gets it as its own
            held.bind(("127.0.0.1", 0))
            with self.assertRaises(OSError):
                serving.ask("127.0.0.1", held.getsockname()[1], "qwen3-32b", 8192, lambda line: None)
        self.assertLess(time.monotonic() - started, 20)

    def test_a_connection_that_reached_only_itself_is_no_worker(self) -> None:
        # Windows hands a port that was just freed to the next connection. One made to that port is then
        # connected to itself: it reads its own request back, and waited for a worker to say more.
        started = time.monotonic()
        with self.assertRaisesRegex(OSError, "nothing listens there"):
            serving.ask("127.0.0.1", 1, "qwen3-32b", 8192, lambda line: None, connect=to_itself)
        with self.assertRaisesRegex(OSError, "nothing listens there"):
            ask_to_fetch("127.0.0.1", 1, [Piece("https://huggingface.co/a/b/resolve/main/x", 0, 5)], connect=to_itself)
        self.assertLess(time.monotonic() - started, 20)


class LaunchTest(unittest.TestCase):
    """What the worker does for a request: its own catalog, its own cards, llama-server on 127.0.0.1."""

    def setUp(self) -> None:
        self.out = io.StringIO()
        self.log = LineLog(self.out)
        self.stop = threading.Event()
        self.started: list[tuple] = []
        self.pulled: list[str] = []

    def launch(self, model_id="qwen3-32b", cards=None, local="/models/qwen.gguf", **kw) -> int:
        def start(exe, command, path, extra, **options):
            self.started.append((exe, command, path, extra, options))
            return 0

        def pull(entry, base=None, say=None):
            self.pulled.append(entry["id"])
            say("  qwen.gguf: 1.00 / 18.40 GiB")
            return "/models/pulled.gguf"

        cards = cards if cards is not None else [Device("RTX 4090", 24 * GIB, 820e9, llama_device="CUDA0")]
        return serving.launch(model_id, 8192, 51234, self.log, self.stop, lambda host, port: None,
                              devices=lambda: cards, host_of=lambda ram: Host(int(24 * GIB), 20e9, 17e9, threads=36),
                              install=lambda log: Install("b1", "cuda", "/opt/llama"),
                              find=lambda entry, base: local, pull=pull, load=lambda path: QWEN, start=start, **kw)

    def test_a_model_that_is_here_starts_on_this_pcs_cards(self) -> None:
        self.assertEqual(self.launch(), 0)
        [(exe, command, path, extra, options)] = self.started
        self.assertEqual(exe, os.path.join("/opt/llama", "llama-server"))
        self.assertEqual(path, "/models/qwen.gguf")
        # Bound to this PC only: the gate is what the main PC reaches.
        self.assertEqual(extra, ["--host", "127.0.0.1", "--port", "51234", "-a", "qwen3-32b"])
        self.assertNotIn("--rpc", command.args)
        self.assertEqual(command.args[command.args.index("-c") + 1], "8192")
        self.assertTrue(options["warmup"])
        self.assertIs(options["stop"], self.stop)
        self.assertIs(options["log"], self.log)
        self.assertEqual(self.pulled, [])
        said = self.out.getvalue()
        self.assertIn("Qwen3-32B Q4_K_M on 1 GPU(s) of this PC, context 8192", said)
        self.assertIn("starting: ", said)

    def test_a_model_that_is_not_here_is_downloaded_first(self) -> None:
        self.assertEqual(self.launch(local=None), 0)
        self.assertEqual(self.pulled, ["qwen3-32b"])
        self.assertEqual(self.started[0][2], "/models/pulled.gguf")
        self.assertIn("1.00 / 18.40 GiB", self.out.getvalue())  # the main PC sees the download go on

    def test_a_download_stops_when_the_main_pc_is_gone(self) -> None:
        self.stop.set()
        with self.assertRaises(serving.Stopped):
            self.launch(local=None)
        self.assertEqual(self.started, [])

    def test_only_models_of_this_pcs_catalog(self) -> None:
        with self.assertRaisesRegex(LookupError, "not in this PC's catalog"):
            self.launch("some-other-model")
        self.assertEqual((self.pulled, self.started), ([], []))

    def test_a_model_that_needs_the_cpu_too_is_left_to_the_main_pc(self) -> None:
        # The main PC planned with the room the card had; another program has taken some since.
        small = [Device("RTX 4090", 12 * GIB, 820e9, llama_device="CUDA0")]
        with self.assertRaisesRegex(RuntimeError, "does not fit on this PC's GPUs alone"):
            self.launch(cards=small)
        self.assertEqual(self.started, [])


class FakeInstall:
    def __init__(self, folder: str) -> None:
        self.folder = folder

    def tool(self, name: str) -> str:
        return Install("b1", "cuda", self.folder).tool(name)


class WorkerStartTest(Timed):
    """`worker --serve`: the request port takes requests to run a model, and the rig entries say so."""

    def start(self, serve: bool, ask) -> tuple[list[dict], list[str]]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        open(os.path.join(tmp.name, "ggml-rpc-server.exe"), "w").close()
        self.port = worker.free_port()
        self.cards = [Device("GPU0 NVIDIA GeForce RTX 4090", int(24 * GIB), 372e9, llama_device="CUDA0"),
                      Device("GPU1 NVIDIA GeForce RTX 3060", int(12 * GIB), 270e9, llama_device="CUDA1")]
        self.launched: list[tuple] = []

        def launch(model_id, ctx, port, log, stop, on_ready, devices) -> int:
            self.launched.append((model_id, ctx, [d.llama_device for d in devices()]))
            on_ready("127.0.0.1", port)
            stop.wait(5)
            return 130

        out, logs = io.StringIO(), []
        with mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": tmp.name}):
            code = worker.start("127.0.0.1", self.port, ["CUDA0"], False, detect=lambda: list(self.cards),
                                install=lambda: FakeInstall(tmp.name), log=logs.append, out=out,
                                run=lambda argv, log: ask() or 0, serve=serve, launch=launch)
        self.assertEqual(code, 0)
        return json.loads(out.getvalue())["gpus"], logs

    @staticmethod
    def without_address(entries: list[dict]) -> list[dict]:
        return [{key: value for key, value in entry.items() if key != "rpc"} for entry in entries]

    def test_a_worker_that_serves_says_so_and_takes_requests(self) -> None:
        seen = {}

        def ask() -> None:
            [found] = find_workers(at=["127.0.0.1"], port=self.port + 1, timeout=2)
            seen["found"] = found.gpus
            session = serving.ask("127.0.0.1", self.port + 1, "qwen3-32b", 4096, lambda line: None)
            seen["port"] = session.port
            session.close()

        entries, logs = self.start(True, ask)
        self.assertEqual([e.get("serve") for e in entries], [True])
        self.assertTrue(load_gpus(entries)[0].rpc_serve)
        # A main PC that finds the worker by itself (--remote auto) is told the same as one given the entries
        # (but for the address, which it knows from the answer).
        self.assertEqual(seen["found"], self.without_address(entries))
        self.assertIs(seen["found"][0]["serve"], True)
        self.assertEqual(seen["port"], self.port + 2)  # the gate for the model's API
        # The model is planned on the cards this worker lends, read again when it is asked for.
        self.assertEqual(self.launched, [("qwen3-32b", 4096, ["CUDA0"])])
        self.assertTrue(until(lambda: any("the GPUs are free again" in line for line in logs)))
        # Without --cache it downloads no weights for the main PC's own llama-server.
        with self.assertRaises(OSError) as refused:
            def fetch() -> None:
                ask_to_fetch("127.0.0.1", self.port + 1, [Piece("https://huggingface.co/a/b/resolve/main/x", 0, 5)])
            self.start(True, fetch)
        self.assertIn("does not download weights itself", str(refused.exception))

    def test_a_worker_that_only_lends_cards_takes_none(self) -> None:
        seen = {}

        def ask() -> None:
            [found] = find_workers(at=["127.0.0.1"], port=self.port + 1, timeout=2)
            seen["found"] = found.gpus
            try:
                serving.ask("127.0.0.1", self.port + 1, "qwen3-32b", 4096, lambda line: None).close()
            except OSError as e:
                seen["error"] = e

        entries, _ = self.start(False, ask)
        self.assertNotIn("serve", entries[0])
        self.assertFalse(load_gpus(entries)[0].rpc_serve)
        self.assertEqual(seen["found"], self.without_address(entries))
        self.assertIn("error", seen)  # nothing listens for requests
        self.assertEqual(self.launched, [])


if __name__ == "__main__":
    unittest.main()
