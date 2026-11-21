import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

from gpuensemble_planner import cli, discovery, worker
from gpuensemble_planner.devices import (GIB, RPC_GATE_OVERHEAD, RPC_LAN_OVERHEAD, RPC_LOOPBACK_OVERHEAD, Device,
                                         load_gpus, rpc_overhead_from_rtt)
from gpuensemble_planner.discovery import (Approvals, Found, Responder, broadcast_targets, find_workers, is_this_pc,
                                          measure_rtt, one_address_each, parse_reply)
from gpuensemble_planner.tailscale import is_tailnet_address
from gpuensemble_planner.firewall import Network
from gpuensemble_planner.llamabin import Install

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
GPUS = [{"name": "GPU0 NVIDIA GeForce RTX 2080 SUPER on pc2", "vram_gib": 8.0, "reserve_gib": 1.0,
         "bandwidth_gbs": 372, "rpc": "pc2:50052", "llama_device": "CUDA0", "gate": True}]


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def query(name: str = "main") -> bytes:
    return json.dumps({"gpuensemble": "find", "v": 1, "name": name}).encode()


class CardsTest(unittest.TestCase):
    """A main PC's monitor asks a worker how the cards it lends are doing."""

    CARDS = [{"index": 0, "name": "NVIDIA GeForce RTX 4090", "temp_c": 66, "load": 88}]

    def ask(self) -> bytes:
        return json.dumps({"gpuensemble": "cards", "v": 1}).encode()

    def test_a_pc_that_is_let_in_gets_the_readings(self) -> None:
        r = Responder(("127.0.0.1", 0), "pc2", 50052, GPUS, True, lambda peer: True, cards=lambda: self.CARDS)
        self.addCleanup(r.close)
        reply = json.loads(r.answer(self.ask(), "192.168.10.1"))
        self.assertEqual(reply, {"gpuensemble": "cards", "v": 1, "name": "pc2", "port": 50052, "cards": self.CARDS})

    def test_others_get_nothing_and_are_not_asked_about(self) -> None:
        asked = []
        r = Responder(("127.0.0.1", 0), "pc2", 50052, GPUS, True, lambda peer: False,
                      lambda a, n: asked.append(a) or True, cards=lambda: self.CARDS)
        self.addCleanup(r.close)
        self.assertIsNone(r.answer(self.ask(), "192.168.10.1"))
        self.assertEqual(asked, [])  # no approval prompt for a question about the cards

    def test_a_worker_without_readings_or_whose_nvidia_smi_fails_stays_silent(self) -> None:
        r = Responder(("127.0.0.1", 0), "pc2", 50052, GPUS, True, lambda peer: True)
        self.addCleanup(r.close)
        self.assertIsNone(r.answer(self.ask(), "192.168.10.1"))

        def broken():
            raise RuntimeError("nvidia-smi was not found")

        r = Responder(("127.0.0.1", 0), "pc2", 50052, GPUS, True, lambda peer: True, cards=broken)
        self.addCleanup(r.close)
        self.assertIsNone(r.answer(self.ask(), "192.168.10.1"))

    def test_asked_over_the_network(self) -> None:
        port = free_udp_port()
        r = Responder(("127.0.0.1", port), "pc4090", port - 1, GPUS, True, lambda peer: peer == "127.0.0.1",
                      cards=lambda: self.CARDS).start()
        self.addCleanup(r.close)
        silent = free_udp_port()  # nothing answers there: an older worker, or none
        got = discovery.ask_cards([f"127.0.0.1:{port - 1}", f"127.0.0.1:{silent - 1}"], timeout=0.5)
        self.assertEqual(got, {f"127.0.0.1:{port - 1}": ("pc4090", self.CARDS)})
        self.assertEqual(discovery.ask_cards([]), {})
        self.assertEqual(discovery.ask_cards(["no-such-host.invalid:50052"], timeout=0.2), {})


class LiveCardsTest(unittest.TestCase):
    def test_only_the_lent_cards_and_one_reading_per_second(self) -> None:
        reads = []
        now = [0.0]

        def read():
            reads.append(now[0])
            return [{"index": 0, "name": "A"}, {"index": 1, "name": "B"}, {"index": 2, "name": "C"}]

        lent = [Device("GPU2 C", 8 * GIB, 1e11, llama_device="CUDA2"), Device("GPU0 A", 8 * GIB, 1e11,
                                                                               llama_device="CUDA0")]
        cards = worker.LiveCards(lent, read, wait=5.0, clock=lambda: now[0])
        self.assertEqual([c["name"] for c in cards()], ["A", "C"])
        now[0] += 0.5
        cards()
        self.assertEqual(len(reads), 1)
        now[0] += 1.0
        cards()
        self.assertEqual(len(reads), 2)

    def test_a_failed_reading_keeps_the_last_one(self) -> None:
        results = [[{"index": 0, "temp_c": 60}], RuntimeError("gone")]

        def read():
            r = results.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        now = [0.0]
        cards = worker.LiveCards([Device("GPU0 A", 8 * GIB, 1e11, llama_device="CUDA0")], read, wait=5.0,
                                 clock=lambda: now[0])
        self.assertEqual(cards()[0]["temp_c"], 60)
        now[0] += 2
        self.assertEqual(cards()[0]["temp_c"], 60)


class ResponderTest(unittest.TestCase):
    def responder(self, allowed=lambda peer: True, on_request=None) -> Responder:
        r = Responder(("127.0.0.1", 0), "pc2", 50052, GPUS, True, allowed, on_request)
        self.addCleanup(r.close)
        return r

    def test_allowed_main_pc_gets_the_gpus(self) -> None:
        reply = json.loads(self.responder().answer(query(), "192.168.10.1"))
        self.assertEqual((reply["name"], reply["port"], reply["allowed"], reply["gate"]), ("pc2", 50052, True, True))
        self.assertEqual(reply["gpus"][0]["llama_device"], "CUDA0")
        self.assertNotIn("rpc", reply["gpus"][0])  # the main PC uses the address that answered

    def test_other_pcs_get_only_the_name_and_are_shown_once(self) -> None:
        asked = []
        r = self.responder(allowed=lambda peer: False, on_request=lambda a, n: asked.append((a, n)) or True)
        replies = [json.loads(r.answer(query("E5"), "192.168.10.1")) for _ in range(3)]
        self.assertEqual(replies[0], {"gpuensemble": "worker", "v": 1, "name": "pc2", "port": 50052, "allowed": False,
                                      "asking": True})  # the prompt is shown now: the main PC may wait for y
        self.assertEqual([r["asking"] for r in replies[1:]], [False, False])
        self.assertEqual(asked, [("192.168.10.1", "E5")])
        # A worker whose user cannot answer there does not make the main PC wait.
        r = self.responder(allowed=lambda peer: False, on_request=lambda a, n: False)
        self.assertFalse(json.loads(r.answer(query("E5"), "192.168.10.1"))["asking"])

    def test_a_pc_let_in_while_it_asks_gets_the_gpus_at_once(self) -> None:
        # The worker lets its user's own Tailscale PC in without a prompt: no second question is needed.
        let_in: set[str] = set()
        r = self.responder(allowed=lambda peer: peer in let_in, on_request=lambda a, n: let_in.add(a) or False)
        reply = json.loads(r.answer(query("E5"), "100.64.0.1"))
        self.assertTrue(reply["allowed"])
        self.assertEqual(reply["gpus"][0]["llama_device"], "CUDA0")
        self.assertNotIn("asking", reply)

    def test_gpus_are_read_for_each_pc_that_is_let_in(self) -> None:
        # The worker gives a function: the cards' memory is what is free when the main PC asks.
        calls = []

        def gpus():
            calls.append(1)
            return [dict(GPUS[0], vram_gib=8.0 - len(calls))]

        r = Responder(("127.0.0.1", 0), "pc2", 50052, gpus, True, lambda peer: peer == "192.168.10.1")
        self.addCleanup(r.close)
        self.assertNotIn("gpus", json.loads(r.answer(query(), "192.168.10.9")))
        self.assertEqual(calls, [])  # a PC that is not let in does not make the worker read its cards
        first = json.loads(r.answer(query(), "192.168.10.1"))["gpus"][0]
        second = json.loads(r.answer(query(), "192.168.10.1"))["gpus"][0]
        self.assertEqual((first["vram_gib"], second["vram_gib"]), (7.0, 6.0))
        self.assertNotIn("rpc", first)

    def test_ignores_anything_else(self) -> None:
        r = self.responder()
        for junk in (b"", b"\xff\xfe", b"[]", b'{"gpuensemble": "find", "v": 2}', b'{"other": "find", "v": 1}'):
            self.assertIsNone(r.answer(junk, "192.168.10.1"))

    def test_found_over_udp(self) -> None:
        r = Responder(("127.0.0.1", free_udp_port()), "pc2", 50052, GPUS, True, lambda peer: peer == "127.0.0.1")
        self.addCleanup(r.close)
        r.start()
        [w] = find_workers(at=["127.0.0.1"], port=r.address[1], timeout=2)
        self.assertEqual((w.address, w.port, w.name, w.allowed, w.gate), ("127.0.0.1", 50052, "pc2", True, True))
        [dev] = load_gpus(w.rig_entries())
        self.assertEqual((dev.rpc, dev.llama_device), ("127.0.0.1:50052", "CUDA0"))
        self.assertAlmostEqual(dev.rpc_overhead, RPC_LOOPBACK_OVERHEAD + RPC_GATE_OVERHEAD)
        self.assertEqual(find_workers(at=["127.0.0.1"], port=free_udp_port(), timeout=0.3), [])

    def test_vpn_pcs_are_asked_one_by_one_and_twice(self) -> None:
        # A PC behind a VPN: the first packet is lost while the VPN finds its path, the second is answered.
        answers = Responder(("127.0.0.1", free_udp_port()), "pc4090", 50052, GPUS, True, lambda peer: True)
        self.addCleanup(answers.close)
        port = free_udp_port()
        far = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        far.bind(("127.0.0.1", port))
        far.settimeout(5)
        self.addCleanup(far.close)
        got = []

        def serve() -> None:
            try:
                for _ in range(2):
                    data, peer = far.recvfrom(8192)
                    got.append(data)
                far.sendto(answers.answer(data, peer[0]), peer)
            except OSError:
                pass

        threading.Thread(target=serve, daemon=True).start()
        with mock.patch.object(discovery, "VPN_TIMEOUT", 1.5), mock.patch.object(discovery, "VPN_ASK_AGAIN", 0.1), \
                mock.patch.object(discovery, "broadcast_targets", return_value=[]):
            [w] = find_workers(port=port, timeout=0.2, also=["127.0.0.1", "127.0.0.1"])
        self.assertEqual((w.name, w.address), ("pc4090", "127.0.0.1"))
        self.assertEqual(len(got), 2)  # asked once at the start and once again, not once per mention

    def test_one_address_each(self) -> None:
        lan = Found("192.168.10.2", 50052, "pc2", True, True, GPUS)
        vpn = Found("100.64.0.2", 50052, "PC2", True, True, GPUS)
        other = Found("100.64.0.9", 50052, "pc9", False)
        self.assertEqual(one_address_each([vpn, other, lan], is_tailnet_address), [lan, other])
        # The address that lets this PC in wins over the nearer one.
        waiting = Found("192.168.10.2", 50052, "pc2", False, asking=True)
        self.assertEqual(one_address_each([waiting, vpn], is_tailnet_address), [vpn])
        # Two workers of one PC on different ports are two workers.
        second = Found("192.168.10.2", 50062, "pc2", True, True, GPUS)
        self.assertEqual(len(one_address_each([lan, second], is_tailnet_address)), 2)

    def test_round_trip(self) -> None:
        r = Responder(("127.0.0.1", free_udp_port()), "pc2", 50052, GPUS, True, lambda peer: True)
        self.addCleanup(r.close)
        r.start()
        here = Found("127.0.0.1", r.address[1] - 1, "pc2", True)  # the answer comes from rpc port + 1
        rtt = measure_rtt(here, tries=3)
        self.assertIsNotNone(rtt)
        self.assertLess(rtt, 0.5)
        self.assertIsNone(measure_rtt(Found("127.0.0.1", free_udp_port() - 1, "x", True), tries=2, timeout=0.2))

    def test_close_is_prompt(self) -> None:
        r = self.responder().start()
        t = time.monotonic()
        r.close()
        self.assertLess(time.monotonic() - t, 2)


class ParseTest(unittest.TestCase):
    def test_replies(self) -> None:
        good = {"gpuensemble": "worker", "v": 1, "name": "pc2", "port": 50052, "allowed": True, "gpus": GPUS}
        w = parse_reply(json.dumps(good).encode(), "10.0.0.2")
        self.assertEqual((w.endpoint, len(w.gpus)), ("10.0.0.2:50052", 1))
        for bad in (dict(good, v=2), dict(good, port="x"), dict(good, gpus="x"), dict(good, gpus=[1])):
            self.assertIsNone(parse_reply(json.dumps(bad).encode(), "10.0.0.2"))
        self.assertIsNone(parse_reply(b"not json", "10.0.0.2"))
        self.assertEqual(Found("10.0.0.2", 50052, "pc2", True, gate=False, gpus=GPUS).rig_entries()[0].get("gate"),
                         True)  # the entry's own flag stays

    def test_this_pc(self) -> None:
        here = ["192.168.10.1"]
        self.assertTrue(is_this_pc(Found("192.168.10.1", 50052, "x", True), here, "E5"))
        self.assertTrue(is_this_pc(Found("127.0.0.1", 50052, "x", True), here, "E5"))
        self.assertTrue(is_this_pc(Found("10.0.0.1", 50052, "e5", True), here, "E5"))  # its other network card
        self.assertFalse(is_this_pc(Found("192.168.10.2", 50052, "pc2", True), here, "E5"))

    def test_round_trip_in_the_rig_entries(self) -> None:
        self.assertAlmostEqual(rpc_overhead_from_rtt(0.0003), RPC_LAN_OVERHEAD)  # a LAN: the usual guess
        self.assertAlmostEqual(rpc_overhead_from_rtt(0.010), RPC_LOOPBACK_OVERHEAD + 0.010)  # one wait per token
        self.assertAlmostEqual(rpc_overhead_from_rtt(0.010, gate=True), RPC_LOOPBACK_OVERHEAD + 0.010 + RPC_GATE_OVERHEAD)
        far = Found("100.64.0.2", 50052, "pc4090", True, True, GPUS, rtt=0.010)
        [entry] = far.rig_entries()
        self.assertEqual(entry["rpc_overhead_ms"], round((RPC_LOOPBACK_OVERHEAD + 0.010 + RPC_GATE_OVERHEAD) * 1000, 2))
        self.assertAlmostEqual(load_gpus([entry])[0].rpc_overhead, entry["rpc_overhead_ms"] / 1000)
        # The card is named after the address it is reached at, not the LAN address the worker took for its own.
        self.assertEqual(entry["name"], "GPU0 NVIDIA GeForce RTX 2080 SUPER on 100.64.0.2")
        self.assertEqual(GPUS[0]["name"], "GPU0 NVIDIA GeForce RTX 2080 SUPER on pc2")
        self.assertNotIn("rpc_overhead_ms", Found("127.0.0.1", 50052, "pc2", True, gpus=GPUS, rtt=0.001).rig_entries()[0])
        self.assertNotIn("rpc_overhead_ms", Found("10.0.0.2", 50052, "pc2", True, gpus=GPUS).rig_entries()[0])

    def test_broadcast_targets(self) -> None:
        self.assertEqual(broadcast_targets(["192.168.10.1", "10.1.2.3", "8.8.8.8", "192.168.10.7"]),
                         ["255.255.255.255", "192.168.10.255", "10.1.2.255"])


class ApprovalsTest(unittest.TestCase):
    def test_typing(self) -> None:
        admitted, logs = [], []
        a = Approvals(admitted.append, logs.append)
        self.assertFalse(a.handle("y"))  # nobody asked yet
        a.request("192.168.10.1", "E5")
        self.assertIn("PC 'E5' at 192.168.10.1 asks", logs[-1])
        self.assertFalse(a.handle("what"))
        self.assertTrue(a.handle(" Y \n"))
        self.assertFalse(a.handle("y"))  # each request is let in once
        self.assertTrue(a.handle("allow 192.168.10.9"))
        self.assertFalse(a.handle("allow pc1"))
        self.assertEqual(admitted, ["192.168.10.1", "192.168.10.9"])

    def test_deny(self) -> None:
        refused, logs = [], []
        a = Approvals(lambda address: None, logs.append, refused.append)
        a.request("192.168.10.1", "E5")
        self.assertFalse(a.handle("deny 192.168.10.1"))
        self.assertFalse(a.handle("y"))  # no longer waiting
        self.assertFalse(a.handle("deny pc1"))
        self.assertEqual(refused, ["192.168.10.1"])
        self.assertIn("may no longer use these GPUs", logs[-2])

    def test_refusals_are_remembered(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "worker.json")
            discovery.save_approved(["192.168.10.1", "10.0.0.2"], path)
            discovery.save_denied("192.168.10.1", path)
            self.assertEqual(discovery.load_approved(path), ["10.0.0.2"])
            self.assertEqual(discovery.load_denied(path), ["192.168.10.1"])
            discovery.save_approved(["10.0.0.2", "192.168.10.1"], path)  # allowed again
            self.assertEqual(discovery.load_denied(path), [])

    def test_remembered(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "worker.json")
            self.assertEqual(discovery.load_approved(path), [])
            discovery.save_approved(["192.168.10.1", "192.168.10.1", "10.0.0.2"], path)
            self.assertEqual(discovery.load_approved(path), ["10.0.0.2", "192.168.10.1"])
            with open(path, "w") as f:
                f.write("{broken")
            self.assertEqual(discovery.load_approved(path), [])


class FakeInstall:
    def __init__(self, folder: str) -> None:
        self.folder = folder

    def tool(self, name: str) -> str:
        return Install("b1", "cuda", self.folder).tool(name)


class LanWorkerTest(unittest.TestCase):
    def test_first_contact_is_approved_by_typing_y(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        open(os.path.join(tmp.name, "ggml-rpc-server.exe"), "w").close()
        approved = os.path.join(tmp.name, "worker.json")
        port = worker.free_ports()
        read_end, write_end = os.pipe()
        stdin = os.fdopen(read_end, "r")
        typed = os.fdopen(write_end, "w")
        self.addCleanup(stdin.close)
        logs: list[str] = []
        seen = {}

        def fake_server(argv, log):
            inner = int(argv[argv.index("-p") + 1])
            with socket.socket() as srv:
                srv.bind(("127.0.0.1", inner))
                srv.listen(1)
                [before] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
                seen["before"] = before.allowed
                seen["asking"] = before.asking
                typed.write("y\n")
                typed.flush()
                deadline = time.monotonic() + 5
                while not any("may now use" in line for line in logs) and time.monotonic() < deadline:
                    time.sleep(0.05)
                [after] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
                seen["after"] = after
                with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
                    conn, _ = srv.accept()
                    with conn:
                        c.sendall(b"ok")
                        seen["through"] = conn.recv(2)
            return 0

        detected = [Device("GPU0 NVIDIA GeForce RTX 2080 SUPER", int(8 * GIB), 372e9, llama_device="CUDA0")]
        out = io.StringIO()
        code = worker.start("127.0.0.1", port, None, False, detect=lambda: detected,
                            install=lambda: FakeInstall(tmp.name), log=logs.append, out=out, run=fake_server,
                            lan=True, stdin=stdin, approved_path=approved)
        typed.close()
        self.assertEqual(code, 0, logs)
        self.assertFalse(seen["before"])
        self.assertTrue(seen["asking"])  # typing y works in this window, so the main PC may wait for it
        self.assertTrue(seen["after"].allowed)
        self.assertEqual(seen["after"].gpus[0]["llama_device"], "CUDA0")
        self.assertTrue(seen["after"].gate)
        self.assertEqual(seen["through"], b"ok")
        self.assertEqual(out.getvalue(), "")  # nothing to copy by hand
        self.assertTrue(any("lets in no one yet" in line for line in logs), logs)
        self.assertEqual(discovery.load_approved(approved), ["127.0.0.1"])

        # The next start remembers the approval.
        seen.clear()

        def check(argv, log):
            [w] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
            seen["allowed"] = w.allowed
            return 0

        logs.clear()
        worker.start("127.0.0.1", port, None, False, detect=lambda: detected, install=lambda: FakeInstall(tmp.name),
                     log=logs.append, out=io.StringIO(), run=check, lan=True, stdin=io.StringIO(""),
                     approved_path=approved)
        self.assertTrue(seen["allowed"])
        self.assertTrue(any("lets in 127.0.0.1/32" in line for line in logs), logs)


    def test_a_main_pc_that_asks_later_gets_the_memory_free_then(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        open(os.path.join(tmp.name, "ggml-rpc-server.exe"), "w").close()
        port = worker.free_ports()
        logs: list[str] = []
        seen = []
        free = [24.0]

        def detect():
            return [Device("GPU0 NVIDIA GeForce RTX 4090", int(free[0] * GIB), 756e9, llama_device="CUDA0")]

        def ask(argv, log):
            for now in (24.0, 23.0, 4.0, 24.0):  # at the start; the server's own; another program; gone again
                free[0] = now
                [w] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
                seen.append(w.gpus[0]["vram_gib"])
            return 0

        with mock.patch.object(worker, "MEMORY_TTL", 0.0):
            worker.start("127.0.0.1", port, None, False, detect=detect, install=lambda: FakeInstall(tmp.name),
                         log=logs.append, out=io.StringIO(), run=ask, lan=True, stdin=io.StringIO(""),
                         allow=["127.0.0.1"])
        self.assertEqual(seen, [24.0, 24.0, 4.0, 24.0])
        notes = [line for line in logs if "main PCs are now told" in line]
        self.assertEqual(len(notes), 2, logs)
        self.assertIn("4.0 GiB for GPU0 NVIDIA GeForce RTX 4090 (24.0 before): another program uses its memory", notes[0])
        self.assertIn("24.0 GiB for GPU0 NVIDIA GeForce RTX 4090 (4.0 before): its memory is free again", notes[1])

    def worker_asked_by(self, tailnet, stdin=None):
        """Start a --lan worker and ask it once from 127.0.0.1, which stands in for a tailnet address."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        open(os.path.join(tmp.name, "ggml-rpc-server.exe"), "w").close()
        approved = os.path.join(tmp.name, "worker.json")
        port = worker.free_ports()
        logs: list[str] = []
        seen = {}

        def ask(argv, log):
            [seen["answer"]] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
            return 0

        detected = [Device("GPU0 NVIDIA GeForce RTX 4090", int(24 * GIB), 372e9, llama_device="CUDA0")]
        with mock.patch.object(worker.tailscale, "is_tailnet_address", return_value=True):
            worker.start("127.0.0.1", port, None, False, detect=lambda: detected,
                         install=lambda: FakeInstall(tmp.name), log=logs.append, out=io.StringIO(), run=ask,
                         lan=True, stdin=stdin or io.StringIO(""), approved_path=approved, tailnet=tailnet)
        return seen["answer"], logs, discovery.load_approved(approved)

    def test_own_tailscale_pc_is_let_in_without_typing(self) -> None:
        status = {"BackendState": "Running", "Self": {"UserID": 7}, "Peer": {
            "a": {"HostName": "e5", "UserID": 7, "TailscaleIPs": ["127.0.0.1"], "Online": True}}}
        answer, logs, remembered = self.worker_asked_by(lambda: status)
        self.assertTrue(answer.allowed)
        self.assertEqual(answer.gpus[0]["llama_device"], "CUDA0")
        self.assertTrue(any("is signed in to Tailscale with this PC's account (e5); it may use these GPUs" in line
                            for line in logs), logs)
        self.assertTrue(any("through Tailscale" in line and "--ask-tailscale" in line for line in logs), logs)
        self.assertEqual(remembered, [])  # Tailscale is asked again on the next start

    def test_other_tailscale_pcs_are_asked_about(self) -> None:
        friend = {"BackendState": "Running", "Self": {"UserID": 7}, "Peer": {
            "a": {"HostName": "friend", "UserID": 8, "TailscaleIPs": ["127.0.0.1"], "Online": True}}}
        for tailnet in (lambda: friend, lambda: None, None):  # another account, no Tailscale, --ask-tailscale
            answer, logs, _ = self.worker_asked_by(tailnet)
            self.assertFalse(answer.allowed)
            self.assertTrue(answer.asking)
            self.assertTrue(any("asks to use these GPUs. Type y" in line for line in logs), logs)
            # Said only where it applies: on a PC that is on a tailnet and lets its own PCs in.
            self.assertEqual(any("through Tailscale" in line for line in logs), tailnet is not None and
                             tailnet() is not None, logs)


class FakeHome:
    """Stands in for homelan.HomeLan: 127.0.0.1 is on the home LAN while ``home`` is True."""

    def __init__(self) -> None:
        self.home = True
        self.on_change = None

    def find(self, address, wait=0.0):
        return Network("Ethernet", "Network 2", "private") if self.home and address == "127.0.0.1" else None

    def start(self):
        return self

    def close(self) -> None:
        pass


class FreePortsTest(unittest.TestCase):
    def test_three_in_a_row(self) -> None:
        first = worker.free_ports(3)
        for port in range(first, first + 3):
            for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
                with socket.socket(socket.AF_INET, kind) as s:
                    s.bind(("127.0.0.1", port))


class TrustLanTest(unittest.TestCase):
    def start_worker(self, run, home, typed="", approved=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        open(os.path.join(tmp.name, "ggml-rpc-server.exe"), "w").close()
        approved = approved or os.path.join(tmp.name, "worker.json")
        port = worker.free_ports()
        logs: list[str] = []
        detected = [Device("GPU0 NVIDIA GeForce RTX 4090", int(24 * GIB), 372e9, llama_device="CUDA0")]
        read_end, write_end = os.pipe()
        stdin = os.fdopen(read_end, "r")
        self.addCleanup(stdin.close)
        pipe = os.fdopen(write_end, "w")
        worker.start("127.0.0.1", port, None, False, detect=lambda: detected, install=lambda: FakeInstall(tmp.name),
                     log=logs.append, out=io.StringIO(), run=lambda argv, log: run(port, pipe, logs),
                     lan=True, stdin=stdin, approved_path=approved, trust_lan=home)
        pipe.close()
        return logs, approved

    def test_a_pc_on_the_home_lan_is_let_in_without_asking(self) -> None:
        seen = {}

        def run(port, typed, logs):
            [seen["w"]] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
            return 0
        logs, approved = self.start_worker(run, FakeHome())
        self.assertTrue(seen["w"].allowed)
        self.assertFalse(any("Type y" in line for line in logs), logs)
        self.assertTrue(any("is on this PC's home network 'Network 2' (Ethernet), which Windows treats as private; "
                            "it may use these GPUs" in line for line in logs), logs)
        self.assertTrue(any("let in without asking when Windows treats that network as private" in line
                            for line in logs), logs)
        self.assertEqual(discovery.load_approved(approved), [])  # asked about again on the next start

    def test_not_on_a_home_lan_it_is_asked_about(self) -> None:
        home = FakeHome()
        home.home = False
        seen = {}

        def run(port, typed, logs):
            [seen["w"]] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
            return 0
        logs, _ = self.start_worker(run, home)
        self.assertFalse(seen["w"].allowed)
        self.assertTrue(seen["w"].asking)

    def test_refused_and_remembered(self) -> None:
        seen = {}

        def run(port, typed, logs):
            [seen["before"]] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
            typed.write("deny 127.0.0.1\n")
            typed.flush()
            deadline = time.monotonic() + 5
            while not any("may no longer use" in line for line in logs) and time.monotonic() < deadline:
                time.sleep(0.05)
            [seen["after"]] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
            return 0
        logs, approved = self.start_worker(run, FakeHome())
        self.assertTrue(seen["before"].allowed)
        self.assertFalse(seen["after"].allowed)
        self.assertEqual(discovery.load_denied(approved), ["127.0.0.1"])

        # The next start does not let it in by itself, and says why.
        def again(port, typed, logs):
            [seen["next"]] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
            return 0
        logs, _ = self.start_worker(again, FakeHome(), approved=approved)
        self.assertFalse(seen["next"].allowed)
        self.assertFalse(seen["next"].asking)
        self.assertTrue(any("was refused here before and is not let in" in line for line in logs), logs)

    def test_a_network_that_stops_being_home(self) -> None:
        home = FakeHome()
        seen = {}

        def run(port, typed, logs):
            [seen["before"]] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
            home.home = False
            home.on_change()
            [seen["after"]] = find_workers(at=["127.0.0.1"], port=port + 1, timeout=2)
            return 0
        logs, _ = self.start_worker(run, home)
        self.assertTrue(seen["before"].allowed)
        self.assertFalse(seen["after"].allowed)
        self.assertTrue(any("no longer on a private network" in line for line in logs), logs)


class CliAutoTest(unittest.TestCase):
    def setUp(self) -> None:
        # The workers here are made up: measuring them would only wait for timeouts.
        patch = mock.patch.object(cli, "measure_rtt", return_value=None)
        self.measure = patch.start()
        self.addCleanup(patch.stop)
        # No tailnet, whatever the PC that runs the tests is signed in to.
        patch = mock.patch.object(cli.tailscale, "read_status", return_value=None)
        self.tailnet = patch.start()
        self.addCleanup(patch.stop)

    def test_tailnet_pcs_are_asked_too(self) -> None:
        self.tailnet.return_value = {"BackendState": "Running", "Self": {"UserID": 1}, "Peer": {
            "a": {"HostName": "pc4090", "UserID": 1, "TailscaleIPs": ["100.64.0.2"], "Online": True},
            "b": {"HostName": "phone", "UserID": 1, "TailscaleIPs": ["100.64.0.3"], "Online": False}}}
        lan = Found("192.168.10.2", 50052, "pc4090", False, asking=True)
        vpn = Found("100.64.0.2", 50052, "pc4090", True, True, GPUS)
        err = io.StringIO()
        with mock.patch.object(cli, "find_workers", return_value=[vpn, lan]) as find, contextlib.redirect_stderr(err):
            [dev] = cli.found_remote("auto", wait=60, sleep=mock.Mock(side_effect=AssertionError))
        self.assertEqual(find.call_args.kwargs, {"port": 50053, "also": ["100.64.0.2"]})
        self.assertEqual(dev.rpc, "100.64.0.2:50052")  # the address that lets this PC in; nothing to wait for
        # Addresses named on purpose are the only ones asked.
        with mock.patch.object(cli, "find_workers", return_value=[]) as find, contextlib.redirect_stderr(err):
            cli.found_remote("auto:192.168.10.2")
        self.assertEqual(find.call_args.kwargs, {"at": ["192.168.10.2"], "port": 50053})

    def test_round_trip_goes_into_the_plan(self) -> None:
        far = Found("100.64.0.2", 50052, "pc4090", True, True, GPUS)
        err = io.StringIO()
        self.measure.return_value = 0.012
        with mock.patch.object(cli, "find_workers", return_value=[far]), contextlib.redirect_stderr(err):
            [dev] = cli.found_remote("auto:100.64.0.2")
        self.assertIn("remote: pc4090 at 100.64.0.2:50052 (round trip 12.0 ms):", err.getvalue())
        self.assertAlmostEqual(dev.rpc_overhead, rpc_overhead_from_rtt(0.012, gate=True), places=5)
        self.assertGreater(dev.rpc_overhead, 0.012)  # one round trip of 12 ms, and the gate

    def test_remote_auto(self) -> None:
        workers = [Found("192.168.10.2", 50052, "pc2", True, True, GPUS),
                   Found("192.168.10.3", 50052, "pc3", False)]
        err = io.StringIO()
        with mock.patch.object(cli, "find_workers", return_value=workers) as find, \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(err):
            code = cli.main(["args", os.path.join(EXAMPLES, "llama-70b-q4_k_m.json"),
                             "--rig", os.path.join(EXAMPLES, "rig-e5-turing.json"), "--remote", "auto:192.168.10.2"])
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(find.call_args.kwargs, {"at": ["192.168.10.2"], "port": 50053})
        self.assertIn("remote: pc2 at 192.168.10.2:50052", err.getvalue())
        self.assertIn("pc3 at 192.168.10.3:50052 has not let this PC in yet", err.getvalue())
        self.assertIn("--rpc 192.168.10.2:50052", out.getvalue())

    def test_own_worker_is_left_out(self) -> None:
        workers = [Found("192.168.10.1", 50052, "E5", True, True, GPUS),
                   Found("192.168.10.2", 50052, "pc2", True, True, GPUS)]
        err = io.StringIO()
        with mock.patch.object(cli, "find_workers", return_value=workers), \
                mock.patch.object(cli, "is_this_pc", side_effect=lambda w: w.name == "E5"), \
                contextlib.redirect_stderr(err):
            devices = cli.found_remote("auto")
            self.assertEqual([d.rpc for d in devices], ["192.168.10.2:50052"])
            self.assertIn("left out the worker on this PC itself (E5", err.getvalue())
            # An address named on purpose is used, as the loopback tests do.
            self.assertEqual(len(cli.found_remote("auto:192.168.10.1,192.168.10.2")), 2)

    def test_waits_for_y_on_the_worker(self) -> None:
        asked = Found("192.168.10.2", 50052, "pc2", False, asking=True)
        again = Found("192.168.10.2", 50052, "pc2", False)
        let_in = Found("192.168.10.2", 50052, "pc2", True, True, GPUS)
        err = io.StringIO()
        with mock.patch.object(cli, "find_workers", side_effect=[[asked], [again], [], [let_in]]) as find, \
                contextlib.redirect_stderr(err):
            devices = cli.found_remote("auto:192.168.10.2", wait=60, sleep=lambda s: None)
        self.assertEqual([d.rpc for d in devices], ["192.168.10.2:50052"])
        self.assertEqual(find.call_args.kwargs, {"at": ["192.168.10.2"], "port": 50053, "timeout": 1.0})
        self.assertIn("pc2 at 192.168.10.2 can lend GPUs to this PC once you type y in its worker window", err.getvalue())
        self.assertIn("waiting up to 60 s", err.getvalue())
        self.assertIn("pc2 at 192.168.10.2:50052 let this PC in", err.getvalue())

    def test_gives_up_waiting(self) -> None:
        asked = Found("192.168.10.2", 50052, "pc2", False, asking=True)
        for sleep in (lambda s: time.sleep(0.01), mock.Mock(side_effect=KeyboardInterrupt)):
            err = io.StringIO()
            with mock.patch.object(cli, "find_workers", return_value=[asked]), contextlib.redirect_stderr(err):
                self.assertEqual(cli.found_remote("auto:192.168.10.2", wait=0.05, sleep=sleep), [])
            self.assertIn("has not let this PC in yet; type y in its worker window and start again", err.getvalue())
        self.assertIn("going on without", err.getvalue())

    def test_no_wait_for_a_pc_asked_before(self) -> None:
        ignored = Found("192.168.10.2", 50052, "pc2", False)  # its user was asked on an earlier start
        err = io.StringIO()
        with mock.patch.object(cli, "find_workers", return_value=[ignored]) as find, contextlib.redirect_stderr(err):
            self.assertEqual(cli.found_remote("auto:192.168.10.2", wait=60, sleep=mock.Mock(side_effect=AssertionError)),
                             [])
        find.assert_called_once()
        self.assertNotIn("waiting", err.getvalue())

    def test_nobody_lends(self) -> None:
        err = io.StringIO()
        with mock.patch.object(cli, "find_workers", return_value=[]), contextlib.redirect_stderr(err):
            self.assertEqual(cli.found_remote("auto"), [])
        self.assertIn("no other PC lends its GPUs on this LAN or this PC's Tailscale network", err.getvalue())

    def test_find_command(self) -> None:
        with mock.patch.object(cli, "find_workers", return_value=[Found("192.168.10.2", 50052, "pc2", True, True, GPUS)]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["find"]), 0)
        self.assertIn("pc2 at 192.168.10.2:50052: lets this PC in", out.getvalue())
        self.assertIn("CUDA0: GPU0 NVIDIA GeForce RTX 2080 SUPER on pc2, 8.0 GiB", out.getvalue())
        with mock.patch.object(cli, "find_workers", return_value=[]), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["find", "--at", "10.0.0.2"]), 1)


if __name__ == "__main__":
    unittest.main()
