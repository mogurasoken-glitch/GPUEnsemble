import errno
import os
import socket
import sys
import threading
import unittest
from unittest import mock

from gpuensemble_planner import gate
from gpuensemble_planner.gate import Gate, is_allowed, parse_allowed


class Echo:
    """A TCP server on 127.0.0.1 that sends back what it gets, like a stand-in RPC server."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.address = self.sock.getsockname()
        self.connections = 0
        self.ended = threading.Event()  # a connection was closed from the other side
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self.echo, args=(conn,), daemon=True).start()

    def echo(self, conn: socket.socket) -> None:
        with conn:
            while data := conn.recv(65536):
                conn.sendall(data)
        self.ended.set()

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    parts, got = [], 0
    while got < n:
        chunk = sock.recv(min(1 << 20, n - got))
        if not chunk:
            break
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


class GateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.echo = Echo()
        self.addCleanup(self.echo.close)
        self.logs: list[str] = []

    def gate(self, allowed: list[str]) -> Gate:
        g = Gate(("127.0.0.1", 0), self.echo.address, parse_allowed(allowed), self.logs.append).start()
        self.addCleanup(g.close)
        return g

    def test_allowed_client_reaches_the_server(self) -> None:
        g = self.gate(["127.0.0.1"])
        payload = os.urandom(8 << 20)  # like a tensor upload
        with socket.create_connection(g.address, timeout=10) as c:
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sender = threading.Thread(target=c.sendall, args=(payload,))
            sender.start()
            back = recv_exactly(c, len(payload))
            sender.join()
            self.assertEqual(back, payload)
            for _ in range(20):  # many small request/answer turns, as RPC calls are
                c.sendall(b"x" * 40)
                self.assertEqual(recv_exactly(c, 40), b"x" * 40)
        self.assertEqual(g.refused, 0)

    def test_other_clients_are_turned_away(self) -> None:
        g = self.gate(["10.0.0.0/8", "192.168.10.1"])
        with socket.create_connection(g.address, timeout=10) as c:
            c.settimeout(10)
            try:
                self.assertEqual(c.recv(1), b"")  # closed without a word
            except ConnectionResetError:
                pass
        self.assertEqual(g.refused, 1)
        self.assertEqual(self.echo.connections, 0)
        self.assertIn("refused a connection from 127.0.0.1", self.logs[0])

    def test_server_down(self) -> None:
        self.echo.close()
        g = self.gate(["127.0.0.1"])
        with socket.create_connection(g.address, timeout=10) as c:
            c.settimeout(15)
            try:
                self.assertEqual(c.recv(1), b"")
            except ConnectionResetError:
                pass
        self.assertTrue(any("cannot reach the RPC server" in line for line in self.logs))

    def test_close_stops_listening(self) -> None:
        g = self.gate(["127.0.0.1"])
        g.close()
        with self.assertRaises(OSError):
            socket.create_connection(g.address, timeout=2).close()


class Vanished:
    """The gate's side of a main PC that sent a request and then lost power: nothing more
    comes from it, and the system's probes end the wait with a timeout."""

    def __init__(self) -> None:
        self.requests = [b"hello"]
        self.answers = b""
        self.closed = False

    def recv(self, n: int) -> bytes:
        if self.requests:
            return self.requests.pop()
        raise TimeoutError(errno.ETIMEDOUT, "the connection timed out")

    def sendall(self, data: bytes) -> None:
        self.answers += data

    def close(self) -> None:
        self.closed = True

    def setsockopt(self, *args) -> None:
        pass

    def ioctl(self, *args) -> None:
        pass

    def shutdown(self, how: int) -> None:
        pass


class GoneTest(unittest.TestCase):
    """A main PC that goes away without closing its connection must not hold the RPC server."""

    def setUp(self) -> None:
        self.echo = Echo()
        self.addCleanup(self.echo.close)
        self.logs: list[str] = []

    def test_the_main_pcs_connection_is_probed(self) -> None:
        probed = []

        def keep_alive(sock: socket.socket) -> None:
            real(sock)
            probed.append((sock.getpeername(), sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE),
                           sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE)
                           if sys.platform == "linux" else None))

        real = gate._keep_alive
        with mock.patch.object(gate, "_keep_alive", keep_alive):
            g = Gate(("127.0.0.1", 0), self.echo.address, parse_allowed(["127.0.0.1"]), self.logs.append).start()
            self.addCleanup(g.close)
            with socket.create_connection(g.address, timeout=10) as c:
                c.sendall(b"x")
                self.assertEqual(recv_exactly(c, 1), b"x")
                mine = c.getsockname()
        # The main PC's side, not the RPC server's (that one is on this PC and cannot vanish).
        self.assertEqual(self.logs, [])  # the system took the settings
        self.assertEqual([p[0] for p in probed], [mine])
        self.assertTrue(probed[0][1])
        if sys.platform == "linux":
            self.assertEqual(probed[0][2], gate.KEEPALIVE_IDLE)

    def test_a_main_pc_that_stopped_answering_is_let_go(self) -> None:
        pc = Vanished()
        gate._join(pc, self.echo.address, self.logs.append, "10.0.0.1")
        # The RPC server saw the connection end (that is when it frees what the PC loaded).
        self.assertTrue(self.echo.ended.wait(10))
        self.assertTrue(pc.closed)
        self.assertEqual(pc.answers, b"hello")
        self.assertEqual(len(self.logs), 1)
        self.assertIn("the PC at 10.0.0.1 stopped answering", self.logs[0])

    def test_a_system_that_cannot_probe_still_joins(self) -> None:
        def refuse(sock: socket.socket) -> None:
            raise OSError(errno.ENOPROTOOPT, "protocol not available")

        with mock.patch.object(gate, "_keep_alive", refuse):
            g = Gate(("127.0.0.1", 0), self.echo.address, parse_allowed(["127.0.0.1"]), self.logs.append).start()
            self.addCleanup(g.close)
            with socket.create_connection(g.address, timeout=10) as c:
                c.sendall(b"x")
                self.assertEqual(recv_exactly(c, 1), b"x")
        self.assertEqual(len(self.logs), 1)
        self.assertIn("cannot probe the connection from 127.0.0.1", self.logs[0])

    def test_a_main_pc_that_closes_is_not_reported(self) -> None:
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with socket.create_connection(listener.getsockname(), timeout=10) as c:
            pc, _ = listener.accept()
            joined = threading.Thread(target=gate._join, args=(pc, self.echo.address, self.logs.append, "10.0.0.1"))
            joined.start()
            c.sendall(b"hello")
            self.assertEqual(recv_exactly(c, 5), b"hello")
        joined.join(10)
        self.assertFalse(joined.is_alive())
        self.assertTrue(self.echo.ended.wait(10))
        self.assertEqual(self.logs, [])

    def test_only_silence_counts_as_gone(self) -> None:
        self.assertTrue(gate._went_silent(TimeoutError(errno.ETIMEDOUT, "timed out")))
        self.assertTrue(gate._went_silent(OSError(errno.ENETRESET, "dropped on reset")))  # Windows, no answer to the probes
        self.assertTrue(gate._went_silent(ConnectionAbortedError(errno.ECONNABORTED, "aborted")))  # Windows, answer undelivered
        self.assertFalse(gate._went_silent(ConnectionResetError(errno.ECONNRESET, "reset")))  # llama-server was stopped
        self.assertFalse(gate._went_silent(BrokenPipeError(errno.EPIPE, "broken pipe")))


class AllowTest(unittest.TestCase):
    def test_parse(self) -> None:
        nets = parse_allowed(["192.168.10.1", " 10.0.0.0/8 ", "", "::1"])
        self.assertEqual([str(n) for n in nets], ["192.168.10.1/32", "10.0.0.0/8", "::1/128"])
        with self.assertRaisesRegex(ValueError, "not an IP"):
            parse_allowed(["pc1"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_allowed(["", " "])

    def test_match(self) -> None:
        nets = parse_allowed(["192.168.10.0/24", "::1"])
        self.assertTrue(is_allowed("192.168.10.7", nets))
        self.assertTrue(is_allowed("::ffff:192.168.10.7", nets))  # IPv4 on a dual-stack socket
        self.assertTrue(is_allowed("::1", nets))
        self.assertFalse(is_allowed("192.168.11.7", nets))
        self.assertFalse(is_allowed("127.0.0.1", nets))
        self.assertFalse(is_allowed("not-an-ip", nets))


if __name__ == "__main__":
    unittest.main()
