import json
import subprocess
import unittest
from unittest import mock

from gpuensemble_planner import tailscale
from gpuensemble_planner.tailscale import is_tailnet_address, own_pc, peers

# As `tailscale status --json` prints it, with what is read here.
STATUS = {
    "BackendState": "Running",
    "Self": {"HostName": "pc4090", "UserID": 1001, "TailscaleIPs": ["100.64.0.2", "fd7a:115c:a1e0::2"], "Online": True},
    "Peer": {
        "nodekey:a": {"HostName": "e5", "UserID": 1001, "TailscaleIPs": ["100.64.0.1", "fd7a:115c:a1e0::1"],
                      "Online": True},
        "nodekey:b": {"HostName": "phone", "UserID": 1001, "TailscaleIPs": ["100.64.0.3"], "Online": False},
        "nodekey:c": {"HostName": "web-prod", "UserID": 1001, "TailscaleIPs": ["100.64.0.4"], "Online": True,
                      "Tags": ["tag:server"]},
        "nodekey:d": {"HostName": "friend", "UserID": 2002, "TailscaleIPs": ["100.64.0.5"], "Online": True},
        "nodekey:e": {"HostName": "shared", "UserID": 1001, "TailscaleIPs": ["100.64.0.6"], "Online": True,
                      "ShareeNode": True},
        "nodekey:f": {"HostName": "v6only", "UserID": 1001, "TailscaleIPs": ["fd7a:115c:a1e0::7"], "Online": True},
        "nodekey:g": "broken",
    },
}


class StatusTest(unittest.TestCase):
    def test_online_peers_are_asked(self) -> None:
        found = peers(STATUS)
        self.assertEqual([(p.name, p.address, p.own) for p in found],
                         [("e5", "100.64.0.1", True), ("friend", "100.64.0.5", False),
                          ("shared", "100.64.0.6", False), ("web-prod", "100.64.0.4", False)])
        self.assertEqual(peers(None), [])
        self.assertEqual(peers({"BackendState": "Running"}), [])

    def test_no_more_than_a_home_tailnet(self) -> None:
        many = {"Self": {"UserID": 1}, "Peer": {
            f"k{i}": {"HostName": f"pc{i:03}", "UserID": 2, "TailscaleIPs": [f"100.64.1.{i}"], "Online": True}
            for i in range(200)}}
        self.assertEqual(len(peers(many)), tailscale.MAX_PEERS)

    def test_own_pc(self) -> None:
        self.assertEqual(own_pc("100.64.0.1", STATUS).name, "e5")
        self.assertEqual(own_pc("fd7a:115c:a1e0:0::1", STATUS).name, "e5")
        self.assertEqual(own_pc("100.64.0.3", STATUS).name, "phone")  # offline in the list, yet it is asking
        for other in ("100.64.0.4", "100.64.0.5", "100.64.0.6", "100.64.0.9", "192.168.10.1", "pc"):
            self.assertIsNone(own_pc(other, STATUS), other)
        self.assertIsNone(own_pc("100.64.0.1", None))
        # A tagged worker has no account of its own to share with anyone.
        tagged = dict(STATUS, Self=dict(STATUS["Self"], Tags=["tag:gpu"]))
        self.assertIsNone(own_pc("100.64.0.1", tagged))
        # Without a user (a status this code does not understand), nobody is let in.
        self.assertIsNone(own_pc("100.64.0.1", {"Self": {}, "Peer": {"k": {"TailscaleIPs": ["100.64.0.1"]}}}))

    def test_tailnet_addresses(self) -> None:
        for address, inside in (("100.64.0.1", True), ("100.127.255.254", True), ("100.128.0.1", False),
                                ("192.168.10.1", False), ("fd7a:115c:a1e0::1", True), ("fe80::1%eth0", False),
                                ("pc", False)):
            self.assertEqual(is_tailnet_address(address), inside, address)


class ReadTest(unittest.TestCase):
    def read(self, stdout: bytes = b"", error: Exception | None = None, exe: str | None = "tailscale"):
        done = subprocess.CompletedProcess([], 0, stdout=stdout, stderr=b"")
        with mock.patch.object(tailscale, "find_exe", return_value=exe), \
                mock.patch.object(subprocess, "run", side_effect=error, return_value=done) as run:
            return tailscale.read_status(), run

    def test_reads_the_status(self) -> None:
        status, run = self.read(json.dumps(STATUS).encode())
        self.assertEqual(status["Self"]["HostName"], "pc4090")
        self.assertEqual(run.call_args.args[0], ["tailscale", "status", "--json"])

    def test_no_tailscale(self) -> None:
        status, run = self.read(exe=None)
        self.assertIsNone(status)
        run.assert_not_called()
        for stdout, error in ((b"not json", None), (b"[]", None), (b"", OSError("gone")),
                              (b"", subprocess.TimeoutExpired("tailscale", 5)),
                              (json.dumps(dict(STATUS, BackendState="Stopped")).encode(), None)):
            self.assertIsNone(self.read(stdout, error)[0])


if __name__ == "__main__":
    unittest.main()
