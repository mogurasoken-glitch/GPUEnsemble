import ipaddress
import threading
import unittest

from gpuensemble_planner import firewall, homelan
from gpuensemble_planner.firewall import Network


def net(profile, address="192.168.10.2", prefix=24, alias="Ethernet", name="Network 2"):
    return Network(alias, name, profile, [address], [ipaddress.ip_network(f"{address}/{prefix}", strict=False)])


class HomeNetworkTest(unittest.TestCase):
    def test_same_private_lan(self) -> None:
        self.assertEqual(homelan.home_network("192.168.10.7", [net("private")]).alias, "Ethernet")
        self.assertIsNotNone(homelan.home_network("192.168.10.7", [net("domain")]))

    def test_public_network_is_not_home(self) -> None:
        self.assertIsNone(homelan.home_network("192.168.10.7", [net("public")]))

    def test_other_lan_or_other_kinds_of_address(self) -> None:
        nets = [net("private")]
        for address in ("192.168.11.7",  # another network
                        "192.168.10.2",  # this PC itself
                        "8.8.8.8", "100.64.0.9", "127.0.0.1", "169.254.1.1", "fe80::1", "not an address"):
            self.assertIsNone(homelan.home_network(address, nets), address)

    def test_too_wide_a_network(self) -> None:
        self.assertIsNone(homelan.home_network("10.20.30.40", [net("private", "10.0.0.2", 8)]))
        self.assertIsNotNone(homelan.home_network("10.0.30.40", [net("private", "10.0.0.2", 16)]))

    def test_subnets_read_from_windows(self) -> None:
        networks, _ = firewall.parse([
            "net\tEthernet\tNetwork 2\tPrivate",
            "addr\tEthernet\t192.168.10.2\t24",
            "addr\tEthernet\tfe80::1234%5\t64",
            "addr\tEthernet\t192.168.10.3\tbroken",
        ])
        self.assertEqual(networks[0].subnets, [ipaddress.ip_network("192.168.10.0/24"), ipaddress.ip_network("fe80::/64")])
        self.assertIsNotNone(homelan.home_network("192.168.10.9", networks))

    def test_not_windows(self) -> None:
        self.assertEqual(firewall.read_networks(read=lambda exe: ["net\tEthernet\tN\tPrivate"])[0].profile, "private")


class HomeLanTest(unittest.TestCase):
    def test_reads_in_the_background_and_again(self) -> None:
        readings = [[net("private")], [net("public")]]
        changed = threading.Event()
        lan = homelan.HomeLan(read=lambda: readings.pop(0) if len(readings) > 1 else readings[0], refresh=0.05,
                              on_change=changed.set)
        lan.start()
        try:
            self.assertTrue(changed.wait(2))
            deadline = threading.Event()
            for _ in range(100):
                if lan.find("192.168.10.7") is None:
                    break
                deadline.wait(0.02)
            self.assertIsNone(lan.find("192.168.10.7"))  # the network became public
        finally:
            lan.close()

    def test_waits_for_the_first_reading(self) -> None:
        lan = homelan.HomeLan(read=lambda: [net("private")])
        lan.start()
        try:
            self.assertIsNotNone(lan.find("192.168.10.7", wait=2))
        finally:
            lan.close()

    def test_a_failed_reading_keeps_what_was_known(self) -> None:
        calls = []

        def read():
            calls.append(1)
            if len(calls) > 1:
                raise OSError("no PowerShell")
            return [net("private")]
        lan = homelan.HomeLan(read=read)
        lan.update()
        lan.update()
        self.assertIsNotNone(lan.find("192.168.10.7"))


if __name__ == "__main__":
    unittest.main()
