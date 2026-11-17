import unittest

from gpuensemble_planner import firewall
from gpuensemble_planner.firewall import Network, Rule, listening_on, notes, parse

HOME = Network("Ethernet", "Network 2", "private")
CAFE = Network("Wi-Fi", "Cafe", "public")


def allow(*profiles: str) -> Rule:
    return Rule("Allow", set(profiles), "python.exe")


def block(*profiles: str) -> Rule:
    return Rule("Block", set(profiles), "python.exe")


class ParseTest(unittest.TestCase):
    def test_lines(self) -> None:
        networks, rules = parse([
            "net\tEthernet\tNetwork 2\tPrivate\r\n",
            "net\tWi-Fi\tCafe\tPublic",
            "net\tvEthernet\tcorp\tDomainAuthenticated",
            "net\tbroken",
            "rule\tAllow\t2\tpython.exe",
            "rule\tBlock\t4\tpython.exe",
            "rule\tAllow\t2147483647\tPython",  # every profile
            "rule\tAllow\t3\tPython 3.12",
            "rule\tAllow\tx\tbroken",
            "WARNING: something else",
            "rules\tread",
        ])
        self.assertEqual([(n.alias, n.name, n.profile) for n in networks],
                         [("Ethernet", "Network 2", "private"), ("Wi-Fi", "Cafe", "public"),
                          ("vEthernet", "corp", "domain")])
        self.assertEqual([(r.action, r.profiles) for r in rules],
                         [("Allow", {"private"}), ("Block", {"public"}), ("Allow", {"domain", "private", "public"}),
                          ("Allow", {"domain", "private"})])

    def test_addresses(self) -> None:
        networks, _ = parse([
            "net\tEthernet\tNetwork 2\tPrivate",
            "net\tTailscale\tTailnet\tPrivate",
            "addr\tEthernet\t192.168.10.2",
            "addr\tEthernet\tfe80::1234",
            "addr\tTailscale\t100.64.0.2",
            "addr\tLoopback Pseudo-Interface 1\t127.0.0.1",
            "addr\tbroken",
        ])
        self.assertEqual([n.addresses for n in networks], [["192.168.10.2", "fe80::1234"], ["100.64.0.2"]])

    def test_rules_not_read(self) -> None:
        # Without "rules read" the rules are unknown (e.g. the COM object failed), not absent.
        self.assertEqual(parse(["net\tWi-Fi\tCafe\tPublic"])[1], None)
        self.assertEqual(parse(["rules\tread"])[1], [])


class NotesTest(unittest.TestCase):
    def test_allowed_on_a_private_network(self) -> None:
        self.assertEqual(notes([HOME], [allow("private"), block("public")]), [])

    def test_windows_will_ask(self) -> None:
        self.assertEqual(notes([HOME], []), [])  # no rule yet: Windows asks when the worker listens

    def test_public_network(self) -> None:
        # What "allow on private networks" leaves when the LAN is set to public.
        for rules in ([allow("private"), block("public")], []):
            [note] = notes([CAFE], rules)
            self.assertIn("Windows treats the network 'Cafe' (Wi-Fi) as public", note)
            self.assertIn("Settings > Network & internet > Wi-Fi > Private network", note)
        self.assertEqual(notes([CAFE], [allow("public")]), [])

    def test_blocked_after_cancel(self) -> None:
        [note] = notes([HOME], [Rule("Block", {"private", "public"}, "python")])
        self.assertIn("Windows Firewall blocks Python on the private network 'Network 2' (Ethernet) (rule 'python')",
                      note)
        self.assertIn("Allow an app through firewall, and tick Python for private networks", note)
        # A block rule wins over an allow rule.
        self.assertEqual(len(notes([HOME], [allow("private", "public", "domain"), block("private")])), 1)

    def test_rules_unknown(self) -> None:
        self.assertEqual(notes([HOME], None), [])
        [note] = notes([CAFE], None)
        self.assertIn("as public; if Python is allowed on private networks only", note)

    def test_rules_but_not_for_this_network(self) -> None:
        [note] = notes([HOME], [allow("domain")])
        self.assertIn("Python is not allowed in on the private network 'Network 2'", note)

    def test_only_the_network_the_worker_listens_on(self) -> None:
        # Two sites over Tailscale: the worker listened on the Tailscale address, but warned about the
        # public Ethernet, which no main PC used.
        lan = Network("Ethernet 12", "Network 7", "public", ["192.168.1.20", "fe80::1"])
        tailnet = Network("Tailscale", "Tailnet", "private", ["100.64.0.2", "fd7a:115c:a1e0::1"])
        both = [lan, tailnet]
        self.assertEqual(listening_on(both, "100.64.0.2"), [tailnet])
        self.assertEqual(listening_on(both, "fd7a:115c:a1e0:0::1"), [tailnet])
        self.assertEqual(listening_on(both, "0.0.0.0"), both)
        self.assertEqual(listening_on(both, None), both)
        self.assertEqual(listening_on(both, "10.9.9.9"), [])  # an address with no network profile: nothing to say
        unknown = [Network("Ethernet 12", "Network 7", "public")]
        self.assertEqual(listening_on(unknown, "100.64.0.2"), unknown)  # addresses not read: say what is known
        self.assertEqual(notes(listening_on(both, "100.64.0.2"), []), [])
        self.assertEqual(len(notes(listening_on(both, "192.168.1.20"), [])), 1)

    def test_reads_this_python(self) -> None:
        seen = []
        out = firewall.reachability_notes(lambda exe: seen.append(exe) or ["net\tWi-Fi\tCafe\tPublic", "rules\tread"], exe="C:\\p.exe")
        self.assertEqual(seen, ["C:\\p.exe"])
        self.assertEqual(len(out), 1)
        lines = ["net\tEthernet 12\tNetwork 7\tPublic", "addr\tEthernet 12\t192.168.1.20",
                 "net\tTailscale\tTailnet\tPrivate", "addr\tTailscale\t100.64.0.2", "rules\tread"]
        self.assertEqual(firewall.reachability_notes(lambda exe: lines, exe="C:\\p.exe", host="100.64.0.2"), [])
        self.assertEqual(len(firewall.reachability_notes(lambda exe: lines, exe="C:\\p.exe", host="0.0.0.0")), 1)


if __name__ == "__main__":
    unittest.main()
