import sys
import unittest
from unittest import mock

from gpuensemble_planner import cardhold
from gpuensemble_planner.devices import GIB, Device
from gpuensemble_planner.plan import Plan, Stage


def _dev(name, llama_device=None, rpc=None):
    return Device(name, 8 * GIB, 400e9, llama_device=llama_device, rpc=rpc)


def _stage(device, first, last):
    return Stage(device, first, last, 0, 0, 0, 8 * GIB, 0.01)


def _plan(stages, cpu_layers=0):
    return Plan("m", "dense", 8192, stages, cpu_layers, 0, 0.05)


TI, SUPER, FAR = _dev("Ti", "CUDA0"), _dev("SUPER", "CUDA1"), _dev("4090", rpc="100.1.2.3:50052")
DEVICES = [TI, SUPER, FAR]


class CardsToHold(unittest.TestCase):
    def test_one_card_alone_waits_for_nothing(self):
        self.assertEqual(cardhold.cards_to_hold(_plan([_stage("Ti", 0, 35)]), DEVICES), [])

    def test_split_over_this_pcs_cards(self):
        plan = _plan([_stage("SUPER", 0, 9), _stage("Ti", 10, 35)])
        self.assertEqual(cardhold.cards_to_hold(plan, DEVICES), [1, 0])

    def test_one_card_and_the_cpu(self):
        self.assertEqual(cardhold.cards_to_hold(_plan([_stage("Ti", 4, 35)], cpu_layers=4), DEVICES), [0])

    def test_another_pcs_card_is_not_held_here(self):
        plan = _plan([_stage("SUPER", 0, 3), _stage("4090", 4, 16), _stage("Ti", 17, 35)])
        self.assertEqual(cardhold.cards_to_hold(plan, DEVICES), [1, 0])

    def test_empty_stages_and_non_cuda_cards_are_left_out(self):
        vk = _dev("vk", "Vulkan0")
        plan = _plan([_stage("SUPER", 0, -1), _stage("vk", 0, 9), _stage("4090", 10, 35)])
        self.assertEqual(cardhold.cards_to_hold(plan, DEVICES + [vk]), [])

    def test_only_the_remote_card(self):
        self.assertEqual(cardhold.cards_to_hold(_plan([_stage("4090", 0, 35)], cpu_layers=2), DEVICES), [])


class ParseUuids(unittest.TestCase):
    def test_nvidia_smi_output(self):
        out = ("0, GPU-3bf3c946-bd3b-4fe0-487e-fd429f46d8fd\n"
               "1, GPU-bdfb08da-eba4-b5e5-6702-74f8397134d4\n")
        self.assertEqual(cardhold.parse_uuids(out), {
            0: bytes.fromhex("3bf3c946bd3b4fe0487efd429f46d8fd"),
            1: bytes.fromhex("bdfb08daeba4b5e5670274f8397134d4")})

    def test_bad_lines_are_skipped(self):
        self.assertEqual(cardhold.parse_uuids("index, uuid\n0, [N/A]\n1, GPU-12\n"), {})


class PowerModeNote(unittest.TestCase):
    def test_nothing_to_say(self):
        self.assertIsNone(cardhold.power_mode_note(None))
        self.assertIsNone(cardhold.power_mode_note(cardhold.PREFER_MAXIMUM_PERFORMANCE))

    def test_other_modes_are_named(self):
        self.assertIn('"Adaptive"', cardhold.power_mode_note(0))
        self.assertIn("Prefer maximum performance", cardhold.power_mode_note(5))


class HoldFor(unittest.TestCase):
    def setUp(self):
        self.lines = []

    def test_nothing_held_for_one_card(self):
        with mock.patch.object(cardhold.CardHold, "open") as opened:
            self.assertIsNone(cardhold.hold_for(_plan([_stage("Ti", 0, 35)]), DEVICES, self.lines.append))
        opened.assert_not_called()
        self.assertEqual(self.lines, [])

    @unittest.skipUnless(sys.platform == "win32", "held on Windows only")
    def test_failure_is_a_note(self):
        plan = _plan([_stage("SUPER", 0, 9), _stage("Ti", 10, 35)])
        with mock.patch.object(cardhold.CardHold, "open", side_effect=OSError("no Vulkan loader")):
            self.assertIsNone(cardhold.hold_for(plan, DEVICES, self.lines.append))
        self.assertEqual(len(self.lines), 1)
        self.assertIn("no Vulkan loader", self.lines[0])

    @unittest.skipUnless(sys.platform == "win32", "held on Windows only")
    def test_held_cards_are_named(self):
        plan = _plan([_stage("SUPER", 0, 9), _stage("Ti", 10, 35)])
        hold = cardhold.CardHold()
        hold.held = [0, 1]
        with mock.patch.object(cardhold.CardHold, "open", return_value=hold), \
                mock.patch.object(cardhold, "power_mode", return_value=0):
            self.assertIs(cardhold.hold_for(plan, DEVICES, self.lines.append), hold)
        self.assertIn("keeping CUDA0, CUDA1 at full clocks", self.lines[0])
        self.assertIn('"Adaptive"', self.lines[1])

    @unittest.skipIf(sys.platform == "win32", "held on Windows")
    def test_not_held_elsewhere(self):
        plan = _plan([_stage("SUPER", 0, 9), _stage("Ti", 10, 35)])
        self.assertIsNone(cardhold.hold_for(plan, DEVICES, self.lines.append))


class OpenWithNothing(unittest.TestCase):
    def test_no_indices_opens_nothing(self):
        hold = cardhold.CardHold.open([])
        self.assertEqual(hold.held, [])
        hold.close()


if __name__ == "__main__":
    unittest.main()
