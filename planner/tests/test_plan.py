import dataclasses
import os
import unittest

from gpuensemble_planner.devices import (DETECTED_FREE_MARGIN, GIB, Device, Host, detect_host, known_bandwidth,
                                         load_rig, parse_nvidia_smi, ram_for_weights)
from gpuensemble_planner.model import LayerProfile, ModelProfile
from gpuensemble_planner.plan import SEQUENCES, PlanError, make_plan, slowed_host

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def dense_model(n_layers: int, layer_gib: float, head_gib: float = 0.5) -> ModelProfile:
    return ModelProfile(
        name="dense",
        architecture="llama",
        layers=[LayerProfile(int(layer_gib * GIB), 0, 4096) for _ in range(n_layers)],
        head_bytes=int(head_gib * GIB),
        embed_bytes=int(0.4 * GIB),
    )


def moe_model(n_layers: int, dense_gib: float, expert_gib: float) -> ModelProfile:
    return ModelProfile(
        name="moe",
        architecture="moe",
        layers=[LayerProfile(int(dense_gib * GIB), int(expert_gib * GIB), 2048) for _ in range(n_layers)],
        head_bytes=int(1.0 * GIB),
        embed_bytes=int(1.0 * GIB),
        n_expert=128,
        n_expert_used=4,
    )


def rig() -> list[Device]:
    return [
        Device("ti", int(11 * GIB), 616e9),
        Device("super-a", int(8 * GIB), 496e9),
        Device("super-b", int(8 * GIB), 496e9),
    ]


HOST = Host(ram_bytes=int(52 * GIB), cpu_bandwidth=15e9, expert_bandwidth=10e9)


class PlanTest(unittest.TestCase):
    def check_contiguous(self, plan, n_layers: int) -> None:
        nxt = plan.cpu_layers
        for s in plan.stages:
            self.assertEqual(s.first_layer, nxt)
            nxt = s.last_layer + 1
        self.assertEqual(nxt, n_layers)

    def test_32b_fits_on_three_cards(self) -> None:
        model = dense_model(64, 0.272, head_gib=0.6)
        plan = make_plan(model, rig(), HOST, context=8192)
        self.assertEqual(plan.cpu_layers, 0)
        self.check_contiguous(plan, 64)
        for s in plan.stages:
            self.assertLessEqual(s.used_bytes, s.capacity_bytes)
        # The output head sits on the last stage only.
        self.assertGreater(plan.stages[-1].head_bytes, 0)
        self.assertTrue(all(s.head_bytes == 0 for s in plan.stages[:-1]))
        # About 28 tokens/s from published bandwidths.
        self.assertGreater(plan.tokens_per_second, 25)
        self.assertLess(plan.tokens_per_second, 32)

    def test_fast_card_gets_more_layers(self) -> None:
        # With plenty of room everywhere, everything goes to the fastest card.
        model = dense_model(10, 0.5)
        plan = make_plan(model, rig(), HOST, context=1024)
        by_name = {s.device: s.n_layers for s in plan.stages}
        self.assertEqual(by_name["ti"], 10)

    def test_too_big_for_vram_spills_a_cpu_prefix(self) -> None:
        model = dense_model(80, 0.48, head_gib=0.65)
        plan = make_plan(model, rig(), HOST, context=8192)
        self.assertGreater(plan.cpu_layers, 0)
        self.check_contiguous(plan, 80)
        self.assertTrue(any("CPU" in note for note in plan.notes))

    def test_moe_uses_every_card_for_experts(self) -> None:
        model = moe_model(36, 0.05, 1.577)
        plan = make_plan(model, rig(), HOST, context=8192)
        self.assertEqual(plan.mode, "moe")
        self.check_contiguous(plan, 36)
        for s in plan.stages:
            self.assertGreater(s.n_layers, 0, f"{s.device} was left idle")
            self.assertLessEqual(s.used_bytes, s.capacity_bytes)
        self.assertLessEqual(plan.cpu_bytes, HOST.ram_bytes)

    def test_moe_that_fits_whole_stays_whole(self) -> None:
        model = moe_model(8, 0.05, 1.0)
        plan = make_plan(model, rig(), HOST, context=4096)
        self.assertEqual(plan.mode, "whole")

    def test_slow_pcie_keeps_experts_on_cpu(self) -> None:
        model = moe_model(36, 0.05, 1.577)
        devs = [Device("ti", int(11 * GIB), 616e9, pcie_bandwidth=3.2e9),
                Device("super", int(8 * GIB), 496e9, pcie_bandwidth=1.6e9)]
        plan = make_plan(model, devs, HOST, context=8192, mode="moe")
        self.assertTrue(all(s.expert_path in ("", "cpu") for s in plan.stages))

    def test_fast_pcie_streams_experts(self) -> None:
        model = moe_model(36, 0.05, 1.577)
        devs = [Device("ti", int(11 * GIB), 616e9, pcie_bandwidth=25e9),
                Device("super", int(8 * GIB), 496e9, pcie_bandwidth=25e9)]
        auto = make_plan(model, devs, HOST, context=8192, mode="moe")
        self.assertIn("stream", {s.expert_path for s in auto.stages})
        cpu_host = Host(HOST.ram_bytes, HOST.cpu_bandwidth, HOST.expert_bandwidth, expert_path="cpu")
        cpu = make_plan(model, devs, cpu_host, context=8192, mode="moe")
        self.assertLess(auto.seconds_per_token, cpu.seconds_per_token)

    def test_stream_without_pcie_number_is_an_error(self) -> None:
        model = moe_model(36, 0.05, 1.577)
        host = Host(HOST.ram_bytes, HOST.cpu_bandwidth, HOST.expert_bandwidth, expert_path="stream")
        with self.assertRaises(PlanError):
            make_plan(model, rig(), host, context=8192, mode="moe")

    def test_prompt_tiebreak_puts_fast_link_first(self) -> None:
        # Same cards as the E5 rig, but the slow-link card would be first by
        # generation speed alone. A dense model with CPU layers streams those
        # layers over the first card's link for every prompt batch.
        model = dense_model(64, 0.287, head_gib=0.6)
        devs = [Device("ti", int(11 * GIB), 463e9, pcie_bandwidth=3.2e9),
                Device("super", int(8 * GIB), 362e9, pcie_bandwidth=1.6e9)]
        host = Host(int(52 * GIB), 34e9, 17e9, cpu_overhead=0.03)
        plan = make_plan(model, devs, host, context=640)
        self.assertGreater(plan.cpu_layers, 0)
        self.assertEqual(plan.stages[0].device, "ti")
        self.assertTrue(any("fastest PCIe link" in n for n in plan.notes))

    def test_experts_in_ram_slow_the_cpu(self) -> None:
        model = moe_model(36, 0.05, 1.577)
        devs = rig()[:2]
        host = dataclasses.replace(HOST, expert_path="cpu")
        flat = make_plan(model, devs, host, context=8192, mode="moe")
        self.assertGreater(flat.expert_ram_bytes, 20 * GIB)
        slow = dataclasses.replace(host, expert_slowdown_per_gib=0.016)
        plan = make_plan(model, devs, slow, context=8192, mode="moe")
        self.assertLess(plan.tokens_per_second, flat.tokens_per_second)
        # The plan is priced at the rate for its own experts in RAM.
        settled = make_plan(model, devs, slowed_host(slow, plan.expert_ram_bytes), context=8192, mode="moe")
        self.assertEqual(settled.expert_ram_bytes, plan.expert_ram_bytes)
        self.assertAlmostEqual(settled.seconds_per_token, plan.seconds_per_token)

    def test_slowed_host(self) -> None:
        host = dataclasses.replace(HOST, expert_slowdown_per_gib=0.016)
        self.assertAlmostEqual(slowed_host(host, 0).expert_bandwidth, 10e9)
        self.assertAlmostEqual(slowed_host(host, int(50 * GIB)).expert_bandwidth, 10e9 / 1.8)
        self.assertEqual(slowed_host(host, int(50 * GIB)).expert_slowdown_per_gib, 0.0)

    def test_no_experts_in_ram_no_slowdown(self) -> None:
        model = moe_model(4, 0.05, 0.5)
        host = dataclasses.replace(HOST, expert_slowdown_per_gib=0.016)
        plan = make_plan(model, rig(), host, context=2048)
        self.assertEqual(plan.expert_ram_bytes, 0)
        self.assertEqual(plan.seconds_per_token, make_plan(model, rig(), HOST, context=2048).seconds_per_token)

    def test_does_not_fit(self) -> None:
        model = moe_model(36, 0.05, 1.577)
        small_host = Host(ram_bytes=int(8 * GIB), cpu_bandwidth=15e9, expert_bandwidth=10e9)
        with self.assertRaises(PlanError):
            make_plan(model, rig(), small_host, context=8192)

    def test_recurrent_layers_need_room_for_their_state(self) -> None:
        # Four layers of 1 GiB on a card with room for 4.5 GiB of them and their memory.
        # A state of 96 MiB for each of the server's sequences is 1.5 GiB for the four layers:
        # only three fit (3.0 + 1.1), whatever the context, and the first goes to the CPU.
        card = [Device("card", int(5.7 * GIB), 500e9)]
        plain = ModelProfile("plain", "llama", [LayerProfile(GIB) for _ in range(4)], int(0.1 * GIB), int(0.1 * GIB))
        self.assertEqual(make_plan(plain, card, HOST, context=1024).cpu_layers, 0)
        hybrid = dataclasses.replace(plain, layers=[LayerProfile(GIB, state_bytes=96 * 1024**2) for _ in range(4)])
        plan = make_plan(hybrid, card, HOST, context=1024)
        self.assertEqual(plan.cpu_layers, 1)
        stage = plan.stages[0]
        self.assertEqual(stage.kv_bytes, 3 * SEQUENCES * 96 * 1024**2)
        self.assertLessEqual(stage.used_bytes, stage.capacity_bytes)
        self.assertEqual(make_plan(hybrid, card, HOST, context=8192).cpu_layers, 1)

    def test_sliding_window_layers_keep_only_their_window(self) -> None:
        # Like Gemma 4: five of every six layers look back at the last 1024 tokens only. llama.cpp keeps
        # room for 1024 tokens of each sequence and a batch on them (4 x 1024 + 512), the rest the context.
        whole = dense_model(12, 0.25)
        windowed = dataclasses.replace(whole, layers=[
            LayerProfile(layer.dense_bytes, 0, 4096, kv_window=0 if i % 6 == 5 else 1024)
            for i, layer in enumerate(whole.layers)])
        context = 32768
        kv_whole = sum(s.kv_bytes for s in make_plan(whole, rig(), HOST, context=context).stages)
        plan = make_plan(windowed, rig(), HOST, context=context)
        self.assertEqual(sum(s.kv_bytes for s in plan.stages), 2 * 4096 * context + 10 * 4096 * (4 * 1024 + 512))
        self.assertLess(sum(s.kv_bytes for s in plan.stages), kv_whole / 3)
        # A full context is read only by the two layers that look back at all of it.
        self.assertGreater(plan.tokens_per_second_full,
                           make_plan(whole, rig(), HOST, context=context).tokens_per_second_full)
        # A short context is held whole by every layer.
        short = make_plan(windowed, rig(), HOST, context=2048)
        self.assertEqual(sum(s.kv_bytes for s in short.stages), 12 * 4096 * 2048)

    def test_longer_context_needs_more_room(self) -> None:
        model = dense_model(64, 0.272, head_gib=0.6)
        short = make_plan(model, rig(), HOST, context=2048)
        long = make_plan(model, rig(), HOST, context=32768)
        self.assertGreater(sum(s.kv_bytes for s in long.stages), sum(s.kv_bytes for s in short.stages))


class DeviceTest(unittest.TestCase):
    def test_known_bandwidth_prefers_longest_name(self) -> None:
        self.assertEqual(known_bandwidth("NVIDIA GeForce RTX 2080 Ti"), 616e9)
        self.assertEqual(known_bandwidth("NVIDIA GeForce RTX 2080 SUPER"), 496e9)
        self.assertIsNone(known_bandwidth("Some Future Card"))

    def test_example_rig_loads(self) -> None:
        devices, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
        self.assertEqual(len(devices), 2)
        self.assertIsNotNone(devices[0].pcie_bandwidth)
        self.assertEqual(host.ram_bytes, 52 * GIB)
        self.assertAlmostEqual(host.expert_slowdown_per_gib, 0.016)


class DetectTest(unittest.TestCase):
    def test_nvidia_smi_with_pcie_links(self) -> None:
        # The E5 rig: the Ti drives the display and sits on Gen2 x8, the SUPER on a Gen2 x4 riser.
        out = ("0, NVIDIA GeForce RTX 2080 Ti, 11264, 10600, 2, 8\n"
               "1, NVIDIA GeForce RTX 2080 SUPER, 8192, 7980, 2, 4\n")
        ti, sup = parse_nvidia_smi(out)
        self.assertEqual((ti.llama_device, sup.llama_device), ("CUDA0", "CUDA1"))
        self.assertAlmostEqual(ti.pcie_bandwidth, 3.2e9)
        self.assertAlmostEqual(sup.pcie_bandwidth, 1.6e9)
        self.assertAlmostEqual(ti.bandwidth, 463e9)  # measured on this rig (gpu_table)
        self.assertAlmostEqual(sup.bandwidth, 362e9)
        # About the rig file's room: the idle SUPER total - 1 GiB, the busy Ti free - 0.55 GiB.
        self.assertEqual(sup.usable_bytes, 7 * GIB)
        self.assertEqual(ti.usable_bytes, 10600 * 1024**2 - DETECTED_FREE_MARGIN)

    def test_known_cards(self) -> None:
        # The longest name wins, and a notebook's card is not the desktop card of that name.
        for name, gbs in (("NVIDIA GeForce RTX 4090", 1008), ("NVIDIA GeForce RTX 4070 Ti SUPER", 672),
                          ("NVIDIA GeForce RTX 4070 Ti", 504), ("NVIDIA GeForce RTX 3080 Ti", 912),
                          ("NVIDIA GeForce RTX 3080", 760), ("NVIDIA GeForce RTX 5090", 1792),
                          ("NVIDIA GeForce RTX 2080 SUPER", 496)):
            self.assertEqual(known_bandwidth(name), gbs * 1e9, name)
        self.assertEqual(known_bandwidth("NVIDIA GeForce RTX 4090 Laptop GPU"), 576e9)
        self.assertIsNone(known_bandwidth("NVIDIA GeForce RTX 3050 A Laptop GPU"))
        (gpu,) = parse_nvidia_smi("0, NVIDIA GeForce RTX 4090, 24564, 23036, 4, 16\n")
        self.assertNotIn("guessed", gpu.name)
        self.assertAlmostEqual(gpu.bandwidth, 820e9)

    def test_nvidia_smi_without_link_info(self) -> None:
        (gpu,) = parse_nvidia_smi("0, Some Future GPU, 8192, 8000, [N/A], [N/A]\n")
        self.assertIsNone(gpu.pcie_bandwidth)
        self.assertIn("bandwidth guessed", gpu.name)

    def test_ram_for_weights_leaves_room_for_the_system(self) -> None:
        self.assertAlmostEqual(ram_for_weights(64 * GIB) / GIB, 51.2)
        self.assertEqual(ram_for_weights(32 * GIB), 24 * GIB)
        self.assertEqual(ram_for_weights(16 * GIB), 8 * GIB)
        self.assertEqual(ram_for_weights(8 * GIB), 4 * GIB)
        self.assertEqual(ram_for_weights(7 * GIB), int(3.5 * GIB))

    def test_detect_host_reads_this_computer(self) -> None:
        host = detect_host()
        self.assertGreater(host.ram_bytes, 0)
        self.assertEqual(host.threads, os.cpu_count())
        self.assertEqual(detect_host(ram_gib=40).ram_bytes, 40 * GIB)


class ExampleProfilesTest(unittest.TestCase):
    def test_examples_plan(self) -> None:
        devices, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
        for name in ("qwen3-32b-q4_k_m.json", "llama-70b-q4_k_m.json", "gpt-oss-120b.json", "gpt-oss-20b.json",
                     "qwen3-30b-a3b-q4_k_m.json"):
            with self.subTest(name=name):
                model = ModelProfile.from_json_file(os.path.join(EXAMPLES, name))
                plan = make_plan(model, devices, host, context=8192)
                self.assertGreater(plan.tokens_per_second, 0)


class Phase0CalibrationTest(unittest.TestCase):
    """The E5 rig file is fitted to measurements; keep its estimates near them.

    tg128 from llama-bench (CUDA 12.4, driver 560) running the planner's own
    llama.cpp arguments (phase 1a). Depth 0 is compared with the estimate for
    an empty context.
    """

    def plan(self, profile: str, context: int, **kw):
        devices, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
        host.expert_path = "cpu"  # measured with llama.cpp
        model = ModelProfile.from_json_file(os.path.join(EXAMPLES, profile))
        return make_plan(model, devices, host, context=context, **kw)

    def near(self, estimate: float, measured: float, tolerance: float = 0.12) -> None:
        self.assertGreater(estimate, measured * (1 - tolerance))
        self.assertLess(estimate, measured * (1 + tolerance))

    def test_qwen3_32b_short_context(self) -> None:
        # -ngl 60 -ts 36/24 -t 32: 5 layers on the CPU. (6 layers measured 7.06.)
        plan = self.plan("qwen3-32b-q4_k_m.json", 640)
        self.assertEqual(plan.cpu_layers, 5)
        self.near(plan.tokens_per_second_empty, 7.89)

    def above_unheld(self, estimate: float, measured: float) -> None:
        """For runs measured before `up` held the cards at full clocks (2026-11-16) and not
        measured again: the cards then clocked down while they waited for the CPU, and
        held they ran 14-35% faster in every placement on the E5's two cards with work on
        the CPU that was measured both ways (benchmarks/cpu-rates-held.md)."""
        self.assertGreater(estimate, measured)
        self.assertLess(estimate, measured * 1.35)

    def test_qwen3_32b_long_context(self) -> None:
        # -ngl 56 -ts 34/22 -ctk/-ctv q8_0 -t 32: 9 layers on the CPU.
        # Measured at depth 0 / 4096 / 7552 with the cards not held: 4.65 / 3.84 / 3.42.
        plan = self.plan("qwen3-32b-q4_k_m.json", 8192)
        self.assertEqual((plan.kv_type, plan.cpu_layers), ("q8_0", 9))
        self.above_unheld(plan.tokens_per_second_empty, 4.65)
        self.above_unheld(plan.tokens_per_second, 3.84)
        self.above_unheld(plan.tokens_per_second_full, 3.42)

    # The MoE runs below read the weights into RAM (-lm none --no-host, tg256),
    # as `up` does; with the file memory-mapped, page faults skew short runs.

    def test_gpt_oss_120b(self) -> None:
        # Both cards held at full clocks: -ts 8/29, experts of layers 0-3 and 8-30
        # on the CPU (43 GiB in RAM), -nopo 1 -fa 1 -t 32: 8.53 (6.33 not held;
        # 6.55 with -ts 6/31 before the cards were held).
        plan = self.plan("gpt-oss-120b.json", 8192, chunk=512)  # measured with chunks of 512
        self.near(plan.tokens_per_second_empty, 8.53)
        self.assertEqual(27, sum(round((1 - s.expert_fraction_on_gpu) * s.n_layers) for s in plan.stages))

    def test_gpt_oss_20b(self) -> None:
        # -ts 3/22, all on the GPUs: 100.3 (runs of 93-108).
        self.near(self.plan("gpt-oss-20b.json", 8192).tokens_per_second_empty, 100.3)

    def test_qwen3_30b_a3b(self) -> None:
        # -ts 22/27 -ot blk.(0-3) experts on the CPU, -t 32, cards not held: 43.6
        # (runs of 41.0-47.8; -t 16 gave 37.3).
        self.above_unheld(self.plan("qwen3-30b-a3b-q4_k_m.json", 8192).tokens_per_second_empty, 43.6)

    def test_llama_70b(self) -> None:
        # Both cards held: -ngl 32 -ts 14/18 -fa 1 -t 32 (48 layers on the CPU, the plan
        # has 49): 0.83 at depth 4096. At depth 0 it ran at 1.37 against 0.94 estimated:
        # Llama's wider layers run faster per byte on this CPU than Qwen3.8-27B's, which
        # the CPU rate is fitted to.
        self.near(self.plan("llama-70b-q4_k_m.json", 8192).tokens_per_second, 0.83)


if __name__ == "__main__":
    unittest.main()
