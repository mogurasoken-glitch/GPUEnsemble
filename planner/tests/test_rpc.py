"""GPUs on other PCs, reached through llama.cpp's RPC server (phase 2a)."""

import contextlib
import io
import json
import socket
import os
import tempfile
import threading
import time
import unittest
import unittest.mock

from gpuensemble_planner import worker
from gpuensemble_planner.cli import format_plan, main
from gpuensemble_planner.devices import (DEFAULT_NETWORK_BANDWIDTH, GIB, REMOTE_FREE_MARGIN, RPC_GATE_OVERHEAD,
                                         RPC_LAN_OVERHEAD, RPC_LOOPBACK_OVERHEAD, Device, Host, is_loopback,
                                         load_gpus, load_rig, logits_seconds)
from gpuensemble_planner.llamabin import Install
from gpuensemble_planner.llamacpp import llama_command, rpc_device_names
from gpuensemble_planner.model import LayerProfile, ModelProfile
from gpuensemble_planner.plan import (SLOW_NETWORK_BANDWIDTH, faster_on_other_pc, make_plan, remote_load_seconds,
                                      remote_notes)

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
HOST = Host(int(52 * GIB), 20e9, 17e9, expert_path="cpu")


def arg(cmd, flag):
    return cmd.args[cmd.args.index(flag) + 1]


def dense(n_layers: int, layer_gib: float) -> ModelProfile:
    return ModelProfile("d", "llama", [LayerProfile(int(layer_gib * GIB), 0, 4096)] * n_layers,
                        int(0.5 * GIB), int(0.3 * GIB))


def moe() -> ModelProfile:
    """12 GiB of experts over 24 layers: more than one 8 GiB card holds."""
    layers = [LayerProfile(int(0.05 * GIB), int(0.5 * GIB), 1024)] * 24
    return ModelProfile("m", "moe", layers, int(0.2 * GIB), int(0.2 * GIB), n_expert=32, n_expert_used=4)


def remote(name: str, endpoint: str, vram_gib: float = 8, overhead: float = 2e-3, llama: str = "CUDA0") -> Device:
    return Device(name, int(vram_gib * GIB), 362e9, llama_device=llama, rpc=endpoint, rpc_overhead=overhead)


class LoadGpusTest(unittest.TestCase):
    def test_remote_entries(self) -> None:
        devs = load_gpus([
            {"name": "ti", "vram_gib": 11, "bandwidth_gbs": 463},
            {"name": "a", "vram_gib": 8, "bandwidth_gbs": 362, "rpc": "192.168.10.2:50052"},
            {"name": "b", "vram_gib": 8, "bandwidth_gbs": 362, "rpc": "192.168.10.2:50052"},
            {"name": "c", "vram_gib": 8, "bandwidth_gbs": 362, "rpc": "127.0.0.1:50052", "llama_device": "CUDA1"},
            {"name": "d", "vram_gib": 8, "bandwidth_gbs": 362, "rpc": "192.168.10.3:50052",
             "rpc_overhead_ms": 1.2, "network_gbs": 1.1},
            {"name": "local2", "vram_gib": 8, "bandwidth_gbs": 362},
        ])
        by = {d.name: d for d in devs}
        # Local cards count among themselves, remote ones per server.
        self.assertEqual([by[n].llama_device for n in ("ti", "a", "b", "c", "d", "local2")],
                         ["CUDA0", "CUDA0", "CUDA1", "CUDA1", "CUDA0", "CUDA1"])
        self.assertFalse(by["ti"].is_remote)
        self.assertEqual((by["a"].rpc, by["a"].rpc_overhead), ("192.168.10.2:50052", RPC_LAN_OVERHEAD))
        self.assertEqual(by["c"].rpc_overhead, RPC_LOOPBACK_OVERHEAD)
        self.assertAlmostEqual(by["d"].rpc_overhead, 1.2e-3)
        self.assertEqual(by["d"].network_bandwidth, 1.1e9)
        self.assertEqual(by["ti"].rpc_overhead, 0.0)
        [gated] = load_gpus([{"name": "g", "vram_gib": 8, "bandwidth_gbs": 362, "rpc": "10.0.0.2:50052",
                              "gate": True}])
        self.assertAlmostEqual(gated.rpc_overhead, RPC_LAN_OVERHEAD + RPC_GATE_OVERHEAD)

    def test_bad_entries(self) -> None:
        with self.assertRaisesRegex(ValueError, "host:port"):
            load_gpus([{"name": "a", "vram_gib": 8, "bandwidth_gbs": 362, "rpc": "192.168.10.2"}])
        with self.assertRaisesRegex(ValueError, "own name"):
            load_gpus([{"name": "a", "vram_gib": 8, "bandwidth_gbs": 362},
                       {"name": "a", "vram_gib": 8, "bandwidth_gbs": 362, "rpc": "10.0.0.2:50052"}])

    def test_loopback(self) -> None:
        for e in ("127.0.0.1:50052", "localhost:1", "[::1]:50052", "::1:50052"):
            self.assertTrue(is_loopback(e), e)
        for e in ("192.168.10.2:50052", "pc2:50052", "0.0.0.0:50052"):
            self.assertFalse(is_loopback(e), e)

    def test_example_rig_matches_the_e5_rig(self) -> None:
        # The loopback rig is the E5 rig with the SUPER behind RPC: keep them in step.
        with open(os.path.join(EXAMPLES, "rig-e5-turing.json"), encoding="utf-8") as f:
            e5 = json.load(f)
        with open(os.path.join(EXAMPLES, "rig-e5-rpc-loopback.json"), encoding="utf-8") as f:
            rpc = json.load(f)
        self.assertEqual(e5["host"], rpc["host"])
        self.assertEqual(e5["gpus"][0], rpc["gpus"][0])
        for key in ("vram_gib", "bandwidth_gbs"):
            self.assertEqual(e5["gpus"][1][key], rpc["gpus"][1][key])


class RemotePlanTest(unittest.TestCase):
    def test_remote_stage_pays_the_round_trips(self) -> None:
        model = dense(40, 0.4)  # 16 GiB: needs both cards
        local = [Device("ti", int(11 * GIB), 463e9), Device("super", int(8 * GIB), 362e9)]
        far = [Device("ti", int(11 * GIB), 463e9), remote("super", "10.0.0.2:50052", overhead=2e-3)]
        a = make_plan(model, local, HOST, context=2048)
        b = make_plan(model, far, HOST, context=2048)
        self.assertEqual([s.n_layers for s in a.stages], [s.n_layers for s in b.stages])
        self.assertAlmostEqual(b.seconds_per_token - a.seconds_per_token, 2e-3, places=6)
        self.assertAlmostEqual(b.seconds_empty - a.seconds_empty, 2e-3, places=6)

    def test_the_head_on_another_pc_sends_the_logits_back(self) -> None:
        # Qwen3's 151,936 tokens: 0.6 MB with every token, about 5 ms at 1 Gb/s. Measured between two sites:
        # the same 32 layers answered in 26.3 ms with the head and in 20.1 ms without.
        nbytes = 151936 * 4
        far = remote("4090", "100.64.0.2:50052", vram_gib=24)
        self.assertAlmostEqual(logits_seconds(far, nbytes), nbytes / DEFAULT_NETWORK_BANDWIDTH)
        self.assertAlmostEqual(logits_seconds(far, nbytes), 5.2e-3, places=4)
        far.network_bandwidth = 12.5e6  # a 100 Mb/s line, given in the rig file
        self.assertAlmostEqual(logits_seconds(far, nbytes), 48.6e-3, places=4)
        self.assertEqual(logits_seconds(remote("super", "127.0.0.1:50052"), nbytes), 0.0)  # the same PC
        self.assertEqual(logits_seconds(Device("ti", int(11 * GIB), 463e9), nbytes), 0.0)

        # A model that needs both cards: with nothing to send back the faster order wins either way ...
        model = dense(40, 0.4)
        devs = [Device("ti", int(11 * GIB), 463e9, llama_device="CUDA0"),
                remote("far", "100.64.0.2:50052", vram_gib=11, overhead=9.5e-3)]
        devs[1].bandwidth = 463e9
        order = lambda plan: [s.device for s in plan.stages if s.n_layers]  # noqa: E731
        base = make_plan(model, devs, HOST, context=2048)
        # ... and with a vocabulary the head stays on this PC's card, and the plan costs what it did.
        model.n_vocab = 151936
        plan = make_plan(model, devs, HOST, context=2048)
        self.assertEqual(order(plan), ["far", "ti"])
        self.assertAlmostEqual(plan.seconds_per_token, base.seconds_per_token, places=6)
        # A model that fits on the far card alone pays for the logits there.
        small = dense(20, 0.2)
        alone = [remote("far", "100.64.0.2:50052", vram_gib=11, overhead=9.5e-3)]
        without = make_plan(small, alone, HOST, context=2048).seconds_per_token
        small.n_vocab = 151936
        self.assertAlmostEqual(make_plan(small, alone, HOST, context=2048).seconds_per_token - without,
                               nbytes / DEFAULT_NETWORK_BANDWIDTH, places=6)

    def test_small_model_stays_on_the_local_card(self) -> None:
        # Fits on either card; the remote one is faster on paper but not with its round trips.
        model = dense(20, 0.2)
        devs = [Device("slow", int(8 * GIB), 300e9), remote("fast", "10.0.0.2:50052", overhead=5e-3)]
        devs[1].bandwidth = 400e9
        plan = make_plan(model, devs, HOST, context=2048)
        used = [s.device for s in plan.stages if s.n_layers]
        self.assertEqual(used, ["slow"])
        self.assertEqual(remote_notes(plan, devs), [])
        devs[1].rpc_overhead = 0.0
        self.assertEqual([s.device for s in make_plan(model, devs, HOST, context=2048).stages if s.n_layers],
                         ["fast"])

    def test_a_far_cards_layers_do_not_hide_the_plans_that_end_on_this_pc(self) -> None:
        # With logits to send back, the cheapest split on paper was "every layer on the far card, the
        # head alone on this PC's card", which llama.cpp cannot run. The search used to settle on it in
        # every order that ends on this PC's card, and with those dropped only the orders that end on
        # the far card were left. Qwen3-8B, a 2080 Ti here and a 4090 behind a gate on the LAN:
        # 63 tokens/s planned on the 4090 where the 2080 Ti alone gives 80.
        used = lambda plan: [(s.device, s.n_layers) for s in plan.stages if s.n_layers]  # noqa: E731
        ti = Device("ti", int(11 * GIB), 463e9, llama_device="CUDA0")
        far = remote("far", "192.168.10.2:50052", vram_gib=24, overhead=3.1e-3)
        far.bandwidth = 756e9
        small = dense(36, 0.11)
        small.n_vocab = 151936
        alone = make_plan(small, [ti], HOST, context=2048)
        far_alone = make_plan(small, [far], HOST, context=2048)
        self.assertGreater(far_alone.seconds_per_token, alone.seconds_per_token)
        for devs in ([ti, far], [far, ti]):
            plan = make_plan(small, devs, HOST, context=2048)
            self.assertLessEqual(plan.seconds_per_token, alone.seconds_per_token + 1e-9)
            # The head is on a card that also holds a layer, and on this PC: no logits cross the network.
            self.assertEqual((plan.stages[-1].device, plan.stages[-1].head_bytes), ("ti", small.head_bytes))
            self.assertGreater(plan.stages[-1].n_layers, 0)
        # Between two sites the round trip alone costs more than the far card saves: all on this PC's card.
        far.rpc_overhead = 9.5e-3
        plan = make_plan(small, [ti, far], HOST, context=2048)
        self.assertEqual(used(plan), [("ti", 36)])
        self.assertAlmostEqual(plan.seconds_per_token, alone.seconds_per_token)
        # A model that only the far card holds keeps its last layer and the head on this PC's card:
        # one layer here costs less than the logits coming back with every token.
        big = dense(64, 0.27)
        big.n_vocab = 151936
        far_alone = make_plan(big, [far], HOST, context=2048)
        plan = make_plan(big, [ti, far], HOST, context=2048)
        self.assertEqual(used(plan), [("far", 63), ("ti", 1)])
        saved = far_alone.seconds_per_token - plan.seconds_per_token
        self.assertAlmostEqual(saved, logits_seconds(far, big.logits_bytes), delta=1.5e-3)
        cmd = llama_command(plan, big, [ti, far], tool="server", threads=8, ram_bytes=HOST.ram_bytes)
        self.assertEqual((arg(cmd, "-dev"), arg(cmd, "-ts")), ("RPC0,CUDA0", "63,2"))  # the head counts as a layer of the last card

    def test_experts_in_ram_cost_a_round_trip_per_layer(self) -> None:
        # MoE whose experts do not all fit: each layer whose experts stay in this
        # PC's RAM sends its activations here and back.
        model = moe()
        host = Host(int(52 * GIB), 20e9, 17e9, expert_path="cpu")
        near = make_plan(model, [remote("r", "127.0.0.1:50052", overhead=0.0)], host, context=2048, mode="moe")
        far = make_plan(model, [remote("r", "127.0.0.1:50052", overhead=1e-3)], host, context=2048, mode="moe")
        ram_layers = 24 - round(far.stages[0].expert_fraction_on_gpu * 24)
        self.assertGreater(ram_layers, 0)
        self.assertAlmostEqual(far.seconds_per_token - near.seconds_per_token, (1 + ram_layers) * 1e-3, places=6)

    def test_experts_never_stream_to_a_remote_card(self) -> None:
        model = moe()
        card = remote("r", "10.0.0.2:50052")
        card.pcie_bandwidth = 1e12  # its own PCIe link, on the other PC
        plan = make_plan(model, [card], Host(int(52 * GIB), 20e9, 17e9, expert_path="auto"), context=2048, mode="moe")
        self.assertEqual(plan.stages[0].expert_path, "cpu")

    def test_the_head_is_not_left_alone_on_an_unused_card(self) -> None:
        # With the head counted on the far card, one more layer of experts seemed to fit on the Ti.
        # llama.cpp is not told of a card without layers, so the head lands on the Ti after all.
        model = moe()
        ti = Device("ti", int(11.3 * GIB), 463e9, llama_device="CUDA0")
        far = Device("far", int(24 * GIB), 225e9, llama_device="CUDA0", rpc="100.64.0.2:50052", rpc_overhead=25e-3)
        plan = make_plan(model, [ti, far], HOST, context=2048)
        last = plan.stages[-1]
        self.assertEqual((last.device, last.n_layers), ("ti", 24))
        self.assertEqual(last.head_bytes, model.head_bytes)
        self.assertLessEqual(last.used_bytes, last.capacity_bytes)
        self.assertAlmostEqual(plan.tokens_per_second, 29.9, delta=0.2)  # 33.2 with the head counted elsewhere
        cmd = llama_command(plan, model, [ti, far], tool="server", threads=8, ram_bytes=HOST.ram_bytes)
        self.assertEqual((arg(cmd, "-dev"), cmd.rpc_servers), ("CUDA0", []))
        # A card that is the only one keeps its plan, layers or not.
        alone = make_plan(dense(4, 0.4), [Device("ti", int(11 * GIB), 463e9)], HOST, context=2048)
        self.assertEqual(alone.stages[-1].n_layers, 4)

    def test_a_worker_s_guess_is_replaced_by_what_this_pc_knows(self) -> None:
        # An older worker did not know the RTX 4090 and guessed; the main PC's table has it.
        entry = {"name": "GPU0 NVIDIA GeForce RTX 4090 (bandwidth guessed) on 100.64.0.2", "vram_gib": 24,
                 "bandwidth_gbs": 225, "rpc": "100.64.0.2:50052"}
        [card] = load_gpus([entry])
        self.assertEqual(card.name, "GPU0 NVIDIA GeForce RTX 4090 on 100.64.0.2")
        self.assertAlmostEqual(card.bandwidth, 820e9)
        # A card that nobody knows stays as the worker said.
        [card] = load_gpus([dict(entry, name="GPU0 Some Future GPU (bandwidth guessed) on 100.64.0.2")])
        self.assertIn("bandwidth guessed", card.name)
        self.assertAlmostEqual(card.bandwidth, 225e9)

    def test_load_note(self) -> None:
        model = dense(40, 0.4)
        far = remote("super", "10.0.0.2:50052")
        devs = [Device("ti", int(11 * GIB), 463e9), far]
        plan = make_plan(model, devs, HOST, context=2048)
        [note] = remote_notes(plan, devs)
        self.assertIn("on another PC (10.0.0.2:50052)", note)
        self.assertIn("over 1 Gb/s Ethernet", note)
        self.assertIn(note, plan.notes)
        far.network_bandwidth = 1.1e9
        self.assertIn("at 1.1 GB/s", remote_notes(plan, devs)[0])
        # At another site the link is not 1 Gb/s Ethernet, and a worker that can downloads its share itself.
        far.network_bandwidth, far.rpc_rtt, far.rpc_cache = None, 0.0075, True
        [note] = remote_notes(plan, devs)
        self.assertIn("go over the network the first time this model loads (at the speed of this PC's upload", note)
        self.assertNotIn("Ethernet", note)
        far.rpc_fetch = True
        [note] = remote_notes(plan, devs)
        self.assertIn("it downloads them itself if it can, and this PC sends them if not", note)
        self.assertNotIn("go over the network", note)
        far.rpc_rtt = 0.0004  # on the LAN they are sent, as before
        self.assertIn("over 1 Gb/s Ethernet", remote_notes(plan, devs)[0])
        far.rpc, far.rpc_rtt, far.rpc_cache, far.rpc_fetch = "127.0.0.1:50052", None, False, False
        [note] = remote_notes(plan, devs)
        self.assertIn("through the RPC server on this PC", note)
        self.assertNotIn("Ethernet", note)
        self.assertIn("every time the model loads", note)
        far.rpc_cache = True
        [note] = remote_notes(plan, devs)
        self.assertIn("the first time this model loads", note)
        self.assertIn("later loads send only their hashes", note)

    def test_cards_that_wait_for_another_pc_get_a_warning(self) -> None:
        # Measured: this PC's cards clock down while they wait for the other PC, which the estimate leaves out.
        model = dense(40, 0.4)
        ti = Device("ti", int(11 * GIB), 463e9)
        plan = make_plan(model, [ti, remote("super", "10.0.0.2:50052")], HOST, context=2048)
        self.assertEqual({s.device for s in plan.stages if s.n_layers}, {"ti", "super"})
        [note] = [n for n in plan.notes if "wait while the other PC" in n]
        self.assertIn("20-30% below this estimate", note)
        # Not when the other "PC" is a server on this one, when everything is on the other PC, or without one.
        for devs in ([ti, remote("super", "127.0.0.1:50052")],
                     [remote("far", "10.0.0.2:50052", vram_gib=24)],
                     [ti, Device("super", int(8 * GIB), 362e9)]):
            plan = make_plan(model, devs, HOST, context=2048)
            self.assertEqual([n for n in plan.notes if "wait while" in n], [], [d.name for d in devs])

    def test_a_model_that_another_pc_holds_alone_says_what_that_pc_would_do(self) -> None:
        # Between two sites a model that fits on the far card runs there from end to end, and this PC adds
        # a round trip and the logits to every token. Measured with Qwen3-32B on an RTX 4090: 25.2-25.95 tok/s
        # started here, 41.4-41.5 started on that PC; 14.4-15.6 ms per token apart, 14.7 by this count.
        model = dense(20, 0.2)
        model.n_vocab = 151936
        far = remote("far", "100.64.0.2:50052", vram_gib=24, overhead=9.5e-3)
        plan = make_plan(model, [far], HOST, context=2048)
        network = 9.5e-3 + 151936 * 4 / DEFAULT_NETWORK_BANDWIDTH
        self.assertEqual((plan.other_pc, plan.other_pc_rpc), ("100.64.0.2", "100.64.0.2:50052"))
        self.assertAlmostEqual(plan.seconds_on_other_pc, plan.seconds_per_token - network, places=9)
        # That is the plan the other PC would make for itself with the card as its own.
        own = make_plan(model, [Device("far", far.vram_bytes, far.bandwidth)], HOST, context=2048)
        self.assertAlmostEqual(plan.seconds_on_other_pc, own.seconds_per_token, places=9)
        self.assertAlmostEqual(plan.on_other_pc.seconds_full, own.seconds_full, places=9)
        self.assertEqual((own.other_pc, own.on_other_pc, own.seconds_on_other_pc), ("", None, 0.0))
        self.assertTrue(faster_on_other_pc(plan))
        [note] = [n for n in plan.notes if "started on that PC itself" in n]
        self.assertIn("the other PC (100.64.0.2) holds this model on its cards alone", note)
        self.assertIn(f"about {own.tokens_per_second:.0f} tokens/s instead of {plan.tokens_per_second:.0f}", note)

        # Also when the plan made here uses this PC's cards: the last layer and the head on one of them,
        # or, for a model this PC holds itself, every layer.
        ti = Device("ti", int(11 * GIB), 463e9, llama_device="CUDA0")
        far.bandwidth = 756e9
        for layers, used in ((64, ["far", "ti"]), (20, ["ti"])):
            model = dense(layers, 0.27)
            model.n_vocab = 151936
            plan = make_plan(model, [ti, far], HOST, context=2048)
            self.assertEqual([s.device for s in plan.stages if s.n_layers], used)
            own = make_plan(model, [Device("far", far.vram_bytes, far.bandwidth)], HOST, context=2048)
            self.assertEqual(plan.other_pc, "100.64.0.2")
            self.assertAlmostEqual(plan.seconds_on_other_pc, own.seconds_per_token, places=9)
            self.assertEqual(len([n for n in plan.notes if "started on that PC itself" in n]), 1)

        # Two cards of one other PC: a round trip for each.
        big = dense(40, 0.3)
        pair = [remote("a", "10.0.0.2:50052", overhead=1e-3, llama="CUDA0"),
                remote("b", "10.0.0.2:50052", overhead=1e-3, llama="CUDA1")]
        plan = make_plan(big, pair, HOST, context=2048)
        self.assertEqual(plan.other_pc, "10.0.0.2")
        self.assertAlmostEqual(plan.seconds_per_token - plan.seconds_on_other_pc, 2e-3, places=9)
        # 2 ms of some 40: not worth a note, nor a word in the table of `recommend`.
        self.assertFalse(faster_on_other_pc(plan))
        self.assertEqual([n for n in plan.notes if "started on that PC" in n], [])

        # Not when the model needs this PC's cards too, or two other PCs, or for a server on this PC;
        # nor when the other PC would need its CPU, which this PC knows nothing about.
        for devs in ([ti, remote("super", "10.0.0.2:50052")],
                     [remote("a", "10.0.0.2:50052"), remote("b", "10.0.0.3:50052")],
                     [remote("a", "127.0.0.1:50052"), remote("b", "127.0.0.1:50053")],
                     [remote("a", "10.0.0.2:50052")]):
            plan = make_plan(big, devs, HOST, context=2048)
            self.assertEqual((plan.other_pc, plan.on_other_pc, plan.seconds_on_other_pc), ("", None, 0.0),
                             [d.name for d in devs])
            self.assertEqual([n for n in plan.notes if "started on that PC" in n], [])

    def test_time_allowed_for_sending_the_weights(self) -> None:
        # Two sites over Tailscale: 21.6 GiB took 13 min of an 804 s load, close to giving up at 900 s.
        model = dense(40, 0.4)
        far = remote("super", "100.64.0.2:50052")
        devs = [Device("ti", int(11 * GIB), 463e9), far]
        plan = make_plan(model, devs, HOST, context=2048)
        [stage] = [s for s in plan.stages if s.device == "super"]
        self.assertGreater(stage.weight_bytes, 6 * GIB)
        self.assertAlmostEqual(remote_load_seconds(plan, devs), stage.weight_bytes / SLOW_NETWORK_BANDWIDTH)
        self.assertGreater(remote_load_seconds(plan, devs), 600)
        far.rpc = "127.0.0.1:50052"  # a server on this PC sends nothing over the network
        self.assertEqual(remote_load_seconds(plan, devs), 0)
        local = [Device("ti", int(11 * GIB), 463e9)]
        self.assertEqual(remote_load_seconds(make_plan(dense(10, 0.4), local, HOST, context=2048), local), 0)

    def test_the_worker_says_it_keeps_the_weights(self) -> None:
        detected_cards = [Device("GPU0 RTX 2080 SUPER", int(8 * GIB), 372e9, llama_device="CUDA0")]
        [entry] = worker.rig_entries(detected_cards, "10.0.0.2:50052", cache=True)
        self.assertTrue(entry["cache"])
        self.assertNotIn("cache", worker.rig_entries(detected_cards, "10.0.0.2:50052")[0])
        self.assertTrue(load_gpus([entry])[0].rpc_cache)


class LoopbackCalibrationTest(unittest.TestCase):
    def test_gpt_oss_20b(self) -> None:
        # llama-bench -fa 1 -lm none --no-host 1, tg128: --rpc 127.0.0.1:50052
        # -dev RPC0/CUDA0 -ts 3/22 ran 92.09, against 99.46 with both cards local.
        devices, host = load_rig(os.path.join(EXAMPLES, "rig-e5-rpc-loopback.json"))
        host.expert_path = "cpu"
        model = ModelProfile.from_json_file(os.path.join(EXAMPLES, "gpt-oss-20b.json"))
        plan = make_plan(model, devices, host, context=8192)
        self.assertGreater(plan.tokens_per_second_empty, 92.09 * 0.88)
        self.assertLess(plan.tokens_per_second_empty, 92.09 * 1.12)
        cmd = llama_command(plan, model, devices)
        self.assertEqual(cmd.args[:2], ["--rpc", "127.0.0.1:50052"])
        self.assertEqual((arg(cmd, "-dev"), arg(cmd, "-ts")), ("RPC0,CUDA0", "3,22"))


class RpcCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = dense(40, 0.4)

    def plan_on(self, devs):
        return make_plan(self.model, devs, HOST, context=2048)

    def test_names_follow_the_servers_and_their_cards(self) -> None:
        devs = [
            remote("p2-a", "10.0.0.2:50052", vram_gib=3, overhead=0, llama="CUDA0"),
            remote("p2-b", "10.0.0.2:50052", vram_gib=3, overhead=0, llama="CUDA1"),
            remote("p3-a", "10.0.0.3:50052", vram_gib=12, overhead=0),
            Device("ti", int(11 * GIB), 463e9, llama_device="CUDA0"),
        ]
        devs[0].bandwidth = devs[1].bandwidth = devs[2].bandwidth = 463e9
        plan = self.plan_on(devs)
        servers, names = rpc_device_names(plan, devs)
        used = {s.device for s in plan.stages if s.n_layers}
        expected_servers = [e for e in ("10.0.0.2:50052", "10.0.0.3:50052")
                            if any(d.rpc == e and d.name in used for d in devs)]
        self.assertEqual(servers, expected_servers)
        self.assertEqual(list(names.values()), [f"RPC{i}" for i in range(len(names))])
        cmd = llama_command(plan, self.model, devs)
        self.assertEqual(cmd.args[:2], ["--rpc", ",".join(servers)])

    def test_every_card_of_a_used_server_is_numbered(self) -> None:
        # The server offers both of its cards, so the second is RPC1 even if
        # only it is used; a server the plan leaves out is not contacted.
        devs = [
            Device("ti", int(11 * GIB), 463e9, llama_device="CUDA0"),
            remote("idle", "10.0.0.2:50052", vram_gib=1.2, llama="CUDA0"),
            remote("busy", "10.0.0.2:50052", llama="CUDA1"),
            remote("off", "10.0.0.3:50052", vram_gib=1.2),
        ]
        plan = self.plan_on(devs)
        self.assertEqual({s.device for s in plan.stages if s.n_layers}, {"ti", "busy"})
        servers, names = rpc_device_names(plan, devs)
        self.assertEqual(servers, ["10.0.0.2:50052"])
        self.assertEqual(names, {"idle": "RPC0", "busy": "RPC1"})
        cmd = llama_command(plan, self.model, devs, tool="bench")
        self.assertEqual(sorted(arg(cmd, "-dev").split("/")), ["CUDA0", "RPC1"])

    def test_local_plan_has_no_rpc(self) -> None:
        devs = [Device("ti", int(11 * GIB), 463e9, llama_device="CUDA0"),
                Device("super", int(8 * GIB), 362e9, llama_device="CUDA1")]
        self.assertNotIn("--rpc", llama_command(self.plan_on(devs), self.model, devs).args)


class FakeInstall:
    def __init__(self, folder: str) -> None:
        self.folder = folder

    def tool(self, name: str) -> str:
        return Install("b1", "cuda", self.folder).tool(name)


def detected() -> list[Device]:
    return [
        Device("GPU0 NVIDIA GeForce RTX 2080 SUPER", int(8 * GIB), 372e9, llama_device="CUDA0"),
        Device("GPU1 NVIDIA GeForce RTX 2080 SUPER", int(7.5 * GIB), 372e9, llama_device="CUDA1"),
    ]


class WorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.logs: list[str] = []
        self.ran: list[list[str]] = []

    def start(self, host="127.0.0.1", wanted=None, cache=False, server="ggml-rpc-server.exe", run=None, port=50052,
              **kw):
        if server:
            open(os.path.join(self.dir, server), "w").close()
        out = io.StringIO()

        def run_(argv, log):
            self.ran.append(argv)
            return 0 if run is None else run(argv, log)

        kw.setdefault("reachability", lambda host: [])  # never read this computer's firewall in a test
        code = worker.start(host, port, wanted, cache, detect=detected, install=lambda: FakeInstall(self.dir),
                            log=self.logs.append, out=out, run=run_, **kw)
        return code, out.getvalue()

    def test_serves_every_card_and_prints_rig_entries(self) -> None:
        code, out = self.start(host="192.168.10.2", cache=True, allow_any=True)
        self.assertEqual(code, 0)
        [argv] = self.ran
        self.assertEqual(argv[1:], ["-H", "192.168.10.2", "-p", "50052", "-d", "CUDA0,CUDA1", "-c"])
        self.assertTrue(argv[0].endswith("ggml-rpc-server.exe"))
        entries = json.loads(out)["gpus"]
        self.assertEqual([e["llama_device"] for e in entries], ["CUDA0", "CUDA1"])
        self.assertEqual({e["rpc"] for e in entries}, {"192.168.10.2:50052"})
        self.assertTrue(any("no password" in line for line in self.logs))
        # What the main PC reads back is the same cards, planned with 1 GiB more left free than
        # that PC would leave itself: its desktop and other programs use the cards too.
        back = load_gpus(entries)
        self.assertEqual([d.usable_bytes for d in back], [d.usable_bytes - REMOTE_FREE_MARGIN for d in detected()])
        self.assertEqual([d.rpc_overhead for d in back], [RPC_LAN_OVERHEAD] * 2)
        # A server on this PC itself serves this PC's own cards: the same room.
        here = load_gpus([dict(e, rpc="127.0.0.1:50052") for e in entries])
        self.assertEqual([d.usable_bytes for d in here], [d.usable_bytes for d in detected()])

    def test_room_now_against_room_at_the_start(self) -> None:
        def cards(*gib):
            return [Device(f"GPU{i} NVIDIA GeForce RTX 4090", int(g * GIB), 756e9, llama_device=f"CUDA{i}")
                    for i, g in enumerate(gib)]

        start = cards(23.3, 23.3, 12.0, 23.3)
        # The worker's own server holds about 1 GiB: not a reason to offer less. Another program's
        # 10 GiB is; and a card that another program gave back has more room than at the start.
        now = worker.current_devices(start, cards(22.3, 13.3, 23.3))
        self.assertEqual([round(d.vram_bytes / GIB, 1) for d in now], [23.3, 13.3, 23.3, 23.3])
        self.assertEqual([d.llama_device for d in now], ["CUDA0", "CUDA1", "CUDA2", "CUDA3"])  # a missing card stays

    def test_cards_are_read_again_for_a_main_pc_that_asks_later(self) -> None:
        now = [0.0]
        free = [8.0]
        reads = []
        logs: list[str] = []

        def detect():
            reads.append(now[0])
            if free[0] is None:
                raise RuntimeError("nvidia-smi failed")
            return [Device("GPU0 NVIDIA GeForce RTX 2080 SUPER", int(free[0] * GIB), 372e9, llama_device="CUDA0")]

        live = worker.LiveEntries(detect(), detect, lambda ds: [{"vram_gib": round(d.vram_bytes / GIB, 2)} for d in ds],
                                  logs.append, ttl=2.0, wait=5.0, clock=lambda: now[0])
        reads.clear()
        free[0] = 3.0
        self.assertEqual(live(), [{"vram_gib": 8.0}])  # just read
        self.assertEqual(reads, [])
        now[0] = 2.5
        self.assertEqual(live(), [{"vram_gib": 3.0}])
        self.assertEqual(live(), [{"vram_gib": 3.0}])  # the main PC's next questions get the same reading
        self.assertEqual(reads, [2.5])
        self.assertEqual(len(logs), 1)
        self.assertIn("3.0 GiB for GPU0 NVIDIA GeForce RTX 2080 SUPER (8.0 before)", logs[0])
        # nvidia-smi fails: the last reading stands, and nothing is said.
        now[0] = 10.0
        free[0] = None
        self.assertEqual(live(), [{"vram_gib": 3.0}])
        self.assertEqual(len(logs), 1)

    def test_a_slow_reading_does_not_hold_the_answer(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        now = [0.0]

        def detect():
            release.wait(10)
            return [Device("GPU0 NVIDIA GeForce RTX 2080 SUPER", int(3 * GIB), 372e9, llama_device="CUDA0")]

        start = [Device("GPU0 NVIDIA GeForce RTX 2080 SUPER", int(8 * GIB), 372e9, llama_device="CUDA0")]
        live = worker.LiveEntries(start, detect, lambda ds: [round(d.vram_bytes / GIB, 2) for d in ds],
                                  ttl=2.0, wait=0.05, clock=lambda: now[0])
        now[0] = 3.0
        began = time.monotonic()
        self.assertEqual(live(), [8.0])  # the last reading, after a short wait
        self.assertEqual(live(), [8.0])  # and no second nvidia-smi while the first still runs
        self.assertLess(time.monotonic() - began, 5)
        reading = live.reading
        release.set()
        reading.join(5)
        self.assertEqual(live(), [3.0])

    def test_chosen_cards_in_the_order_given(self) -> None:
        code, out = self.start(wanted=["CUDA1"])
        self.assertEqual(self.ran[0][-1], "CUDA1")
        [entry] = json.loads(out)["gpus"]
        self.assertEqual((entry["llama_device"], entry["rpc"]), ("CUDA1", "127.0.0.1:50052"))
        self.assertFalse(any("no password" in line for line in self.logs))

    def test_old_server_name(self) -> None:
        self.start(server="rpc-server")
        self.assertTrue(self.ran[0][0].endswith("rpc-server"))

    def test_every_network(self) -> None:
        code, out = self.start(host="0.0.0.0", allow_any=True)
        self.assertEqual(self.ran[0][1:3], ["-H", "0.0.0.0"])
        rpc = json.loads(out)["gpus"][0]["rpc"]
        self.assertNotIn("0.0.0.0", rpc)
        self.assertTrue(any("every network" in line for line in self.logs))

    def test_firewall_notes_on_a_network(self) -> None:
        def run(argv, log):
            for _ in range(250):  # the notes are read in the background
                if any(line.startswith("note: ") for line in self.logs):
                    break
                time.sleep(0.02)
            return 0

        asked = []
        self.start(host="0.0.0.0", allow_any=True, run=run, reachability=lambda host: asked.append(host) or ["public network"])
        self.assertIn("note: public network", self.logs)
        self.start(host="127.0.0.1", reachability=lambda host: asked.append(host) or [])
        self.assertEqual(asked, ["0.0.0.0"])  # 127.0.0.1 is reached from this PC only: nothing to check

    def test_cache_stays_in_the_home(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"GPUENSEMBLE_HOME": self.dir}):
            os.environ.pop("LLAMA_CACHE", None)
            env = worker.server_env(cache=True)
            self.assertEqual(env["LLAMA_CACHE"], os.path.join(self.dir, "rpc-cache"))
            self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertNotIn("LLAMA_CACHE", worker.server_env(cache=False))
            os.environ["LLAMA_CACHE"] = "elsewhere"
            self.assertEqual(worker.server_env(cache=True)["LLAMA_CACHE"], "elsewhere")

    def test_unreachable_servers(self) -> None:
        listening = socket.socket()
        listening.bind(("127.0.0.1", 0))
        listening.listen(1)
        self.addCleanup(listening.close)
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        free_port = closed.getsockname()[1]
        closed.close()
        up = f"127.0.0.1:{listening.getsockname()[1]}"
        down = f"127.0.0.1:{free_port}"
        self.assertEqual(worker.unreachable_servers([up, down, "nohost:port"], timeout=2), [down, "nohost:port"])
        self.assertEqual(worker.unreachable_servers([]), [])

    def test_lan_needs_allow(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs --allow"):
            self.start(host="192.168.10.2")
        with self.assertRaisesRegex(ValueError, "not an IP address"):
            self.start(host="192.168.10.2", allow=["pc1"])
        self.assertEqual(self.ran, [])

    def test_gate_in_front_of_the_server(self) -> None:
        # The server moves to 127.0.0.1 on a port of its own; the gate takes
        # the worker's address and lets in only --allow.
        port = worker.free_ports()
        seen = {}

        def fake_server(argv, log):
            inner = int(argv[argv.index("-p") + 1])
            seen["argv"] = argv
            with socket.socket() as srv:
                srv.bind(("127.0.0.1", inner))
                srv.listen(1)
                with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
                    conn, _ = srv.accept()
                    with conn:
                        c.sendall(b"ping")
                        seen["got"] = conn.recv(4)
            return 0

        code, out = self.start(port=port, allow=["127.0.0.1"], run=fake_server)
        self.assertEqual(code, 0)
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("-H") + 1], "127.0.0.1")
        self.assertNotEqual(int(argv[argv.index("-p") + 1]), port)
        self.assertEqual(seen["got"], b"ping")
        entry = json.loads(out)["gpus"][0]
        self.assertEqual((entry["rpc"], entry["gate"]), (f"127.0.0.1:{port}", True))
        self.assertAlmostEqual(load_gpus([entry])[0].rpc_overhead, RPC_LOOPBACK_OVERHEAD + RPC_GATE_OVERHEAD)
        self.assertTrue(any("lets in 127.0.0.1/32" in line for line in self.logs))
        # The gate is gone once the worker stops.
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=2).close()

    def test_errors(self) -> None:
        with self.assertRaisesRegex(OSError, "no RPC server"):
            self.start(server=None)
        with self.assertRaisesRegex(ValueError, "no GPU CUDA5"):
            self.start(wanted=["CUDA5"])
        self.assertEqual(self.ran, [])


class CliRemoteTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.remote = os.path.join(tmp.name, "pc2.json")
        with open(self.remote, "w", encoding="utf-8") as f:
            json.dump({"gpus": [{"name": "SUPER on pc2", "vram_gib": 8, "bandwidth_gbs": 362,
                                 "rpc": "192.168.10.2:50052", "llama_device": "CUDA0"}]}, f)
        self.model = os.path.join(tmp.name, "dense.json")
        with open(self.model, "w", encoding="utf-8") as f:
            f.write(dense(60, 0.4).to_json())  # 24 GiB: more than the Ti and SUPER of the E5 rig

    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_remote_gpus_join_the_rig(self) -> None:
        rig = os.path.join(EXAMPLES, "rig-e5-turing.json")
        code, out, err = self.cli("args", self.model, "--rig", rig, "--remote", self.remote, "--ctx", "2048")
        self.assertEqual(code, 0, err)
        self.assertIn("SUPER on pc2", out)
        self.assertIn("--rpc 192.168.10.2:50052", out)
        self.assertIn("RPC0", out)
        self.assertIn("remote: 1 GPU(s)", err)

    def test_long_names_keep_the_columns(self) -> None:
        name = "GPU1 NVIDIA GeForce RTX 2080 SUPER on 127.0.0.1"
        plan = make_plan(dense(40, 0.4), [Device("ti", int(11 * GIB), 463e9), remote(name, "127.0.0.1:50052")],
                         HOST, context=2048)
        lines = format_plan(plan).splitlines()
        header = next(line for line in lines if line.startswith("stage"))
        row = next(line for line in lines if line.startswith(name))
        end = header.index("layers") + len("layers")
        [stage] = [st for st in plan.stages if st.device == name]
        self.assertEqual(row[end - 1], str(stage.last_layer)[-1])
        self.assertEqual(row[end], " ")

    def test_bad_remote_files(self) -> None:
        rig = os.path.join(EXAMPLES, "rig-e5-turing.json")
        with open(self.remote, "w", encoding="utf-8") as f:
            json.dump({"gpus": [{"name": "RTX 2080 Ti 11GB", "vram_gib": 8, "bandwidth_gbs": 362,
                                 "rpc": "192.168.10.2:50052"}]}, f)
        code, _, err = self.cli("plan", self.model, "--rig", rig, "--remote", self.remote)
        self.assertEqual(code, 1)
        self.assertIn("same name", err)
        with open(self.remote, "w", encoding="utf-8") as f:
            json.dump({"gpus": [{"name": "x", "vram_gib": 8, "bandwidth_gbs": 362}]}, f)
        code, _, err = self.cli("plan", self.model, "--rig", rig, "--remote", self.remote)
        self.assertEqual(code, 1)
        self.assertIn("needs \"rpc\"", err)


if __name__ == "__main__":
    unittest.main()
