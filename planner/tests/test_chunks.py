"""Longer prompt chunks (llama.cpp -ub) for plans that keep weights in RAM (engine/PLAN.md)."""
import dataclasses
import os
import unittest
import unittest.mock

from gpuensemble_planner.devices import load_rig
from gpuensemble_planner.llamacpp import llama_command
from gpuensemble_planner.model import ModelProfile
from gpuensemble_planner.plan import make_plan

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def arg(cmd, flag):
    return cmd.args[cmd.args.index(flag) + 1]


class ChunkTest(unittest.TestCase):
    def rig(self):
        devices, host = load_rig(os.path.join(EXAMPLES, "rig-e5-turing.json"))
        host.expert_path = "cpu"
        return devices, host

    def plan(self, profile: str, context: int = 8192, devices=None, **kw):
        devs, host = self.rig()
        devs = devs if devices is None else devices
        model = ModelProfile.from_json_file(os.path.join(EXAMPLES, profile))
        return make_plan(model, devs, host, context=context, **kw), model, devs

    def test_experts_in_ram_get_4096_when_generation_costs_little(self) -> None:
        # E5 rig, gpt-oss-120b under llama-server (engine/prototype/RESULTS-server-stock.md, result 5):
        # with chunks of 4096 an 8000-token prompt answers after 41 s instead of 166 s, a 190-token
        # follow-up after 5.2 s instead of 4.4 s, generation at 10.5 tokens/s instead of 11.0.
        plan, model, devs = self.plan("gpt-oss-120b.json", context=12288)
        self.assertEqual(plan.ubatch, 4096)
        cmd = llama_command(plan, model, devs)
        self.assertEqual(arg(cmd, "-ub"), "4096")
        self.assertEqual(cmd.env["GGML_OP_OFFLOAD_MIN_BATCH"], "600")
        short = self.plan("gpt-oss-120b.json", context=12288, chunk=512)[0]
        self.assertLessEqual(plan.seconds_per_token, short.seconds_per_token * 1.10)
        # asked for 512: the experts are computed on the CPU, as before
        cmd = llama_command(short, model, devs, tool="bench")
        self.assertNotIn("-ub", cmd.args)
        self.assertNotIn("GGML_OP_OFFLOAD_MIN_BATCH", cmd.env)
        # Qwen3-30B-A3B would lose about 15% of its generation by the estimate (not measured): it keeps 512.
        self.assertEqual(self.plan("qwen3-30b-a3b-q4_k_m.json")[0].ubatch, 512)
        # and so does a context that 4096 does not leave room in
        self.assertEqual(self.plan("gpt-oss-120b.json", context=4096)[0].ubatch, 512)

    def test_experts_in_ram_with_a_longer_chunk_copy_to_the_gpu(self) -> None:
        plan, model, devs = self.plan("gpt-oss-120b.json", chunk=1024)
        short = self.plan("gpt-oss-120b.json", chunk=512)[0]
        self.assertEqual((plan.ubatch, short.ubatch), (1024, 512))
        self.assertLessEqual(plan.seconds_per_token, short.seconds_per_token * 1.06)
        self.assertTrue(any("chunks of 1024" in n for n in plan.notes))
        for tool in ("bench", "server"):
            cmd = llama_command(plan, model, devs, tool=tool)
            self.assertEqual((arg(cmd, "-b"), arg(cmd, "-ub")), ("4096", "1024"))
            self.assertNotIn("-nopo", cmd.args)
            self.assertNotIn("--no-op-offload", cmd.args)
            self.assertIn("-ot", cmd.args)
            # Short prompts are still computed on the CPU.
            self.assertEqual(cmd.env["GGML_OP_OFFLOAD_MIN_BATCH"], "600")

    def test_cpu_layers_get_longer_chunks_when_generation_keeps_its_speed(self) -> None:
        # Llama 3.3 70B loses two layers to the CPU for chunks of 1024, 3.6% by the estimate:
        # taken when that much is allowed.
        with unittest.mock.patch("gpuensemble_planner.plan.CHUNK_GEN_COST", 0.05):
            plan, model, devs = self.plan("llama-70b-q4_k_m.json")
        short = self.plan("llama-70b-q4_k_m.json", chunk=512)[0]
        self.assertEqual(plan.ubatch, 1024)
        self.assertLessEqual(plan.seconds_per_token, short.seconds_per_token * 1.05)
        cmd = llama_command(plan, model, devs)
        self.assertEqual((arg(cmd, "-b"), arg(cmd, "-ub")), ("4096", "1024"))
        self.assertNotIn("--no-op-offload", cmd.args)
        self.assertNotIn("GGML_OP_OFFLOAD_MIN_BATCH", cmd.env)

    def test_cpu_layers_keep_512_when_longer_chunks_cost_layers(self) -> None:
        # E5 rig, Llama 3.3 70B with chunks of 1024: pp4096 52 tokens/s instead of 35.5, but two
        # more layers on the CPU and generation at 1.28 instead of 1.35.
        for profile in ("llama-70b-q4_k_m.json", "qwen3-32b-q4_k_m.json"):
            plan, model, devs = self.plan(profile)
            self.assertEqual(plan.ubatch, 512)
            self.assertNotIn("-ub", llama_command(plan, model, devs).args)
        plan, model, devs = self.plan("llama-70b-q4_k_m.json", chunk=1024)
        self.assertEqual(arg(llama_command(plan, model, devs), "-ub"), "1024")

    def test_a_size_can_be_asked_for(self) -> None:
        plan, model, devs = self.plan("gpt-oss-120b.json", chunk=2048)
        self.assertEqual(plan.ubatch, 2048)
        self.assertEqual(arg(llama_command(plan, model, devs), "-ub"), "2048")
        # The longer chunks take VRAM: more experts stay in RAM than with 512.
        self.assertGreater(plan.expert_ram_bytes, self.plan("gpt-oss-120b.json", chunk=512)[0].expert_ram_bytes)
        for s in plan.stages:
            self.assertLessEqual(s.used_bytes, s.capacity_bytes)

    def test_nothing_in_ram_keeps_512(self) -> None:
        for chunk in ("auto", 2048):
            plan, model, devs = self.plan("gpt-oss-20b.json", chunk=chunk)
            self.assertEqual(plan.ubatch, 512)
            self.assertNotIn("-ub", llama_command(plan, model, devs).args)
            self.assertEqual(any("stay at 512" in n for n in plan.notes), chunk == 2048)

    def test_a_chunk_is_at_most_half_the_context(self) -> None:
        self.assertEqual(self.plan("gpt-oss-120b.json", context=2048, chunk=1024)[0].ubatch, 1024)
        self.assertEqual(self.plan("gpt-oss-120b.json", context=2048, chunk=2048)[0].ubatch, 512)
        self.assertEqual(self.plan("llama-70b-q4_k_m.json", context=1024, chunk=1024)[0].ubatch, 512)

    def test_a_card_on_another_pc_keeps_512(self) -> None:
        # Not measured with a card on another PC.
        devs, _ = self.rig()
        devs = [devs[0], dataclasses.replace(devs[1], rpc="10.0.0.2:50052", rpc_overhead=2e-3)]
        plan, model, devs = self.plan("gpt-oss-120b.json", devices=devs, chunk=1024)
        self.assertTrue(any(s.n_layers and s.device == devs[1].name for s in plan.stages))
        self.assertEqual(plan.ubatch, 512)

    def test_chunks_of_4096_can_be_asked_for(self) -> None:
        # E5 rig under llama-server, gpt-oss-120b (engine/prototype/RESULTS-server-stock.md): an 8000-token
        # prompt in 52.5 s instead of 166 s with chunks of 512, generation at 10.4 tokens/s instead of 11.0.
        plan, model, devs = self.plan("gpt-oss-120b.json", context=12288, chunk=4096)
        self.assertEqual(plan.ubatch, 4096)
        cmd = llama_command(plan, model, devs)
        self.assertEqual((arg(cmd, "-b"), arg(cmd, "-ub")), ("4096", "4096"))
        self.assertEqual(cmd.env["GGML_OP_OFFLOAD_MIN_BATCH"], "600")
        self.assertNotIn("--no-op-offload", cmd.args)
        # Half the context at most.
        plan, _, _ = self.plan("gpt-oss-120b.json", context=4096, chunk=4096)
        self.assertEqual(plan.ubatch, 512)

    def test_the_server_does_not_set_the_logits_aside(self) -> None:
        # llama-server asks for the last token's logits only: its compute buffer was the same on the card
        # with the head as on the other (1182 and 1208 MiB with chunks of 4096). llama-bench's prompt test
        # asks for every token's, 0.77 MiB each for this vocabulary.
        def ram(all_logits: bool) -> int:
            devs, host = self.rig()
            host.all_logits = all_logits
            model = ModelProfile.from_json_file(os.path.join(EXAMPLES, "gpt-oss-120b.json"))
            return make_plan(model, devs, host, context=12288, chunk=4096).cpu_bytes
        self.assertLess(ram(False), ram(True))  # more of the model fits on the cards


if __name__ == "__main__":
    unittest.main()
