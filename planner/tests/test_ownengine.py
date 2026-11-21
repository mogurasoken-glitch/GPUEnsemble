import contextlib
import dataclasses
import io
import os
import tempfile
import unittest
from unittest import mock

from gpuensemble_planner import cli, monitor
from gpuensemble_planner.devices import GIB, Device, load_rig
from gpuensemble_planner.llamabin import Install
from gpuensemble_planner.model import LayerProfile, ModelProfile
from gpuensemble_planner.ownengine import _ranges, engine_command, engine_placement
from gpuensemble_planner.plan import PlanError

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
RIG = os.path.join(EXAMPLES, "rig-e5-turing.json")
GPT_OSS = ModelProfile.from_json_file(os.path.join(EXAMPLES, "gpt-oss-120b.json"))
QWEN = ModelProfile.from_json_file(os.path.join(EXAMPLES, "qwen3-32b-q4_k_m.json"))


def arg(cmd, flag):
    return cmd.args[cmd.args.index(flag) + 1]


class EngineCommandTest(unittest.TestCase):
    def test_the_development_pc_gets_the_placement_measured_there(self) -> None:
        devs, host = load_rig(RIG)
        cmd = engine_command(GPT_OSS, devs, 12288, threads=16)
        # the 8 GB card first, the 11 GB card last (it takes the output)
        self.assertEqual(arg(cmd, "--devices"), "CUDA1,CUDA0")
        self.assertEqual(arg(cmd, "--layers"), "18,19")
        self.assertEqual(arg(cmd, "--host-experts"), "0-30")
        self.assertEqual(cmd.cpu_expert_layers, list(range(31)))
        self.assertEqual(arg(cmd, "-c"), "12288")
        self.assertEqual(arg(cmd, "-t"), "16")
        for flag in ("--alternate", "--ram", "--flash"):
            self.assertIn(flag, cmd.args)
        # the small kernel on the last card has a backend of its own (it works at the start of an answer)
        self.assertEqual(cmd.env["GPUE_AWAKE"], "0,512,1,0,1")
        self.assertEqual(cmd.env["GGML_OP_OFFLOAD_MIN_BATCH"], arg(cmd, "--min-batch"))
        self.assertEqual(cmd.env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        # generated tokens gather their attention on the 11 GB card
        self.assertEqual(arg(cmd, "--gen-device"), "1")

    def test_generation_keeps_the_placement_when_the_last_card_has_no_room_for_the_copies(self) -> None:
        devs, host = load_rig(RIG)
        cmd = engine_command(GPT_OSS, devs, 16384, threads=16)
        # the experts stay where they were with the smaller context: none moves to RAM to make room
        self.assertEqual(arg(cmd, "--host-experts"), "0-30")
        self.assertNotIn("--gen-device", cmd.args)
        self.assertEqual(cmd.env["GPUE_AWAKE"], "0,512,1,0,1")

    def test_two_8_gb_cards_fit_but_need_more_ram_than_a_64_gb_pc_gives(self) -> None:
        devs, host = load_rig(RIG)
        small = next(d for d in devs if "SUPER" in d.name)
        two = [dataclasses.replace(small, llama_device="CUDA0"), dataclasses.replace(small, llama_device="CUDA1")]
        p = engine_placement(GPT_OSS, two, 12288)
        # the rule of bytes still finds a placement: experts of three layers on the cards, 33 in RAM
        self.assertEqual(len(p.host_expert_layers), 33)
        self.assertEqual(p.gather, 1)
        for share in p.cards:
            self.assertLessEqual(share.used_bytes, share.capacity_bytes)
        self.assertGreater(p.host_expert_bytes, host.ram_bytes)
        # with the PC's RAM given, the plan is refused instead of starting a server that would swap
        with self.assertRaisesRegex(PlanError, "GiB of experts in RAM"):
            engine_placement(GPT_OSS, two, 12288, ram_bytes=host.ram_bytes)
        # the development PC as it is passes the same check
        engine_placement(GPT_OSS, devs, 12288, ram_bytes=host.ram_bytes)

    def test_one_large_card_keeps_more_experts(self) -> None:
        cmd = engine_command(GPT_OSS, [Device("big", 24 * GIB, 800e9, llama_device="CUDA0")], 8192)
        self.assertEqual(arg(cmd, "--devices"), "CUDA0")
        self.assertEqual(arg(cmd, "--layers"), "37")
        self.assertNotIn("--gen-device", cmd.args)
        self.assertEqual(cmd.env["GPUE_AWAKE"], "0,512,0")
        self.assertLess(len(cmd.cpu_expert_layers), 31)
        # the experts kept on the card are those of the last layers
        self.assertEqual(cmd.cpu_expert_layers, list(range(len(cmd.cpu_expert_layers))))

    def test_everything_on_the_cards_needs_no_host_experts(self) -> None:
        small = ModelProfile("s", "gpt-oss", [LayerProfile(GIB // 64, GIB // 4, 2048)] * 4, GIB // 4, GIB // 8,
                             n_expert=8, n_expert_used=2)
        cmd = engine_command(small, [Device("a", 8 * GIB, 300e9, llama_device="CUDA0")], 4096)
        self.assertNotIn("--host-experts", cmd.args)
        self.assertEqual(cmd.cpu_expert_layers, [])

    def test_other_models_and_other_cards_are_refused(self) -> None:
        devs, _ = load_rig(RIG)
        with self.assertRaisesRegex(PlanError, "gpt-oss models only"):
            engine_command(QWEN, devs, 4096)
        with self.assertRaisesRegex(PlanError, "NVIDIA card in this PC"):
            engine_command(GPT_OSS, [Device("v", 16 * GIB, 300e9, llama_device="Vulkan0")], 4096)
        with self.assertRaisesRegex(PlanError, "NVIDIA card in this PC"):
            engine_command(GPT_OSS, [Device("r", 24 * GIB, 800e9, llama_device="CUDA0", rpc="10.0.0.2:50052")], 4096)
        with self.assertRaisesRegex(PlanError, "too small"):
            engine_command(GPT_OSS, [Device("tiny", 3 * GIB, 100e9, llama_device="CUDA0")], 4096)

    def test_threads_stay_at_what_was_measured(self) -> None:
        devs, _ = load_rig(RIG)
        self.assertEqual(arg(engine_command(GPT_OSS, devs, 4096, threads=32), "-t"), "16")
        self.assertEqual(arg(engine_command(GPT_OSS, devs, 4096, threads=8), "-t"), "8")
        self.assertNotIn("-t", engine_command(GPT_OSS, devs, 4096).args)

    def test_ranges(self) -> None:
        self.assertEqual(_ranges([0, 1, 2, 5, 7, 8]), "0-2,5,7-8")
        self.assertEqual(_ranges([]), "")


class PlacementTest(unittest.TestCase):
    def test_what_each_card_of_the_development_pc_holds(self) -> None:
        devs, _ = load_rig(RIG)
        p = engine_placement(GPT_OSS, devs, 12288)
        small, large = p.cards
        self.assertEqual((small.llama_device, large.llama_device), ("CUDA1", "CUDA0"))
        self.assertEqual(small.layers, list(range(0, 36, 2)))
        self.assertEqual(large.layers, list(range(1, 36, 2)))
        self.assertEqual((small.expert_layers, large.expert_layers), ([32, 34], [31, 33, 35]))
        self.assertEqual(p.host_expert_layers, list(range(31)))
        self.assertEqual(p.host_expert_bytes, sum(GPT_OSS.layers[il].expert_bytes for il in range(31)))
        # the copies for generated tokens are on the card that computes their attention, and only there
        self.assertEqual(p.gather, 1)
        self.assertEqual(small.copy_bytes, 0)
        self.assertGreater(large.copy_bytes, 0)
        for share in p.cards:
            self.assertLessEqual(share.used_bytes, share.capacity_bytes)
            self.assertEqual(share.used_bytes, share.weight_bytes + share.kv_bytes + share.work_bytes + share.copy_bytes)
        # the output is counted with the last card
        self.assertEqual(large.weight_bytes, GPT_OSS.head_bytes + sum(
            GPT_OSS.layers[il].dense_bytes + (GPT_OSS.layers[il].expert_bytes if il in (31, 33, 35) else 0)
            for il in large.layers))

    def test_no_copies_without_the_room(self) -> None:
        devs, _ = load_rig(RIG)
        p = engine_placement(GPT_OSS, devs, 16384)
        self.assertIsNone(p.gather)
        self.assertEqual([share.copy_bytes for share in p.cards], [0, 0])

    def test_the_command_follows_the_placement(self) -> None:
        devs, _ = load_rig(RIG)
        p = engine_placement(GPT_OSS, devs, 12288)
        self.assertEqual(engine_command(GPT_OSS, devs, 12288, threads=16, placement=p).args,
                         engine_command(GPT_OSS, devs, 12288, threads=16).args)

    def test_the_table(self) -> None:
        devs, _ = load_rig(RIG)
        text = cli.format_placement(engine_placement(GPT_OSS, devs, 12288))
        self.assertIn("engine:  GPUEnsemble's own (experimental)   context: 12288 tokens   prompt chunk: 4096 tokens", text)
        rows = {line.split("  ")[0]: line for line in text.splitlines()}
        self.assertRegex(rows["RTX 2080 SUPER 8GB"], r"0, 2 \.\. 34 +2 of 18 .* 0\.0 +6\.\d/7\.0$")
        self.assertRegex(rows["RTX 2080 Ti 11GB"], r"1, 3 \.\. 35 +3 of 18 .* 0\.8 +9\.\d/9\.\d$")
        self.assertIn("experts in RAM: those of 31 layers (49.0 GiB)", text)
        self.assertIn("no speed estimate", text)
        self.assertIn("note: generated tokens compute their attention on RTX 2080 Ti 11GB", text)

    def test_the_table_for_one_card(self) -> None:
        text = cli.format_placement(engine_placement(GPT_OSS, [Device("big", 24 * GIB, 800e9, llama_device="CUDA0")], 8192))
        self.assertRegex(text, r"(?m)^big +0-35 +11 of 36 ")
        self.assertNotIn("compute their attention", text)


class UpWithOwnEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = tmp.name
        env = {k: v for k, v in os.environ.items() if k != "GPUENSEMBLE_ENGINE_EXE"}
        env["GPUENSEMBLE_HOME"] = self.home
        patches = [
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(cli, "pull", side_effect=lambda entry, base=None: os.path.join(self.home, entry["id"] + ".gguf")),
            mock.patch.object(cli, "installed", return_value=None),
            mock.patch.object(cli, "setup", return_value=Install("b1", "cuda", "/opt/llama")),
            mock.patch.object(cli, "_load_model", return_value=GPT_OSS),
            mock.patch.object(cli, "run", return_value=0),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def up(self, *args: str) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(["up", "--rig", RIG, "--yes", *args])
        return code, err.getvalue()

    def test_starts_gpue_server_with_its_own_placement(self) -> None:
        code, err = self.up("--model", "gpt-oss-120b", "--ctx", "12288", "--engine", "own", "--engine-exe", "/opt/gpue/gpue-server")
        self.assertEqual(code, 0, err)
        exe, command, path, extra = cli.run.call_args.args
        self.assertEqual(exe, "/opt/gpue/gpue-server")
        self.assertIn("--alternate", command.args)
        self.assertEqual(arg(command, "--host-experts"), "0-30")
        # nothing of llama-server's own options
        self.assertEqual(extra, ["--host", "127.0.0.1", "--port", "8080", "-a", "gpt-oss-120b"])
        self.assertIn("experimental", err)
        self.assertIn("starting gpue-server", err)
        cli.setup.assert_not_called()  # llama.cpp is not installed for it
        # the table is the engine's own placement, not llama.cpp's plan
        self.assertIn("engine:  GPUEnsemble's own", err)
        self.assertRegex(err, r"(?m)^RTX 2080 Ti 11GB +1, 3 \.\. 35 +3 of 18 ")
        self.assertNotIn("estimated generation speed", err)
        self.assertNotIn("experts on GPU", err)

    def test_monitor_gets_the_engines_placement(self) -> None:
        with mock.patch.object(monitor, "write_run", wraps=monitor.write_run) as write:
            code, err = self.up("--model", "gpt-oss-120b", "--ctx", "12288", "--engine", "own",
                                "--engine-exe", "/opt/gpue/gpue-server")
        self.assertEqual(code, 0, err)
        info = write.call_args.args[0]
        self.assertEqual(info["engine"], "own")
        self.assertIsNone(info["seconds_per_token"])
        self.assertEqual([(s["gpu"], s["index"], s["first"], s["last"], s["count"], s["ram_experts"], s["generates"])
                          for s in info["stages"]],
                         [("RTX 2080 SUPER 8GB", 1, 0, 34, 18, 16, False), ("RTX 2080 Ti 11GB", 0, 1, 35, 18, 15, True)])

    def test_the_path_can_come_from_the_environment(self) -> None:
        with mock.patch.dict(os.environ, {"GPUENSEMBLE_ENGINE_EXE": "/env/gpue-server"}):
            code, err = self.up("--model", "gpt-oss-120b", "--engine", "own")
        self.assertEqual(code, 0, err)
        self.assertEqual(cli.run.call_args.args[0], "/env/gpue-server")

    def test_without_a_path_it_says_what_is_missing(self) -> None:
        code, err = self.up("--model", "gpt-oss-120b", "--engine", "own")
        self.assertEqual(code, 1)
        self.assertIn("--engine-exe", err)
        cli.run.assert_not_called()

    def test_a_model_the_engine_cannot_run(self) -> None:
        with mock.patch.object(cli, "_load_model", return_value=QWEN):
            code, err = self.up("--model", "qwen3-32b", "--engine", "own", "--engine-exe", "/opt/gpue/gpue-server")
        self.assertEqual(code, 1)
        self.assertIn("gpt-oss models only", err)
        cli.run.assert_not_called()

    def test_llama_cpp_stays_the_default(self) -> None:
        code, err = self.up("--model", "gpt-oss-120b")
        self.assertEqual(code, 0, err)
        self.assertEqual(cli.run.call_args.args[0], os.path.join("/opt/llama", "llama-server"))
        self.assertIn("--metrics", cli.run.call_args.args[3])


if __name__ == "__main__":
    unittest.main()
