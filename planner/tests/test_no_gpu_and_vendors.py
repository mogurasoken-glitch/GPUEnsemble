"""A main PC without a GPU, and AMD / Intel cards through llama.cpp's Vulkan build."""

import os
import subprocess
import tempfile
import unittest
from unittest import mock

from gpuensemble_planner import cli, devices
from gpuensemble_planner.devices import (GIB, Adapter, Device, Host, detect_gpus, is_integrated, parse_list_devices,
                                         parse_video_controllers)
from gpuensemble_planner.gpu_table import find
from gpuensemble_planner.llamabin import Install, choose_backend
from gpuensemble_planner.llamacpp import llama_command
from gpuensemble_planner.model import LayerProfile, ModelProfile
from gpuensemble_planner.plan import PlanError, make_plan, streams_cpu_weights

VULKAN_LIST = """ggml_vulkan: Found 3 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (AMD proprietary driver) | uma: 0 | fp16: 1
Available devices:
  Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 23012 MiB free)
  Vulkan1: Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11800 MiB free)
  Vulkan2: NVIDIA GeForce RTX 2080 Ti (11264 MiB, 10200 MiB free)
  Vulkan3: AMD Radeon RX 6800 XT (RADV NAVI21) (16368 MiB, 16000 MiB free)
  Vulkan4: AMD Radeon(TM) Graphics (2048 MiB, 1900 MiB free)
  Vulkan5: Some Future GPU (8192 MiB, 8000 MiB free)
"""


def dense_model(n_layers: int = 40, layer_gib: float = 0.4, head_gib: float = 0.5) -> ModelProfile:
    return ModelProfile(name="dense", architecture="llama",
                        layers=[LayerProfile(int(layer_gib * GIB), 0, 4096) for _ in range(n_layers)],
                        head_bytes=int(head_gib * GIB), embed_bytes=int(0.4 * GIB))


HOST = Host(ram_bytes=int(200 * GIB), cpu_bandwidth=60e9, expert_bandwidth=40e9, threads=32)


def far_card(name: str, gib: float, endpoint: str) -> Device:
    return Device(name, int(gib * GIB), 756e9, reserve_bytes=2 * GIB, llama_device="CUDA0", rpc=endpoint,
                  rpc_overhead=0.008, rpc_rtt=0.0075)


class TableTest(unittest.TestCase):
    def test_amd_and_intel_names_as_their_drivers_give_them(self) -> None:
        for name, vram, key, gbs in (
            ("AMD Radeon RX 7900 XTX", 24, "radeon rx 7900 xtx", 960),
            ("AMD Radeon RX 7900 XT", 20, "radeon rx 7900 xt", 800),
            ("AMD Radeon RX 6800 XT (RADV NAVI21)", 16, "radeon rx 6800 xt", 512),
            ("AMD Radeon RX 9060 XT", 8, "radeon rx 9060 xt", 320),
            ("AMD Radeon RX 7600M XT", 8, "radeon rx 7600m xt", 288),
            ("AMD Radeon VII", 16, "radeon vii", 1024),
            ("AMD Radeon Pro VII", 16, "radeon pro vii", 1024),
            ("AMD Radeon PRO W7900", 48, "radeon pro w7900", 864),
            ("AMD Instinct MI100", 32, "mi100", 1229),
            ("AMD Radeon Instinct MI50", 32, "mi50", 1024),
            ("Intel(R) Arc(TM) A770 Graphics", 16, "arc a770", 560),
            ("Intel(R) Arc(TM) A770 Graphics", 8, "arc a770", 512),
            ("Intel(R) Arc(TM) B580 Graphics", 12, "arc b580", 456),
            ("Intel(R) Arc(TM) Pro B60 Graphics", 24, "arc pro b60", 456),
            ("Intel(R) Arc(TM) A770M Graphics", 16, "arc a770m", 512),
        ):
            spec = find(name, vram)
            self.assertIsNotNone(spec, name)
            self.assertEqual((spec.key, spec.gbs), (key, gbs), name)

    def test_built_in_graphics_are_not_in_the_table(self) -> None:
        for name in ("AMD Radeon(TM) Graphics", "Intel(R) UHD Graphics 630", "Intel(R) Arc(TM) Graphics",
                     "AMD Radeon 780M", "Intel(R) Iris(R) Xe Graphics"):
            self.assertIsNone(find(name), name)
            self.assertTrue(is_integrated(name), name)

    def test_cards_are_not_taken_for_built_in_graphics(self) -> None:
        for name in ("AMD Radeon RX 7900 XTX", "Intel(R) Arc(TM) A770 Graphics", "Intel(R) Arc(TM) Pro B60 Graphics",
                     "NVIDIA GeForce RTX 2080 Ti", "AMD Radeon VII", "AMD Radeon RX 7600M XT"):
            self.assertFalse(is_integrated(name), name)


class VulkanListTest(unittest.TestCase):
    def test_cards_of_every_maker_with_vulkan_names(self) -> None:
        found, notes = parse_list_devices(VULKAN_LIST)
        self.assertEqual([d.llama_device for d in found], ["Vulkan0", "Vulkan1", "Vulkan2", "Vulkan3", "Vulkan5"])
        amd = found[0]
        self.assertEqual(amd.name, "Vulkan0 AMD Radeon RX 7900 XTX")
        self.assertAlmostEqual(amd.bandwidth, 960e9 * 0.75)
        # Free memory minus the margin, as for a card nvidia-smi lists.
        self.assertEqual(amd.vram_bytes - amd.reserve_bytes, 23012 * 1024**2 - devices.DETECTED_FREE_MARGIN)
        self.assertEqual(found[2].bandwidth, 463e9)  # the 2080 Ti's measured figure, through Vulkan too
        self.assertEqual(found[3].name, "Vulkan3 AMD Radeon RX 6800 XT (RADV NAVI21)")
        self.assertTrue(found[4].name.endswith(devices.GUESSED))
        self.assertEqual(len(notes), 1)
        self.assertIn("Vulkan4", notes[0])

    def test_cpu_build_lists_nothing(self) -> None:
        self.assertEqual(parse_list_devices("Available devices:\n"), ([], []))


class AdapterTest(unittest.TestCase):
    def test_video_controllers(self) -> None:
        text = """[{"Name": "NVIDIA GeForce RTX 4090", "PNPDeviceID": "PCI\\\\VEN_10DE&DEV_2684&SUBSYS_1"},
                   {"Name": "AMD Radeon RX 7900 XTX", "PNPDeviceID": "PCI\\\\VEN_1002&DEV_744C&SUBSYS_2"},
                   {"Name": "Intel(R) UHD Graphics 770", "PNPDeviceID": "PCI\\\\VEN_8086&DEV_4680&SUBSYS_3"},
                   {"Name": "ASPEED Graphics Family(WDDM)", "PNPDeviceID": "PCI\\\\VEN_1A03&DEV_2000&SUBSYS_4"},
                   {"Name": "Microsoft Basic Display Adapter", "PNPDeviceID": "ROOT\\\\BASICDISPLAY\\\\0000"}]"""
        found = parse_video_controllers(text)
        self.assertEqual([a.vendor for a in found], ["nvidia", "amd", "intel", "other", "other"])
        self.assertEqual([a.name for a in found if a.pools], ["AMD Radeon RX 7900 XTX"])

    def test_one_adapter_and_bad_output(self) -> None:
        one = parse_video_controllers('{"Name": "Intel(R) Arc(TM) B580 Graphics", "PNPDeviceID": "PCI\\\\VEN_8086&DEV_E20B"}')
        self.assertEqual(one, [Adapter("Intel(R) Arc(TM) B580 Graphics", "intel")])
        self.assertTrue(one[0].pools)
        self.assertEqual(parse_video_controllers(""), [])
        self.assertEqual(parse_video_controllers("not json"), [])

    def test_linux_amd_card_has_its_own_memory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for card, vendor, vram in (("card0", "0x1002", 24 * GIB), ("card1", "0x1002", 512 << 20),
                                       ("card2", "0x8086", None)):
                os.makedirs(os.path.join(root, card, "device"))
                with open(os.path.join(root, card, "device", "vendor"), "w") as f:
                    f.write(vendor + "\n")
                if vram is not None:
                    with open(os.path.join(root, card, "device", "mem_info_vram_total"), "w") as f:
                        f.write(str(vram))
            os.makedirs(os.path.join(root, "card0-DP-1"))
            found = devices._linux_adapters(root)
        self.assertEqual(found, [Adapter("AMD GPU (card0)", "amd")])

    def test_adapters_are_read_once(self) -> None:
        calls = []

        def run(argv, **kw):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, '{"Name": "AMD Radeon RX 7600", "PNPDeviceID": "PCI\\\\VEN_1002"}', "")

        with mock.patch.object(devices, "_adapters_cache", None):
            first = devices.adapters(run, platform="win32")
            again = devices.adapters(run, platform="win32")
        self.assertEqual(first, again)
        self.assertEqual(len(calls), 1)

    def test_japanese_windows(self) -> None:
        # PowerShell is told to write UTF-8 and Python to read it so: with the console's cp932 the
        # Japanese name of the basic display adapter broke the read and hid every card.
        seen = {}

        def run(argv, **kw):
            seen.update(kw)
            seen["command"] = argv[-1]
            text = ('[{"Name": "Microsoft 基本ディスプレイ アダプター", "PNPDeviceID": "ROOT\\\\BASICDISPLAY"},'
                    ' {"Name": "AMD Radeon RX 7900 XTX", "PNPDeviceID": "PCI\\\\VEN_1002&DEV_744C"}]')
            return subprocess.CompletedProcess(argv, 0, text, "")

        with mock.patch.object(devices, "_adapters_cache", None):
            found = devices.adapters(run, platform="win32")
        self.assertEqual(seen["encoding"], "utf-8")
        self.assertIn("UTF8", seen["command"])
        self.assertEqual([a.name for a in found if a.pools], ["AMD Radeon RX 7900 XTX"])


class DetectTest(unittest.TestCase):
    def test_vulkan_build_lists_the_cards(self) -> None:
        inst = Install("b12375", "vulkan", "/x")
        run = lambda argv, **kw: subprocess.CompletedProcess(argv, 0, VULKAN_LIST, "")  # noqa: E731
        found, notes = detect_gpus(inst, run=run, nvidia=lambda: self.fail("nvidia-smi is not asked"))
        self.assertEqual(len(found), 5)
        self.assertEqual(len(notes), 1)

    def test_nvidia_and_a_card_that_needs_vulkan(self) -> None:
        nv = [Device("GPU0 NVIDIA GeForce RTX 4090", 24 * GIB, 820e9, llama_device="CUDA0")]
        found, notes = detect_gpus(Install("b12375", "cuda", "/x"), nvidia=lambda: nv,
                                   others=lambda: [Adapter("AMD Radeon RX 7900 XTX", "amd")])
        self.assertEqual(found, nv)
        self.assertIn("AMD Radeon RX 7900 XTX", notes[0])
        self.assertIn("Vulkan", notes[0])

    def test_no_gpu_is_not_an_error(self) -> None:
        def missing():
            raise RuntimeError("nvidia-smi was not found")

        found, notes = detect_gpus(None, nvidia=missing, others=lambda: [])
        self.assertEqual(found, [])
        self.assertIn("no GPU found on this PC", notes[0])

    def test_backend_for_amd_and_intel(self) -> None:
        self.assertEqual(choose_backend("win32", None, ["AMD Radeon RX 7900 XTX"])[0], "vulkan")
        backend, why = choose_backend("win32", (560, 94), ["Intel(R) Arc(TM) B580 Graphics"])
        self.assertEqual(backend, "vulkan")
        self.assertIn("NVIDIA", why)
        self.assertEqual(choose_backend("win32", None, [])[0], "cpu")
        self.assertEqual(choose_backend("win32", (560, 94), [])[0], "cuda")

    def test_up_installs_vulkan_for_an_amd_card(self) -> None:
        calls = []

        def fake_setup(**kw):
            calls.append(kw)
            return Install("b12375", "vulkan", "/x")

        with mock.patch.object(cli, "installed", return_value=Install("b12375", "cuda", "/x")), \
                mock.patch.object(cli, "other_cards", return_value=[Adapter("AMD Radeon RX 7900 XTX", "amd")]), \
                mock.patch.object(cli, "setup", side_effect=fake_setup), \
                mock.patch.object(cli, "detect_gpus", return_value=([], [])) as detect:
            cli.local_gpus(install=True)
            self.assertEqual(calls[0]["backend"], "vulkan")
            self.assertEqual(detect.call_args[0][0].backend, "vulkan")
            calls.clear()
            cli.local_gpus(install=False)  # plan and recommend only say so
            self.assertEqual(calls, [])


class NoGpuPlanTest(unittest.TestCase):
    def test_cpu_only_plan(self) -> None:
        model = dense_model()
        plan = make_plan(model, [], HOST, context=4096)
        self.assertEqual(plan.stages, [])
        self.assertEqual(plan.cpu_layers, model.n_layers)
        self.assertGreater(plan.tokens_per_second, 0)
        # Every weight and the head are read from RAM once per token.
        self.assertGreater(plan.seconds_per_token, (16 * GIB + 0.5 * GIB) / HOST.cpu_bandwidth)
        self.assertIn("no GPU: the whole model runs on this PC's CPU", plan.notes)
        cmd = llama_command(plan, model, [], threads=32)
        self.assertEqual(cmd.args[:4], ["-ngl", "0", "-dev", "none"])
        self.assertIn("--no-op-offload", cmd.args)
        self.assertNotIn("--rpc", cmd.args)

    def test_cpu_only_needs_the_ram(self) -> None:
        small = Host(ram_bytes=int(8 * GIB), cpu_bandwidth=20e9, expert_bandwidth=10e9)
        with self.assertRaises(PlanError):
            make_plan(dense_model(), [], small, context=4096)

    def test_main_pc_without_a_gpu_uses_the_other_pcs(self) -> None:
        model = dense_model(100, 0.5)  # 50 GiB: more than the far cards hold
        far = [far_card("RTX 4090 on far", 24, "100.64.0.2:50052"), far_card("RTX 3090 on far2", 24, "100.64.0.3:50052")]
        plan = make_plan(model, far, HOST, context=4096)
        used = [s for s in plan.stages if s.n_layers]
        self.assertEqual(len(used), 2)
        self.assertGreater(plan.cpu_layers, 0)  # the rest stays in this PC's RAM
        cmd = llama_command(plan, model, far, threads=32)
        self.assertIn("--rpc", cmd.args)
        self.assertEqual(cmd.args[cmd.args.index("-dev") + 1].split(","), ["RPC0", "RPC1"])
        # The CPU's weights would cross the network for each prompt batch: op offload stays off.
        self.assertFalse(streams_cpu_weights(plan, model, far))
        self.assertIn("--no-op-offload", cmd.args)

    def test_a_local_first_card_still_streams(self) -> None:
        model = dense_model(80, 0.5)
        local = Device("GPU0 RTX 2080 Ti", 11 * GIB, 463e9, llama_device="CUDA0", pcie_bandwidth=3.2e9)
        plan = make_plan(model, [local], HOST, context=4096)
        self.assertTrue(plan.cpu_layers)
        self.assertTrue(streams_cpu_weights(plan, model, [local]))


if __name__ == "__main__":
    unittest.main()
