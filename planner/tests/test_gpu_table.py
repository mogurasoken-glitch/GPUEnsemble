import os
import unittest

from gpuensemble_planner import gpu_table
from gpuensemble_planner.devices import effective_bandwidth, parse_nvidia_smi
from gpuensemble_planner.gpu_table import GPUS, find


class GpuTableTest(unittest.TestCase):
    def gbs(self, name: str, vram=None) -> float | None:
        spec = find(name, vram)
        return spec.gbs if spec else None

    def test_names_as_nvidia_smi_gives_them(self) -> None:
        for name, gbs in (
            ("NVIDIA GeForce RTX 4090", 1008), ("NVIDIA GeForce RTX 4090 D", 1008),
            ("NVIDIA GeForce RTX 4070 Ti SUPER", 672), ("NVIDIA GeForce RTX 4070 Ti", 504),
            ("NVIDIA GeForce RTX 5070 Ti", 896), ("NVIDIA GeForce RTX 5050", 320),
            ("NVIDIA GeForce GTX 1080 Ti", 484), ("NVIDIA GeForce GTX 1660 SUPER", 336),
            ("NVIDIA GeForce GTX 1650", 128), ("NVIDIA TITAN Xp", 548), ("NVIDIA TITAN X (Pascal)", 480),
            ("NVIDIA GeForce GTX TITAN X", 336), ("NVIDIA TITAN RTX", 672),
            ("Quadro RTX 6000", 672), ("NVIDIA RTX 6000 Ada Generation", 960),
            ("NVIDIA RTX 4000 SFF Ada Generation", 280), ("NVIDIA RTX A6000", 768), ("NVIDIA RTX A2000 12GB", 288),
            ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 1792),
            ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", 1792),
            ("Quadro P4000", 243), ("Tesla P40", 347), ("Tesla P4", 192), ("Tesla T4", 320),
            ("Tesla V100-PCIE-16GB", 900), ("Tesla V100S-PCIE-32GB", 1134),
            ("NVIDIA A10", 600), ("NVIDIA A10G", 600), ("NVIDIA A2", 200), ("NVIDIA L4", 300),
            ("NVIDIA L40S", 864), ("NVIDIA H100 PCIe", 2000), ("NVIDIA H100 80GB HBM3", 3350),
            ("NVIDIA GeForce RTX 3070 Laptop GPU", 448), ("NVIDIA GeForce RTX 4060 Laptop GPU", 256),
            ("NVIDIA GeForce RTX 2080 with Max-Q Design", 384),
        ):
            self.assertEqual(self.gbs(name), gbs, name)

    def test_names_with_sizes_and_generation(self) -> None:
        for name, vram, gbs in (
            ("NVIDIA GeForce RTX 3050 6GB Laptop GPU", 6, 168), ("NVIDIA GeForce RTX 3050 4GB Laptop GPU", 4, 192),
            ("NVIDIA RTX 3500 Ada Generation Laptop GPU", 12, 432), ("NVIDIA RTX 500 Ada Generation Laptop GPU", 4, 128),
            ("NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU", 24, 896), ("NVIDIA RTX A5000 Laptop GPU", 16, 448),
            ("NVIDIA GeForce RTX 5090 D v2", 24, 1344), ("NVIDIA GeForce RTX 5090 D", 32, 1792),
            ("NVIDIA RTX 5880 Ada Generation", 48, 960), ("NVIDIA RTX A1000", 8, 192), ("NVIDIA H800 PCIe", 80, 2000),
            ("NVIDIA H20", 96, 4000), ("NVIDIA L20", 48, 864), ("NVIDIA A800 80GB PCIe", 80, 1935),
            ("NVIDIA A100-SXM4-80GB", 80, 2039), ("Tesla V100-PCIE-16GB", 16, 900),
        ):
            self.assertEqual(self.gbs(name, vram), gbs, name)
        self.assertIsNone(find("NVIDIA GeForce MX450"))

    def test_whole_words_only(self) -> None:
        self.assertIsNone(find("NVIDIA A1000"))  # neither A100 nor A10
        self.assertIsNone(find("NVIDIA GeForce RTX 40900"))
        self.assertIsNone(find("Some Future GPU"))

    def test_memory_size_picks_the_variant(self) -> None:
        self.assertEqual(self.gbs("NVIDIA GeForce RTX 3060", 12), 360)
        self.assertEqual(self.gbs("NVIDIA GeForce RTX 3060", 8), 240)
        self.assertEqual(self.gbs("NVIDIA GeForce RTX 3060"), 360)
        self.assertEqual(self.gbs("NVIDIA GeForce RTX 3080", 12), 912)
        self.assertEqual(self.gbs("NVIDIA A100-SXM4-80GB", 80), 2039)
        self.assertEqual(self.gbs("NVIDIA A100 80GB PCIe", 79.2), 1935)
        self.assertEqual(self.gbs("NVIDIA A100-PCIE-40GB", 39.4), 1555)
        # nvidia-smi gives MiB; 8192 MiB is 8 GiB.
        (gpu,) = parse_nvidia_smi("0, NVIDIA GeForce RTX 3060, 8192, 7900, 4, 16\n")
        self.assertAlmostEqual(gpu.bandwidth, 240e9 * 0.75)

    def test_notebooks_never_get_desktop_figures(self) -> None:
        self.assertIsNone(find("NVIDIA GeForce GTX 1050 Ti with Max-Q Design"))
        self.assertIsNone(find("NVIDIA GeForce RTX 3050 A Laptop GPU"))
        self.assertIsNone(find("NVIDIA RTX 6000 Ada Generation Laptop GPU"))
        self.assertIsNone(find("NVIDIA RTX PRO 6000 Blackwell Generation Laptop GPU"))

    def test_measured_cards_use_the_measurement(self) -> None:
        self.assertEqual(effective_bandwidth("NVIDIA GeForce RTX 2080 Ti"), 463e9)
        self.assertEqual(effective_bandwidth("NVIDIA GeForce RTX 2080 SUPER"), 362e9)
        self.assertEqual(effective_bandwidth("NVIDIA GeForce RTX 4090"), 820e9)  # measured
        self.assertEqual(effective_bandwidth("NVIDIA GeForce RTX 4090 D"), 1008e9 * 0.75)
        measured = [g for g in GPUS if g.source == "measured"]
        self.assertTrue(all(g.measured_by for g in measured))

    def test_every_key_is_reachable(self) -> None:
        # A key that a longer key always swallows could never be found.
        for g in GPUS:
            name = g.key.replace("max-q", "Max-Q Design") if "max-q" in g.key else g.key
            self.assertEqual(find(name, g.vram_gb).key, g.key, g.key)
        # Same key, same size twice would make one entry unreachable.
        pairs = [(g.key, g.vram_gb) for g in GPUS]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_markdown(self) -> None:
        text = gpu_table.markdown_table()
        self.assertIn("| rtx 2080 ti | desktop | Turing | 11 GB | 616 | 463 | measured:", text)
        self.assertEqual(text.count("\n") - 1, len(GPUS))

    def test_gpus_md_is_up_to_date(self) -> None:
        path = os.path.join(os.path.dirname(__file__), "..", "GPUS.md")
        with open(path, encoding="utf-8") as f:
            self.assertIn(gpu_table.markdown_table(), f.read(),
                          "regenerate planner/GPUS.md with python -m gpuensemble_planner gpus")


if __name__ == "__main__":
    unittest.main()
