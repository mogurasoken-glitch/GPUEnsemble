"""THIRD_PARTY_NOTICES.md must name the page of every model in the catalog."""

import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CATALOG = os.path.join(ROOT, "planner", "src", "gpuensemble_planner", "catalog.json")
NOTICES = os.path.join(ROOT, "THIRD_PARTY_NOTICES.md")


class NoticesTest(unittest.TestCase):
    def test_every_catalog_model_is_listed(self) -> None:
        if not os.path.exists(NOTICES):  # an installed copy of the planner has no repository around it
            self.skipTest("THIRD_PARTY_NOTICES.md is not next to this checkout")
        with open(CATALOG, encoding="utf-8") as f:
            repos = {m["hf_repo"] for m in json.load(f)["models"]}
        with open(NOTICES, encoding="utf-8") as f:
            text = f.read()
        missing = sorted(r for r in repos if f"https://huggingface.co/{r})" not in text)
        self.assertEqual(missing, [], "add these to THIRD_PARTY_NOTICES.md")


if __name__ == "__main__":
    unittest.main()
