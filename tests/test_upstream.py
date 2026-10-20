"""Audit source provenance and the executable upstream data bridge."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UpstreamProvenanceTests(unittest.TestCase):
    def test_vendored_files_match_exact_git_blob_hashes(self):
        root = ROOT / "third_party/auctionnet"
        manifest = json.loads((root / "PROVENANCE.json").read_text())
        for item in manifest["files"]:
            with self.subTest(path=item["path"]):
                content = (root / item["path"]).read_bytes()
                digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                self.assertEqual(digest, item["git_blob_sha"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "upstream generator requires NumPy")
    def test_pinned_generator_converts_complete_sparse_requests(self):
        from autobid.data import load_requests
        spec = importlib.util.spec_from_file_location("upstream_bridge", ROOT / "scripts/generate_auctionnet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.csv"
            second = Path(tmp) / "second.csv"
            manifest = module.generate(first, seed=7, requests=1000, players=6, categories=3)
            module.generate(second, seed=7, requests=1000, players=6, categories=3)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            requests = load_requests(first, steps=72)
            self.assertEqual(len(requests), manifest["generator"]["actual_opportunities"])
            self.assertTrue(all(0 <= r.time_step < 72 for r in requests))
            self.assertTrue(all(1 <= len(r.participants) <= 6 for r in requests))
            self.assertTrue(any(len(r.participants) < 6 for r in requests))
            self.assertEqual(manifest["generator"]["commit"], "c20de631a40e24669d1d6a8e2acefd8f492f172d")


if __name__ == "__main__":
    unittest.main()
