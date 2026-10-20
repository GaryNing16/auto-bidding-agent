import csv
import json
import tempfile
import unittest
from pathlib import Path

from autobid.adapters import REQUIRED, convert_auctionnet, sha256_file
from autobid.data import FIELDS, load_requests, summarize_requests, synthetic_requests, write_requests


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def csv(self, name, fields, rows):
        path = self.directory / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def test_sparse_membership_and_numeric_index_order(self):
        path = self.csv("requests.csv", FIELDS, [
            (10, 0, "r10", "A", "x", 5),
            (2, 0, "r2", "B", "y", 0),
            (11, 0, "r10", "C", "z", 7),
        ])
        requests = load_requests(path)
        self.assertEqual([r.request_id for r in requests], ["r2", "r10"])
        self.assertEqual([p.player_id for p in requests[0].participants], ["B"])
        self.assertEqual([p.pvalue for p in requests[1].participants], [5, 7])
        summary = summarize_requests(requests)
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["missing_pairs"], 3)
        self.assertEqual(summary["zero_value_rows"], 1)
        self.assertEqual(summary["requests_by_step"][1], 0)  # Empty slot stays empty.

    def test_reject_duplicate_pair_index_type_and_cross_slot(self):
        first = (0, 0, "r", "A", "x", 1)
        invalid_second_rows = [
            (1, 0, "r", "A", "x", 1),  # duplicate request/player
            (0, 0, "s", "B", "y", 1),  # duplicate index
            (1, 0, "s", "A", "y", 1),  # changing type
            (1, 1, "r", "B", "y", 1),  # cross-slot request
        ]
        for second in invalid_second_rows:
            with self.subTest(second=second), self.assertRaises(ValueError):
                load_requests(self.csv("bad.csv", FIELDS, [first, second]))

    def test_invalid_values_are_distinct_from_zero(self):
        for value in ("", "nan", "inf", "-1", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_requests(self.csv("bad.csv", FIELDS, [(0, 0, "r", "A", "x", value)]))
        for step in ("0.5", "-1", "72", "1.0"):
            with self.subTest(step=step), self.assertRaises(ValueError):
                load_requests(self.csv("bad.csv", FIELDS, [(0, step, "r", "A", "x", 0)]))

    def test_headers_empty_episode_and_roundtrip(self):
        self.assertEqual(load_requests(self.csv("empty.csv", FIELDS, [])), [])
        with self.assertRaises(ValueError):
            load_requests(self.csv("extra.csv", (*FIELDS, "secret"), []))
        first = synthetic_requests(seed=17, requests_per_step=2)
        self.assertEqual(first, synthetic_requests(seed=17, requests_per_step=2))
        self.assertNotEqual(first, synthetic_requests(seed=18, requests_per_step=2))
        path = self.directory / "roundtrip.csv"
        write_requests(first, path)
        self.assertEqual(first, load_requests(path))
        self.assertTrue(any(len(r.participants) < 3 for r in first))
        self.assertEqual(set(r.time_step for r in first), set(range(72)))

    def upstream_rows(self):
        # Four requests per old slot makes synthetic 48->72 reslotting observable.
        # Deliberately reverse numeric pv order; converter restores numeric order.
        rows = []
        for step in range(48):
            for j in (3, 2, 1, 0):
                pv = step * 4 + j
                rows.append((7, step, pv, "A", "x", 0.01 + pv / 10000))
                if pv % 2:
                    rows.append((7, step, pv, "B", "y", 0.02 + pv / 10000))
        return rows

    def test_upstream_conversion_groups_requests_and_fills_all_slots(self):
        source = self.csv("upstream.csv", REQUIRED, self.upstream_rows())
        target = self.directory / "converted.csv"
        manifest = convert_auctionnet(source, target, period=7, pvalue_scale=100)
        requests = load_requests(target)
        self.assertEqual(len(requests), 192)
        self.assertEqual(set(r.time_step for r in requests), set(range(72)))
        self.assertEqual(requests[0].request_id, "auctionnet:7:0")
        self.assertEqual(len(requests[0].participants), 1)
        self.assertEqual(len(requests[1].participants), 2)
        self.assertEqual(requests[0].participants[0].pvalue, 1)
        self.assertAlmostEqual(requests[1].participants[1].pvalue, 2.01)
        self.assertEqual(manifest["source_sha256"], sha256_file(source))
        self.assertEqual(json.loads(target.with_suffix(".manifest.json").read_text()), manifest)

    def test_upstream_subsample_preserves_full_request_and_time(self):
        source = self.csv("upstream.csv", REQUIRED, self.upstream_rows())
        all_path = self.directory / "all.csv"
        subset_path = self.directory / "subset.csv"
        convert_auctionnet(source, all_path, period="7")
        manifest = convert_auctionnet(source, subset_path, period="7", max_requests=10)
        all_requests = {r.request_id: r for r in load_requests(all_path)}
        subset = load_requests(subset_path)
        self.assertEqual(len(subset), 10)
        self.assertGreater(subset[-1].time_step, 60)
        for req in subset:
            self.assertEqual(req, all_requests[req.request_id])
        self.assertIn("complete-request", manifest["sampling"])

    def test_upstream_period_selection_and_time_base(self):
        source = self.csv("upstream.csv", REQUIRED, [
            (7, 48, 1, "A", "x", 0.01),
            (8, 1, 1, "B", "y", 0.02),
        ])
        target = self.directory / "out.csv"
        convert_auctionnet(source, target, period="7", time_base=1)
        self.assertEqual(len(load_requests(target)), 1)
        self.assertEqual(load_requests(target)[0].time_step, 71)
        with self.assertRaises(ValueError):
            convert_auctionnet(source, target, period="7", time_base=0)
        with self.assertRaises(ValueError):
            convert_auctionnet(source, target, period="9")

    def test_upstream_duplicates_types_cross_slots_and_overflow_rejected(self):
        first = (7, 0, 1, "A", "x", 0.01)
        for second in ((7, 0, 1, "A", "x", 0.02),
                       (7, 1, 1, "B", "y", 0.02),
                       (7, 0, 2, "A", "y", 0.02)):
            with self.subTest(second=second), self.assertRaises(ValueError):
                convert_auctionnet(self.csv("bad.csv", REQUIRED, [first, second]),
                                   self.directory / "out.csv", period=7)
        with self.assertRaises(ValueError):
            convert_auctionnet(self.csv("bad.csv", REQUIRED, [(7, 0, 1, "A", "x", "1e308")]),
                               self.directory / "out.csv", period=7, pvalue_scale=100)


if __name__ == "__main__":
    unittest.main()
