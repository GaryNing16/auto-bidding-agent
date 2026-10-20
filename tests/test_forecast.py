import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from autobid.data import write_requests
from autobid.forecast import (SupplyForecast, evaluate_forecast, fit_forecast,
                              split_by_episode, write_split_manifest)
from autobid.types import AuctionConfig, Participation, Request


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.config = AuctionConfig(budgets={"A": 10, "B": 10},
                                    player_types={"A": "type0", "B": "type0"}, steps=2)
        self.train = self.csv("train", [Request(0, "train-r0", (Participation("A", "type0", 10),))])
        self.validation = self.csv("validation", [Request(0, "valid-r0", (
            Participation("A", "type0", 6), Participation("B", "type0", 2)))])
        self.model = self.folder / "forecast.json"

    def tearDown(self):
        self.temp.cleanup()

    def csv(self, name, requests):
        path = self.folder / (name + ".csv")
        write_requests(requests, path)
        return path

    def test_zero_supply_advertisers_and_empty_slots_count_in_mean(self):
        fit_forecast([self.train], self.config, self.model)
        model = SupplyForecast.load(self.model)
        self.assertEqual(model.predict(0, "type0"), {
            "opportunity_count": 0.5, "potential_value": 5.0, "used_fallback": False})
        self.assertEqual(model.predict(1, "type0")["potential_value"], 0.0)
        self.assertEqual(model.predict(0, "unknown"), {
            "opportunity_count": 0.5, "potential_value": 5.0, "used_fallback": True})

    def test_holdout_mae_and_unchanged_fitted_model(self):
        fit_forecast([self.train], self.config, self.model)
        model_before = self.model.read_bytes()
        report = evaluate_forecast(self.model, [self.validation], self.config)
        self.assertEqual(report["opportunity_count_mae"], 0.25)
        self.assertEqual(report["potential_value_mae"], 1.0)
        self.assertEqual(report["player_time_samples"], 4)
        self.assertEqual(model_before, self.model.read_bytes())

    def test_reject_train_eval_overlap_even_renamed(self):
        fit_forecast([self.train], self.config, self.model)
        renamed = self.folder / "different-name.csv"
        renamed.write_bytes(self.train.read_bytes())
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_forecast(self.model, [renamed], self.config)

    def test_source_period_overlap_and_stale_manifest(self):
        for path in (self.train, self.validation):
            path.with_suffix(".manifest.json").write_text(json.dumps({
                "upstream_repository": "https://github.com/alimama-tech/AuctionNet",
                "source_delivery_period": "7", "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        fit_forecast([self.train], self.config, self.model)
        with self.assertRaisesRegex(ValueError, "source_group"):
            evaluate_forecast(self.model, [self.validation], self.config)
        self.validation.with_suffix(".manifest.json").write_text('{"output_sha256":"stale"}')
        with self.assertRaisesRegex(ValueError, "stale"):
            split_by_episode([self.train, self.validation])

    def test_split_whole_source_groups_without_hash_overlap(self):
        paths = [self.train, self.validation]
        paths.append(self.csv("test", [Request(1, "test-r0", (Participation("A", "type0", 8),))]))
        result = split_by_episode(paths, seed=7)
        repeated = split_by_episode(reversed(paths), seed=7)
        self.assertEqual(result, repeated)
        self.assertTrue(all(len(value) == 1 for value in result["splits"].values()))
        hashes = [entry["sha256"] for value in result["splits"].values() for entry in value]
        self.assertEqual(len(hashes), len(set(hashes)))
        destination = self.folder / "split.json"
        write_split_manifest(result, destination)
        self.assertEqual(json.loads(destination.read_text()), result)

    def test_invalid_split_and_type_fail(self):
        with self.assertRaisesRegex(ValueError, "not enough"):
            split_by_episode([self.train, self.validation])
        with self.assertRaisesRegex(ValueError, "positive fraction"):
            split_by_episode([self.train], validation_fraction=0.5, test_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_forecast([self.train, self.train], self.config, self.model)
        wrong_type = AuctionConfig(budgets={"A": 10, "B": 10},
                                   player_types={"A": "wrong", "B": "type0"}, steps=2)
        with self.assertRaisesRegex(ValueError, "player_type mismatch"):
            fit_forecast([self.train], wrong_type, self.model)


if __name__ == "__main__":
    unittest.main()
