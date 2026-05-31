import csv
import json
import tempfile
import unittest
from pathlib import Path

from inline_eval import LocalArtifactLogger


class LocalArtifactLoggerTests(unittest.TestCase):
    def test_logger_writes_summary_csv_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LocalArtifactLogger(Path(tmpdir))
            payload = {
                "loss/order": 0.2,
                "loss/overall_weighted": 0.3,
                "gt_acc/order": 90.0,
                "gt_acc/overall_weighted": 85.0,
            }
            samples = [
                {
                    "step": 20,
                    "corpus": "order",
                    "sample_id": 0,
                    "raw_output": '{"sender":"ORDER SUPPLIES SARL"}',
                }
            ]

            logger.write_step(20, payload, samples)

            summary = json.loads((Path(tmpdir) / "step_20_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["payload"]["gt_acc/order"], 90.0)

            lines = (Path(tmpdir) / "step_20_samples.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["corpus"], "order")

            with (Path(tmpdir) / "history.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], "20")
            self.assertEqual(rows[0]["gt_acc/order"], "90.0")
            self.assertEqual(logger.logged_steps(), {20})


if __name__ == "__main__":
    unittest.main()
