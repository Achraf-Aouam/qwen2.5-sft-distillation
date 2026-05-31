import unittest

from accuracy_eval import calculate_dataset_accuracy, evaluate_single_example
from json_utils import normalize_key, parse_json_response_with_details, string_similarity


class AccuracyEvalTests(unittest.TestCase):
    def test_json_repair_recovers_truncated_object(self):
        result = parse_json_response_with_details('{"sender":"ACME","total":12.5')
        self.assertIsNotNone(result.parsed)
        self.assertEqual(result.parsed["sender"], "ACME")
        self.assertTrue(result.used_repair)

    def test_key_normalization_handles_accents_and_punctuation(self):
        self.assertEqual(normalize_key("modéle vehicule"), "modelevehicule")
        self.assertEqual(normalize_key("modèle_vehicule"), "modelevehicule")

    def test_missing_empty_ground_truth_is_accepted(self):
        accuracy, details = evaluate_single_example(
            instruction="sys",
            input_text="doc",
            gt_output='{"numero_de_chassis": null}',
            llm_output="{}",
        )
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(details["field_matches"][0]["match_type"], "missing_ok")

    def test_numeric_and_date_normalization_match(self):
        accuracy, details = evaluate_single_example(
            instruction="sys",
            input_text="doc",
            gt_output='{"date_expedition": "03/01/2025", "total_ht": 5944.05}',
            llm_output='{"date_expedition":"3-1-25","total_ht":"5 944,05"}',
        )
        self.assertEqual(accuracy, 1.0)
        match_types = {item["match_type"] for item in details["field_matches"]}
        self.assertIn("date", match_types)
        self.assertIn("numeric", match_types)

    def test_high_threshold_fuzzy_accepts_close_enough_strings(self):
        accuracy, details = evaluate_single_example(
            instruction="sys",
            input_text="doc",
            gt_output='{"sender": "ORDER SUPPLIES SARL"}',
            llm_output='{"sender": "ORDER SUPPLIES SARL"}',
        )
        self.assertEqual(accuracy, 1.0)
        self.assertIn(details["field_matches"][0]["match_type"], {"normalized", "fuzzy"})
        self.assertGreaterEqual(string_similarity("ORDER SUPPLIES SARL", "ORDER SUPPLIES SARL"), 0.96)

    def test_obvious_string_mismatch_is_rejected(self):
        accuracy, details = evaluate_single_example(
            instruction="sys",
            input_text="doc",
            gt_output='{"sender": "ORDER SUPPLIES SARL"}',
            llm_output='{"sender": "Acme Fashion"}',
        )
        self.assertEqual(accuracy, 0.0)
        self.assertEqual(details["field_matches"][0]["match_type"], "mismatch")

    def test_weighted_dataset_accuracy_uses_field_counts(self):
        results = [
            (1.0, {"correct": 1, "total": 1}),
            (0.0, {"correct": 0, "total": 3}),
        ]
        self.assertEqual(calculate_dataset_accuracy(results), 25.0)


if __name__ == "__main__":
    unittest.main()
