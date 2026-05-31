"""Accuracy evaluation logic for comparing model predictions with ground truth."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from json_utils import (
    dates_match,
    find_matching_key,
    looks_like_date,
    looks_like_number,
    normalize_string_for_match,
    numbers_match,
    parse_json_response_with_details,
    string_similarity,
)

FUZZY_VALUE_MATCH_THRESHOLD = 0.96
FUZZY_KEY_MATCH_THRESHOLD = 0.8


def is_value_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped in {"null", "None"}:
            return True
    return False


def compare_values(gt_value: Any, pred_value: Any) -> Tuple[bool, str, float | None]:
    if is_value_empty(gt_value) and is_value_empty(pred_value):
        return True, "normalized", None

    if is_value_empty(gt_value) != is_value_empty(pred_value):
        return False, "mismatch", None

    if looks_like_date(gt_value) or looks_like_date(pred_value):
        if dates_match(gt_value, pred_value):
            return True, "date", None

    if looks_like_number(gt_value) or looks_like_number(pred_value):
        if numbers_match(gt_value, pred_value):
            return True, "numeric", None

    if isinstance(gt_value, str) and isinstance(pred_value, str):
        if gt_value.strip() == pred_value.strip():
            return True, "exact", None
        normalized_gt = normalize_string_for_match(gt_value)
        normalized_pred = normalize_string_for_match(pred_value)
        if normalized_gt == normalized_pred:
            return True, "normalized", None
        if normalized_gt and normalized_pred:
            similarity = string_similarity(normalized_gt, normalized_pred)
            if similarity >= FUZZY_VALUE_MATCH_THRESHOLD:
                return True, "fuzzy", similarity
        return False, "mismatch", None

    if isinstance(gt_value, (int, float)) and isinstance(pred_value, (int, float)):
        if gt_value == pred_value:
            return True, "exact", None
        return False, "mismatch", None

    if isinstance(gt_value, dict) and isinstance(pred_value, dict):
        accuracy, _, _, _ = compare_json_objects(gt_value, pred_value)
        return accuracy == 1.0, "exact" if accuracy == 1.0 else "mismatch", None

    if isinstance(gt_value, list) and isinstance(pred_value, list):
        if len(gt_value) != len(pred_value):
            return False, "mismatch", None
        for gt_item, pred_item in zip(gt_value, pred_value):
            matched, _, _ = compare_values(gt_item, pred_item)
            if not matched:
                return False, "mismatch", None
        return True, "exact", None

    if gt_value == pred_value:
        return True, "exact", None

    return False, "mismatch", None


def compare_json_objects(
    gt_json: Dict[str, Any],
    pred_json: Dict[str, Any],
) -> Tuple[float, int, int, List[Dict[str, Any]]]:
    if not gt_json:
        return 1.0, 0, 0, []

    correct = 0
    total = len(gt_json)
    field_matches: List[Dict[str, Any]] = []

    for gt_key, gt_value in gt_json.items():
        pred_key = find_matching_key(
            gt_key,
            list(pred_json.keys()),
            cutoff=FUZZY_KEY_MATCH_THRESHOLD,
        )

        if pred_key is None:
            if is_value_empty(gt_value):
                correct += 1
                field_matches.append(
                    {
                        "gt_key": gt_key,
                        "pred_key": None,
                        "gt_value": gt_value,
                        "pred_value": None,
                        "matched": True,
                        "match_type": "missing_ok",
                        "value_similarity": None,
                    }
                )
            else:
                field_matches.append(
                    {
                        "gt_key": gt_key,
                        "pred_key": None,
                        "gt_value": gt_value,
                        "pred_value": None,
                        "matched": False,
                        "match_type": "mismatch",
                        "value_similarity": None,
                    }
                )
            continue

        pred_value = pred_json[pred_key]
        matched, match_type, similarity = compare_values(gt_value, pred_value)
        if matched:
            correct += 1

        field_matches.append(
            {
                "gt_key": gt_key,
                "pred_key": pred_key,
                "gt_value": gt_value,
                "pred_value": pred_value,
                "matched": matched,
                "match_type": match_type,
                "value_similarity": similarity,
            }
        )

    accuracy = correct / total if total > 0 else 1.0
    return accuracy, correct, total, field_matches


def evaluate_single_example(
    instruction: str,
    input_text: str,
    gt_output: str,
    llm_output: str,
) -> Tuple[float, Dict[str, Any]]:
    try:
        gt_json = json.loads(gt_output) if isinstance(gt_output, str) else gt_output
    except json.JSONDecodeError:
        return 0.0, {
            "error": "Failed to parse ground truth JSON",
            "correct": 0,
            "total": 0,
            "valid_json": False,
            "field_matches": [],
        }

    if not isinstance(gt_json, dict):
        return 0.0, {
            "error": "Ground truth output must be a JSON object",
            "correct": 0,
            "total": 0,
            "valid_json": False,
            "field_matches": [],
        }

    parse_result = parse_json_response_with_details(llm_output)
    if parse_result.parsed is None:
        total = len(gt_json)
        return 0.0, {
            "error": "Failed to parse LLM output JSON",
            "correct": 0,
            "total": total,
            "gt_keys": list(gt_json.keys()),
            "pred_keys": [],
            "valid_json": False,
            "field_matches": [
                {
                    "gt_key": gt_key,
                    "pred_key": None,
                    "gt_value": gt_value,
                    "pred_value": None,
                    "matched": False,
                    "match_type": "mismatch",
                    "value_similarity": None,
                }
                for gt_key, gt_value in gt_json.items()
            ],
            "raw_output": llm_output,
            "parsed_output": None,
            "parse_result": parse_result.to_dict(),
            "instruction": instruction,
            "input_text": input_text,
        }

    accuracy, correct, total, field_matches = compare_json_objects(gt_json, parse_result.parsed)
    return accuracy, {
        "correct": correct,
        "total": total,
        "gt_keys": list(gt_json.keys()),
        "pred_keys": list(parse_result.parsed.keys()),
        "valid_json": True,
        "field_matches": field_matches,
        "raw_output": llm_output,
        "parsed_output": parse_result.parsed,
        "parse_result": parse_result.to_dict(),
        "instruction": instruction,
        "input_text": input_text,
    }


def load_eval_dataset(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Eval file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def calculate_dataset_accuracy(results: List[Tuple[float, Dict[str, Any]]]) -> float:
    if not results:
        return 0.0

    total_correct = sum(details.get("correct", 0) for _, details in results)
    total_keys = sum(details.get("total", 0) for _, details in results)
    if total_keys == 0:
        return sum(score for score, _ in results) / len(results) * 100
    return (total_correct / total_keys) * 100
