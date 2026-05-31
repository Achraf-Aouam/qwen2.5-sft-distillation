"""JSON parsing, repair, and normalization utilities for LLM outputs."""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class JsonParseResult:
    parsed: Optional[Dict[str, Any]]
    candidate_text: str
    repaired_text: Optional[str]
    used_repair: bool
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def strip_thinking_tags(response_content: str) -> str:
    content = response_content or ""
    if "</think>" in content:
        return content.split("</think>", 1)[1].lstrip()
    return content


def extract_json_candidate(response_content: str) -> str:
    content = strip_thinking_tags(response_content).strip()
    if not content:
        return ""

    markdown_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        content,
        re.DOTALL,
    )
    if markdown_match:
        return markdown_match.group(1).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return content[start : end + 1].strip()

    return content


def parse_json_response_with_details(response_content: str) -> JsonParseResult:
    candidate = extract_json_candidate(response_content)
    if not candidate:
        return JsonParseResult(
            parsed=None,
            candidate_text="",
            repaired_text=None,
            used_repair=False,
            error="No JSON candidate found in model output.",
        )

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return JsonParseResult(parsed, candidate, None, False, None)
        return JsonParseResult(
            parsed=None,
            candidate_text=candidate,
            repaired_text=None,
            used_repair=False,
            error=f"Expected a JSON object but found {type(parsed).__name__}.",
        )
    except json.JSONDecodeError as exc:
        repaired = repair_json_string(candidate)
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return JsonParseResult(
                    parsed=parsed,
                    candidate_text=candidate,
                    repaired_text=repaired,
                    used_repair=repaired != candidate,
                    error=None,
                )
            return JsonParseResult(
                parsed=None,
                candidate_text=candidate,
                repaired_text=repaired,
                used_repair=repaired != candidate,
                error=f"Expected a JSON object but found {type(parsed).__name__}.",
            )
        except Exception as repair_exc:  # pragma: no cover - defensive path
            return JsonParseResult(
                parsed=None,
                candidate_text=candidate,
                repaired_text=repaired,
                used_repair=repaired != candidate,
                error=f"{exc}; repair failed with {repair_exc}",
            )


def parse_json_response(response_content: str) -> Optional[Dict[str, Any]]:
    return parse_json_response_with_details(response_content).parsed


def repair_json_string(json_str: str) -> str:
    json_str = json_str.strip()
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    quotes = re.findall(r'(?<!\\)"', json_str)
    if len(quotes) % 2 != 0:
        json_str += '"'

    open_braces = json_str.count("{")
    close_braces = json_str.count("}")
    open_brackets = json_str.count("[")
    close_brackets = json_str.count("]")

    if open_brackets < close_brackets and open_braces > close_braces and json_str.endswith("]"):
        json_str = json_str[:-1] + "}"
        close_brackets -= 1
        close_braces += 1

    if open_brackets > close_brackets:
        json_str += "]" * (open_brackets - close_brackets)

    if open_braces > close_braces:
        json_str += "}" * (open_braces - close_braces)

    return json_str


def remove_accents(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ASCII", "ignore").decode("utf-8")


def normalize_key(key: str) -> str:
    key = remove_accents(str(key).lower())
    return re.sub(r"[^a-z0-9]", "", key)


def normalize_json_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    return {normalize_key(k): v for k, v in data.items()}


def normalize_string_for_match(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = remove_accents(text)
    text = re.sub(r"\s+", " ", text)
    return text


def string_similarity(left: Any, right: Any) -> float:
    return difflib.SequenceMatcher(
        None,
        normalize_string_for_match(left),
        normalize_string_for_match(right),
    ).ratio()


def find_matching_key(
    target_key: str,
    candidate_keys: List[str],
    cutoff: float = 0.8,
) -> Optional[str]:
    normalized_target = normalize_key(target_key)
    for candidate in candidate_keys:
        if normalize_key(candidate) == normalized_target:
            return candidate

    normalized_candidates = [normalize_key(candidate) for candidate in candidate_keys]
    matches = difflib.get_close_matches(
        normalized_target,
        normalized_candidates,
        n=1,
        cutoff=cutoff,
    )
    if not matches:
        return None

    matched_normalized = matches[0]
    for candidate in candidate_keys:
        if normalize_key(candidate) == matched_normalized:
            return candidate
    return None


def get_value_by_fuzzy_key(
    data: Dict[str, Any],
    target_key: str,
    cutoff: float = 0.8,
) -> Any:
    matching_key = find_matching_key(target_key, list(data.keys()), cutoff)
    if matching_key is None:
        return None
    return data[matching_key]


def extract_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    cleaned = re.sub(r"[€$£¥%]", "", cleaned).strip()
    cleaned = cleaned.replace(" ", "").replace("\u00a0", "")

    has_comma = "," in cleaned
    has_dot = "." in cleaned
    if has_comma and has_dot:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif has_comma:
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        match = re.search(r"-?\d+\.?\d*", cleaned)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None


def numbers_match(gt_value: Any, pred_value: Any, tolerance: float = 0.001) -> bool:
    gt_num = extract_number(gt_value)
    pred_num = extract_number(pred_value)
    if gt_num is None or pred_num is None:
        return False
    if gt_num == pred_num:
        return True
    if gt_num != 0:
        return abs(gt_num - pred_num) / abs(gt_num) < tolerance
    return abs(pred_num) < tolerance


def parse_french_date(value: str) -> Optional[Tuple[int, int, int]]:
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    patterns = [
        r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})",
        r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        if year < 100:
            year = 2000 + year if year < 50 else 1900 + year
        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
            return (day, month, year)
    return None


def dates_match(gt_value: Any, pred_value: Any) -> bool:
    gt_date = parse_french_date(str(gt_value)) if gt_value is not None else None
    pred_date = parse_french_date(str(pred_value)) if pred_value is not None else None
    return gt_date is not None and pred_date is not None and gt_date == pred_date


def looks_like_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.match(r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$", value.strip()))


def looks_like_number(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False
    text = re.sub(r"[€$£¥%\s]", "", value.strip())
    return bool(re.match(r"^-?[\d.,]+$", text))
