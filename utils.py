# utils.py
import re
import json
from typing import Dict, Any, Tuple, Optional

def normalize_text(s: str) -> str:
    """Basic text normalization for comparison."""
    if not s: return ""
    # Collapse whitespace, lowercase
    s = str(s).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def normalize_value(value: Any) -> str:
    """Aggressive normalization for Dates and Numbers."""
    if value is None: return ""
    s = normalize_text(str(value))
    
    # Remove currency symbols and common noise
    s = re.sub(r'[€$£¥\s]', '', s)
    
    # Date Normalization (dd-mm-yyyy -> yyyy-mm-dd)
    # Add your complex regex patterns here from previous notebook
    # ...
    
    return s

def parse_json_output(text: str) -> Dict:
    """Extract JSON from LLM output (handles messy prefixes/suffixes)."""
    try:
        # Find first { and last }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {}

def calculate_accuracy(pred: Dict, gt: Dict) -> Tuple[int, int]:
    """Compare two JSONs key by key."""
    correct = 0
    total = len(gt)
    if total == 0: return 0, 0
    
    for key, gt_val in gt.items():
        pred_val = pred.get(key)
        if normalize_value(pred_val) == normalize_value(gt_val):
            correct += 1
            
    return correct, total