# file: src/step1/state_to_dict.py
from __future__ import annotations
import numpy as np

def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    return obj

def state_to_dict(state):
    """Convert a Step‑1 state (with NumPy arrays) into JSON‑serializable lists."""
    return _to_json_safe(state)
