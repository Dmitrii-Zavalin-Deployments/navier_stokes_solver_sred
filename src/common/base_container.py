# src/common/base_container.py
from typing import Any
import numpy as np

class ValidatedContainer:
    """The 'Security Guard' logic. Zero dependencies on physics or state."""
    def _get_safe(self, name: str) -> Any:
        attr_name = f"_{name}"
        if not hasattr(self, attr_name):
            raise AttributeError(f"Coding Error: '{attr_name}' not defined in {self.__class__.__name__}.")
        val = getattr(self, attr_name)
        if val is None:
            raise RuntimeError(f"Access Error: '{name}' in {self.__class__.__name__} is uninitialized.")
        return val

    def _set_safe(self, name: str, value: Any, expected_type: type):
        if value is not None and not isinstance(value, expected_type):
            raise TypeError(f"Validation Error: '{name}' must be {expected_type}, got {type(value)}.")
        setattr(self, f"_{name}", value)

    def to_dict(self) -> dict:
        out = {}
        for attr, val in self.__dict__.items():
            clean_key = attr.lstrip('_')
            if isinstance(val, ValidatedContainer):
                out[clean_key] = val.to_dict()
            elif isinstance(val, np.ndarray):
                out[clean_key] = val.tolist()
            elif isinstance(val, dict):
                out[clean_key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val.items()}
            else:
                out[clean_key] = val
        return out