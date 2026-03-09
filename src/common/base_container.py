# src/common/base_container.py

import json
from collections.abc import Iterator
from typing import Any

import jsonschema
import numpy as np


class ValidatedContainer:
    """The 'Security Guard' logic. Now with memory-efficient slots and O(1) attribute validation."""
    __slots__ = []  # Empty slots for the base; children will populate theirs
    
    # Cache for allowed attribute names to avoid re-calculating MRO at runtime
    _ALLOWED_ATTRS = None

    def __iter__(self) -> Iterator[str]:
        """Helper to iterate over attributes defined in slots across the hierarchy."""
        for cls in reversed(self.__class__.__mro__):
            for slot in getattr(cls, '__slots__', []):
                yield slot
    
    def _get_safe(self, name: str) -> Any:
        attr_name = f"_{name}"
        # Use getattr with a sentinel to check if the slot exists
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

    def validate_against_schema(self, schema_path: str):
        """Final Firewall: Automatically flattens structures for validation."""
        with open(schema_path) as f:
            schema = json.load(f)
        
        instance_data = self.to_dict()
        
        if "config" in instance_data:
            config = instance_data.pop("config")
            instance_data.update(config)
            
        if "masks" in instance_data and isinstance(instance_data["masks"], dict):
            if "mask" in instance_data["masks"]:
                instance_data["mask"] = instance_data["masks"]["mask"]

        jsonschema.validate(instance=instance_data, schema=schema)

    def __setattr__(self, name: str, value: Any):
        """Prevents dynamic creation of attributes via O(1) cached lookup."""
        # Lazily initialize cache for this specific subclass
        if self._ALLOWED_ATTRS is None:
            allowed = set()
            for cls in self.__class__.__mro__:
                allowed.update(getattr(cls, '__slots__', []))
            self.__class__._ALLOWED_ATTRS = frozenset(allowed)
        
        # Enforce strict attribute access (excluding the underscore prefix for validation)
        # Note: If adding internal attributes like _x, they must be in __slots__
        if name not in self._ALLOWED_ATTRS:
            raise AttributeError(f"Memory Leak Prevention: Attribute '{name}' not in __slots__ for {self.__class__.__name__}")
        
        super().__setattr__(name, value)
    
    def to_dict(self) -> dict:
        """Serializes the container using the slots hierarchy."""
        out = {}
        for attr in self:
            val = getattr(self, attr, None)
            clean_key = attr.lstrip('_')
            
            # 1. Handle SciPy Sparse Matrices
            if hasattr(val, "toarray"):
                out[clean_key] = val.toarray().tolist()
            
            # 2. Handle Nested Containers
            elif isinstance(val, ValidatedContainer):
                out[clean_key] = val.to_dict()
                
            # 3. Handle NumPy Arrays
            elif isinstance(val, np.ndarray):
                out[clean_key] = val.tolist()
                
            # 4. Handle Dictionaries
            elif isinstance(val, dict):
                out[clean_key] = {
                    k: (v.toarray().tolist() if hasattr(v, "toarray") 
                        else (v.tolist() if isinstance(v, np.ndarray) else v)) 
                    for k, v in val.items()
                }
            else:
                out[clean_key] = val
        return out