# tests/step1/test_allocate_fields_debt.py

import pytest
from src.step1.allocate_fields import allocate_fields

def test_line_24_invalid_dimensions():
    """Trigger: Zero or negative grid dimensions."""
    # Test with a zero dimension
    with pytest.raises(ValueError, match="Invalid grid dimensions"):
        allocate_fields({"nx": 0, "ny": 10, "nz": 10})
        
    # Test with a negative dimension
    with pytest.raises(ValueError, match="Invalid grid dimensions"):
        allocate_fields({"nx": 10, "ny": -5, "nz": 10})