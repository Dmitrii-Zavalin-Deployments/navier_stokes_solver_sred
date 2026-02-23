# tests/step1/test_bc_debt.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions

def test_line_32_invalid_type():
    """Trigger: Invalid BC type."""
    bad_bc = [{"location": "x_min", "type": "warp-zone", "values": {}}]
    with pytest.raises(ValueError, match="Invalid boundary type"):
        parse_boundary_conditions(bad_bc)

def test_line_64_missing_faces():
    """Trigger: Incomplete domain (missing faces)."""
    # Only providing one face instead of six
    incomplete = [{"location": "x_min", "type": "no-slip", "values": {}}]
    with pytest.raises(ValueError, match="Incomplete Domain"):
        parse_boundary_conditions(incomplete)