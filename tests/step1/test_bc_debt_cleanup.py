import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions

def test_line_32_invalid_type():
    """Trigger: Invalid BC type."""
    bad_bc = [{"location": "x_min", "type": "warp-zone", "values": {}}]
    grid_config = {"nx": 2, "ny": 2, "nz": 2}
    
    with pytest.raises(ValueError, match="Invalid boundary type"):
        parse_boundary_conditions(bad_bc, grid_config)

def test_line_64_incomplete_domain():
    """Trigger: Incomplete Domain (missing faces)."""
    # Only 1 face provided
    incomplete = [{"location": "x_min", "type": "no-slip", "values": {}}]
    grid_config = {"nx": 2, "ny": 2, "nz": 2}
    
    with pytest.raises(ValueError, match="Incomplete Domain"):
        parse_boundary_conditions(incomplete, grid_config)