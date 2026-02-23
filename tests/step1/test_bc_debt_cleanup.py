# tests/step1/test_bc_debt_cleanup.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions

def test_line_32_invalid_bc_type():
    """Trigger: Boundary type is not in _VALID_TYPES."""
    # We provide a valid face but an unsupported physics type
    bad_bc = [{
        "location": "x_min",
        "type": "magic-wall", # Invalid type
        "values": {"u": 0}
    }]
    
    with pytest.raises(ValueError, match="Invalid boundary type: magic-wall"):
        parse_boundary_conditions(bad_bc)

def test_line_64_incomplete_domain():
    """Trigger: Missing one or more of the 6 required faces."""
    # We only provide 5 out of 6 faces
    incomplete_bc = [
        {"location": "x_min", "type": "no-slip", "values": {}},
        {"location": "x_max", "type": "no-slip", "values": {}},
        {"location": "y_min", "type": "no-slip", "values": {}},
        {"location": "y_max", "type": "no-slip", "values": {}},
        {"location": "z_min", "type": "no-slip", "values": {}}
        # z_max is missing!
    ]
    
    with pytest.raises(ValueError, match="Incomplete Domain: Missing boundary conditions for faces"):
        parse_boundary_conditions(incomplete_bc)