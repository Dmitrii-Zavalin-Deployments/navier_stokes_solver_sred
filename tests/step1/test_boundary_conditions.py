# tests/step1/test_boundary_conditions.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_grid():
    """Provides canonical grid metadata (Section 5 Compliance)."""
    return solver_input_schema_dummy()["grid"]

@pytest.fixture
def valid_6_face_bc():
    """Provides a complete, mathematically sound 6-face BC list."""
    return [
        {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}},
        {"location": "x_max", "type": "pressure", "values": {"p": 0.0}},
        {"location": "y_min", "type": "no-slip", "values": {}},
        {"location": "y_max", "type": "no-slip", "values": {}},
        {"location": "z_min", "type": "no-slip", "values": {}},
        {"location": "z_max", "type": "no-slip", "values": {}}
    ]

# --- CANONICAL & MATH TESTS ---

def test_full_bc_normalization_and_storage(dummy_grid, valid_6_face_bc):
    """Verifies 6-face closure, float conversion, and key flattening."""
    bc_table = parse_boundary_conditions(valid_6_face_bc, dummy_grid)
    
    # Cuboid must have 6 faces
    assert len(bc_table) == 6
    # Value normalization (int -> float)
    assert isinstance(bc_table["x_min"]["u"], float)
    assert bc_table["x_min"]["u"] == 1.0
    assert bc_table["x_max"]["p"] == 0.0

def test_invalid_location_override(dummy_grid):
    """Verifies error for non-canonical location names (Center of Universe)."""
    bc_list = [{"location": "center_of_universe", "type": "no-slip", "values": {}}]
    with pytest.raises(ValueError, match="(?i)Invalid or missing boundary location"):
        parse_boundary_conditions(bc_list, dummy_grid)

# --- PHYSICAL EXCLUSION & VALIDATION RULES ---

def test_inflow_requires_numeric_uvw(dummy_grid):
    """Theory Check: Inflow requires a full 3D velocity vector (Phase F Mandate)."""
    incomplete = [{"location": "x_min", "type": "inflow", "values": {"u": 5.0}}]
    with pytest.raises(ValueError, match="requires numeric velocity"):
        parse_boundary_conditions(incomplete, dummy_grid)

def test_pressure_requires_numeric_p(dummy_grid):
    """Theory Check: Pressure type must define 'p'."""
    missing_p = [{"location": "x_max", "type": "pressure", "values": {"u": 0.0}}]
    with pytest.raises(ValueError, match="requires numeric 'p'"):
        parse_boundary_conditions(missing_p, dummy_grid)

def test_pressure_exclusion_rule(dummy_grid):
    """Theory Check: Outflow/Slip cannot define pressure (Over-specification)."""
    bad_outflow = [{"location": "x_max", "type": "outflow", "values": {"p": 10.0}}]
    with pytest.raises(ValueError, match="cannot define pressure"):
        parse_boundary_conditions(bad_outflow, dummy_grid)

# --- DEBT & GUARDRAIL TRIGGERS ---

def test_duplicate_location_logic(dummy_grid):
    """Prevention of physics collisions: one face cannot have two types."""
    duplicate_loc = [
        {"location": "y_min", "type": "no-slip"},
        {"location": "y_min", "type": "free-slip"}
    ]
    with pytest.raises(ValueError, match="(?i)Duplicate BC"):
        parse_boundary_conditions(duplicate_loc, dummy_grid)

def test_line_32_invalid_type_trigger(dummy_grid):
    """Trigger: Invalid BC type (Warp-Zone)."""
    bad_type = [{"location": "x_min", "type": "warp-zone"}]
    with pytest.raises(ValueError, match="Invalid boundary type"):
        parse_boundary_conditions(bad_type, dummy_grid)

def test_line_64_incomplete_domain_trigger(dummy_grid):
    """Trigger: Incomplete Domain (less than 6 faces provided)."""
    incomplete = [{"location": "x_min", "type": "no-slip"}]
    with pytest.raises(ValueError, match="Incomplete Domain"):
        parse_boundary_conditions(incomplete, dummy_grid)