# tests/step1/test_boundary_logic_compliance.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_grid():
    """Provides the canonical grid metadata from the dummy."""
    return solver_input_schema_dummy()["grid"]

class TestBoundaryLogicCompliance:
    """
    Ensures 100% compliance with theory_of_operations_boundary_conditions.html.
    Verified for Phase F: Data Intake and Physical Exclusions.
    """

    def test_inflow_action_compliance(self, dummy_grid):
        """Theory Check: Inflow requires numerical u, v, w"""
        # Note: This fails logic validation BEFORE the 6-face audit triggers
        incomplete_inflow = [{"location": "x_min", "type": "inflow", "values": {"u": 5.0}}]
        with pytest.raises(ValueError, match="requires numeric velocity"):
            parse_boundary_conditions(incomplete_inflow, dummy_grid)

    def test_pressure_action_compliance(self, dummy_grid):
        """Theory Check: Pressure requires numerical p"""
        missing_p = [{"location": "x_max", "type": "pressure", "values": {"u": 0.0}}]
        with pytest.raises(ValueError, match="requires numeric 'p'"):
            parse_boundary_conditions(missing_p, dummy_grid)

    def test_uniqueness_and_location_integrity(self, dummy_grid):
        """Theory Check: Detect duplicates (Prevention of physics collisions)"""
        duplicate_loc = [
            {"location": "y_min", "type": "no-slip"},
            {"location": "y_min", "type": "free-slip"}
        ]
        with pytest.raises(ValueError, match="Duplicate BC"):
            parse_boundary_conditions(duplicate_loc, dummy_grid)

    def test_pressure_exclusion_rule(self, dummy_grid):
        """Theory Check: Outflow/Slip cannot define pressure (Over-specification check)"""
        bad_outflow = [{"location": "x_max", "type": "outflow", "values": {"p": 10.0}}]
        with pytest.raises(ValueError, match="cannot define pressure"):
            parse_boundary_conditions(bad_outflow, dummy_grid)

    def test_full_schema_valid_config(self, dummy_grid):
        """Verify valid config parsing and float conversion with all 6 faces."""
        # Must provide all 6 faces to pass the Final Audit (Zero-Debt Mandate)
        valid_bcs = [
            {"location": "x_min", "type": "inflow", "values": {"u": 1, "v": 0, "w": 0}},
            {"location": "x_max", "type": "pressure", "values": {"p": 0.0}},
            {"location": "y_min", "type": "no-slip", "values": {}},
            {"location": "y_max", "type": "no-slip", "values": {}},
            {"location": "z_min", "type": "no-slip", "values": {}},
            {"location": "z_max", "type": "no-slip", "values": {}}
        ]
        parsed = parse_boundary_conditions(valid_bcs, dummy_grid)
        
        # In a 3D domain, we expect exactly 6 entries
        assert len(parsed) == 6
        assert isinstance(parsed["x_min"]["u"], float)
        assert parsed["x_max"]["p"] == 0.0