# tests/step1/test_boundary_logic_compliance.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions

class TestBoundaryLogicCompliance:
    """
    Ensures 100% compliance with theory_of_operations_boundary_conditions.html.
    This suite verifies that the 'Actions' defined in Step 1 documentation 
    are strictly enforced during the initialization phase.
    """

    def test_inflow_action_compliance(self):
        """
        Theory Check: 'Validates that an inflow BC actually provides values: {u, v, w}'
        """
        # 1. Test failure on missing component (w is missing)
        incomplete_inflow = [
            {"location": "x_min", "type": "inflow", "values": {"u": 5.0, "v": 0.0}}
        ]
        with pytest.raises(ValueError, match="Inflow boundary at x_min must provide numerical u, v, and w"):
            parse_boundary_conditions(incomplete_inflow)

        # 2. Test failure on non-numerical type
        bad_type_inflow = [
            {"location": "x_min", "type": "inflow", "values": {"u": "fast", "v": 0, "w": 0}}
        ]
        with pytest.raises(ValueError, match="Inflow boundary at x_min must provide numerical u, v, and w"):
            parse_boundary_conditions(bad_type_inflow)

    def test_pressure_action_compliance(self):
        """
        Theory Check: 'Validates that a pressure BC provides p'
        """
        # 1. Test failure on missing 'p'
        missing_p = [
            {"location": "x_max", "type": "pressure", "values": {"velocity": 10.0}}
        ]
        with pytest.raises(ValueError, match="Pressure boundary at x_max must provide numerical p"):
            parse_boundary_conditions(missing_p)

    def test_uniqueness_and_location_integrity(self):
        """
        Theory Check: 'The logic translates human-readable locations into coordinate-aware objects.'
        Ensures that we don't allow ambiguous or duplicate definitions.
        """
        # 1. Test duplicate location detection
        duplicate_loc = [
            {"location": "y_min", "type": "no-slip"},
            {"location": "y_min", "type": "free-slip"}
        ]
        with pytest.raises(ValueError, match="Duplicate boundary condition defined for location: y_min"):
            parse_boundary_conditions(duplicate_loc)

        # 2. Test invalid location string
        invalid_loc = [
            {"location": "top_surface", "type": "outflow"}
        ]
        with pytest.raises(ValueError, match="Invalid location 'top_surface'"):
            parse_boundary_conditions(invalid_loc)

    def test_full_schema_valid_config(self):
        """
        Verify that a complete, valid BC list passes and maintains data integrity.
        """
        valid_bcs = [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}},
            {"location": "x_max", "type": "pressure", "values": {"p": 0.0}},
            {"location": "z_min", "type": "no-slip"}
        ]
        parsed = parse_boundary_conditions(valid_bcs)
        assert len(parsed) == 3
        assert parsed[0]["values"]["u"] == 1.0
        assert "p" in parsed[1]["values"]