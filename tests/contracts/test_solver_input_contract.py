# tests/contracts/test_solver_input_contract.py

import pytest
from src.solver_input import SolverInput
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

class TestSolverInputContract:
    """
    THE CONSTITUTIONAL PROXY: Verifies that SolverInput acts as a 
    perfect behavioral implementation of the JSON Schema.
    """

    def test_round_trip_integrity(self):
        """
        GIVEN a raw dictionary that matches the JSON Schema
        WHEN it is hydrated into SolverInput and then serialized back via to_dict()
        THEN the output should match the input 100%, preserving all nested keys.
        """
        raw_input = solver_input_schema_dummy()
        
        # 1. Ingest (The Triage Gate)
        obj = SolverInput.from_dict(raw_input)
        
        # 2. Serialize (The Archive Format)
        serialized_output = obj.to_dict()
        
        # 3. Compare (Deep equality check)
        # Note: This ensures every key in the schema is captured by the class attributes
        assert serialized_output == raw_input

    def test_schema_minimum_constraints(self):
        """
        Verifies that 'minimum' and 'exclusiveMinimum' constraints from 
        the schema are enforced by the class @property setters.
        """
        obj = SolverInput()

        # Schema: nx, ny, nz must be integer >= 1
        with pytest.raises(ValueError, match="nx must be >= 1"):
            obj.grid.nx = 0
        
        # Schema: fluid density must be > 0 (exclusiveMinimum)
        with pytest.raises(ValueError, match="Density must be > 0"):
            obj.fluid_properties.density = 0

        # Schema: fluid viscosity must be >= 0 (minimum)
        with pytest.raises(ValueError, match="Viscosity must be >= 0"):
            obj.fluid_properties.viscosity = -0.001

        # Schema: mask items enum [-1, 0, 1]
        with pytest.raises(ValueError, match="Mask contains invalid values"):
            obj.mask.data = [2, 0, -1]

    def test_schema_array_size_constraints(self):
        """
        Ensures arrays like velocity and force_vector require exactly 3 items 
        as per schema 'minItems'/'maxItems' logic (3D Space Contract).
        """
        obj = SolverInput()

        # Initial Conditions Velocity [u, v, w]
        with pytest.raises(ValueError, match="velocity must have 3 items"):
            obj.initial_conditions.velocity = [0.0, 0.0] # Too short

        # External Forces Vector
        with pytest.raises(ValueError, match="force_vector must have 3 items"):
            obj.external_forces.force_vector = [0.0, -9.8, 0.0, 1.0] # Too long

    def test_boundary_condition_enums(self):
        """
        Matches the 'enum' constraints for location and type in the schema.
        Ensures illegal strings are rejected during the hydration phase.
        """
        obj = SolverInput()
        
        # Test invalid location enum (e.g., trying to set a BC on 'center')
        with pytest.raises(ValueError, match="Invalid location"):
            # BoundaryConditionsInput handles dict conversion internally
            obj.boundary_conditions.items = [{"location": "center", "type": "no-slip"}]
            
        # Test invalid type enum (e.g., 'warp-speed' is not a valid N-S boundary condition)
        with pytest.raises(ValueError, match="Invalid type"):
            obj.boundary_conditions.items = [{"location": "x_min", "type": "warp-speed"}]

    def test_optional_comments_preservation(self):
        """
        Ensures that optional 'comment' fields in the JSON are captured and 
        returned, satisfying the human-readability aspect of the schema.
        """
        raw_input = solver_input_schema_dummy()
        raw_input["external_forces"]["comment"] = "Testing custom comment"
        
        obj = SolverInput.from_dict(raw_input)
        
        assert obj.external_forces.comment == "Testing custom comment"
        assert obj.to_dict()["external_forces"]["comment"] == "Testing custom comment"