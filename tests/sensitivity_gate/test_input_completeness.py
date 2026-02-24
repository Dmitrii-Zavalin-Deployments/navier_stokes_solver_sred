import pytest
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_six_face_mandate_violation():
    """
    Gate 1.F: Completeness.
    Must raise RuntimeError (Contract Violation) if a face is missing.
    """
    invalid_input = solver_input_schema_dummy()
    # Remove one mandatory face (e.g., z_max)
    invalid_input["boundary_conditions"] = [
        bc for bc in invalid_input["boundary_conditions"] 
        if bc["location"] != "z_max"
    ]
    
    with pytest.raises(RuntimeError, match="Incomplete domain") as excinfo:
        orchestrate_step1(invalid_input)

def test_domain_inversion_error():
    """
    Gate 1.B: Inversion.
    Must raise RuntimeError if x_max <= x_min.
    """
    invalid_input = solver_input_schema_dummy()
    invalid_input["grid"]["x_min"] = 10.0
    invalid_input["grid"]["x_max"] = 5.0 # Inversion!
    
    with pytest.raises(RuntimeError, match="Domain inversion") as excinfo:
        orchestrate_step1(invalid_input)

def test_zero_volume_resolution_error():
    """
    Gate 1.B: Ensuring positive cell counts.
    """
    invalid_input = solver_input_schema_dummy()
    invalid_input["grid"]["nx"] = 0
    
    with pytest.raises(RuntimeError, match="Resolution must be positive") as excinfo:
        orchestrate_step1(invalid_input)
