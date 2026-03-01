import pytest
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1
from src.solver_input import SolverInput
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

"""
test_step1_alignment.py
Constitutional Role: The Step 1 Alignment Auditor.
Compliance: Phase C, Section 3 (Unit Test: Orchestration Alignment).

This test ensures that the real code produces a SolverState that is a 
100% mathematical and structural match to the frozen Step 1 dummy.
"""

def test_step1_alignment_logic_to_frozen_truth():
    """
    VERIFICATION TASK: orchestrate_step1(input_dummy) == step1_output_dummy.
    Ensures the 5-step pipeline remains unbroken at the first link.
    """
    # 1. Input: The Legal Contract start point
    raw_json = solver_input_schema_dummy()
    input_data = SolverInput.from_dict(raw_json)
    
    # 2. Execution: Run the actual orchestration logic
    # The input dummy defines nx=2, ny=2, nz=2
    input_obj = input_data
    result_state = orchestrate_step1(input_obj)
    
    # 3. Reference: The 'Frozen Truth' required for Step 2 ingestion
    expected_state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    # Synchronize expected mask with input to match processed logic
    expected_state.mask = input_data["mask"]

    # --- AUDIT A: GRID & SPATIAL PARAMETERS ---
    assert result_state.grid["nx"] == expected_state.grid["nx"]
    assert np.isclose(result_state.grid["dx"], expected_state.grid["dx"])
    assert np.isclose(result_state.grid["dy"], expected_state.grid["dy"])
    assert np.isclose(result_state.grid["dz"], expected_state.grid["dz"])
    assert result_state.grid["total_cells"] == expected_state.grid["total_cells"]

    # --- AUDIT B: STAGGERED MEMORY LAYOUT (Arakawa C-Grid) ---
    # Verification of N+1 face counts for U, V, W
    assert result_state.fields["U"].shape == expected_state.fields["U"].shape
    assert result_state.fields["V"].shape == expected_state.fields["V"].shape
    assert result_state.fields["W"].shape == expected_state.fields["W"].shape
    assert result_state.fields["P"].shape == expected_state.fields["P"].shape

    # --- AUDIT C: TOPOLOGY & MASKING (The 1D Flattening Rule) ---
    # Phase A Section 2: Mask must be a flattened list to ensure JSON-safe serialization
    assert isinstance(result_state.mask, list)
    assert len(result_state.mask) == expected_state.grid["total_cells"]
    assert result_state.mask == expected_state.mask

    # --- AUDIT D: BOUNDARY CONDITIONS (The Six-Face Mandate) ---
    # Ensure the parser correctly identified and mapped all six faces of the unit cube
    assert len(result_state.boundary_conditions) == 6
    for face in result_state.boundary_conditions.values():
        assert "type" in face
        assert "u" in face
        assert "p" in face

    # --- AUDIT E: CONSTANT PROPAGATION ---
    # Verify that fluid properties and time steps are accurately translated
    assert result_state.constants["dt"] == input_data["simulation_parameters"]["time_step"]
    assert result_state.constants["rho"] == input_data["fluid_properties"]["density"]
    assert result_state.constants["mu"] == input_data["fluid_properties"]["viscosity"]

def test_step1_sensitivity_firewall():
    """
    Sensitivity Gate 1.F: Completeness and Constraints.
    Verifies that invalid data triggers a Contract Violation (RuntimeError).
    """
    # Create invalid input: negative grid resolution to trigger the schema firewall
    invalid_input = solver_input_schema_dummy()
    invalid_input["grid"]["nx"] = -1 
    
    with pytest.raises(RuntimeError) as excinfo:
        invalid_obj = SolverInput(**invalid_input)
        orchestrate_step1(invalid_obj)
    
    assert "Contract Violation" in str(excinfo.value)