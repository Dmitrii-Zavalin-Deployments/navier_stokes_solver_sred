import pytest
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_logic_gate_1_structural_unit_cube():
    """
    Logic Gate 1: Analytical Structural Verification.
    Scenario: nx=2, ny=2, nz=2, L=1.0
    Analytical Truth: dx=0.5, U_size=12, V_size=12, W_size=12, P_size=8
    """
    # 1. Input: Unit Cube with 2x2x2 resolution
    input_data = solver_input_schema_dummy()
    input_data["grid"].update({
        "nx": 2, "ny": 2, "nz": 2,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0
    })
    
    # 2. Execution
    state = orchestrate_step1(input_data)
    
    # 3. Verification of Analytical Truth (Success Metrics)
    
    # A. Grid Spacing Verification
    assert state.grid["dx"] == 0.5, f"Analytical Failure: dx should be 0.5, got {state.grid['dx']}"
    assert state.grid["dy"] == 0.5
    assert state.grid["dz"] == 0.5
    
    # B. Staggered Node Counts (Vector Lengths)
    # Logic: (N+1)*N*N for velocities, N*N*N for pressure
    u_nodes = state.fields["U"].size
    v_nodes = state.fields["V"].size
    w_nodes = state.fields["W"].size
    p_nodes = state.fields["P"].size
    
    assert u_nodes == 12, f"Logic Gate Violation: U_Vector expected 12 nodes, got {u_nodes}"
    assert v_nodes == 12, f"Logic Gate Violation: V_Vector expected 12 nodes, got {v_nodes}"
    assert w_nodes == 12, f"Logic Gate Violation: W_Vector expected 12 nodes, got {w_nodes}"
    assert p_nodes == 8,  f"Logic Gate Violation: P_Vector expected 8 nodes, got {p_nodes}"

    # C. Global Memory Identity
    expected_total = 12 + 12 + 12 + 8 # 44 nodes total
    actual_total = u_nodes + v_nodes + w_nodes + p_nodes
    assert actual_total == 44, "Structural Integrity Failure: Total memory footprint mismatch."

if __name__ == "__main__":
    # Manual run support
    test_logic_gate_1_structural_unit_cube()
    print("Logic Gate 1: PASSED")
