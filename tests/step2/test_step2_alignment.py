# tests/step2/test_step2_alignment.py

import pytest
import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_step2_orchestration_alignment():
    """
    PHASE C: Orchestration Alignment Audit for Step 2.
    Validates the contract: Step 1 Output -> Orchestrate Step 2 -> Step 2 Frozen Truth.
    """
    # 1. Input: Load the 'Frozen' output from Step 1
    input_state = make_step1_output_dummy()
    
    # 2. Logic: Execute the actual Step 2 orchestration
    # This calls your verified Laplacian, Gradient, and Divergence builders
    output_state = orchestrate_step2(input_state)
    
    # 3. Target: Load the 'Frozen Truth' dummy for Step 2
    expected_state = make_step2_output_dummy()
    
    # 4. Verification: The Zero-Debt Mandate
    # Check that all required operators were built and stored
    assert output_state.ready_for_time_loop is True
    assert "laplacian" in output_state.operators
    assert "gradient" in output_state.operators
    assert "divergence" in output_state.operators
    
    # 5. Verification: Numerical Integrity
    # Ensure the Laplacian matrix in the output matches the expected dummy structure
    actual_L = output_state.operators["laplacian"]
    expected_L = expected_state.operators["laplacian"]
    
    assert actual_L.shape == expected_L.shape, "Matrix dimensions mismatch!"
    assert actual_L.nnz == expected_L.nnz, "Sparsity pattern mismatch!"
    
    # Verify a small slice of data to ensure values are correct (dx, dy, dz scaling)
    np.testing.assert_allclose(actual_L.data[:10], expected_L.data[:10], rtol=1e-7)

    print("\n✅ Step 2 Orchestration Alignment: PASSED")