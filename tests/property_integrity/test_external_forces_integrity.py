# tests/property_integrity/test_external_forces_integrity.py

import numpy as np
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# Consolidated tracking matrix for all stages
ALL_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

def get_force_vector(obj):
    """
    Architecture Bridge: Extracts the force vector regardless of whether 
    the object is a SolverState (manager-based) or StencilBlock (scalar-based).
    """
    if hasattr(obj, "external_forces"):
        return obj.external_forces.force_vector
    # Fallback for StencilBlocks which often store gx, gy, gz directly
    if hasattr(obj, "_gx"):
        return np.array([obj._gx, obj._gy, obj._gz])
    return None

@pytest.mark.parametrize("stage_name, factory", ALL_STAGES)
def test_external_force_integrity_and_persistence(stage_name, factory):
    """
    Physics & Integrity: Verifies force vector existence, dimensions, 
    and numeric validity (no NaN/Inf) across the entire lifecycle.
    """
    state = factory()
    force_vector = get_force_vector(state)
    
    # 1. Existence Check
    assert force_vector is not None, f"{stage_name}: External force data missing from object."
    
    # 2. Structural/Dimension Check
    assert isinstance(force_vector, np.ndarray), f"{stage_name}: Force must be a NumPy array."
    assert force_vector.shape == (3,), f"{stage_name}: Force must be a 3D vector [x, y, z]."
    
    # 3. Scientific Validity Check (Preventing Divergence)
    assert not np.any(np.isnan(force_vector)), f"{stage_name}: Force vector contains NaN."
    assert not np.any(np.isinf(force_vector)), f"{stage_name}: Force vector contains Inf."

def test_external_forces_immutability_logic():
    """
    Theory: Ensure the force vector (e.g., Gravity) remains constant 
    and does not diverge or mutate between initialization and final output.
    """
    s1 = make_step1_output_dummy()
    s_final = make_output_schema_dummy()
    
    v1 = get_force_vector(s1)
    v_final = get_force_vector(s_final)
    
    np.testing.assert_array_equal(
        v1, v_final,
        err_msg="External forces mutated/diverged between Step 1 and Final Output."
    )