# tests/step1/test_step1_output_dummy_schema.py

import numpy as np
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_schema import EXPECTED_STEP1_SCHEMA
from src.step1.orchestrate_step1 import orchestrate_step1


def test_step1_output_matches_schema():
    # 1. Build canonical input (frozen dummy)
    input_dict = solver_input_schema_dummy()

    # 2. Run real Step 1 (returns a dict)
    state = orchestrate_step1(input_dict)

    # 3. Check top-level keys match schema
    for key in EXPECTED_STEP1_SCHEMA:
        assert key in state, f"Missing key: {key}"

    # 4. Validate fields
    assert "fields" in state
    for f in ["P", "U", "V", "W"]:
        assert f in state["fields"]
        assert isinstance(state["fields"][f], np.ndarray)

    # 5. Mask semantics
    assert isinstance(state["mask"], np.ndarray)
    assert isinstance(state["is_fluid"], np.ndarray)
    assert isinstance(state["is_boundary_cell"], np.ndarray)

    # 6. Constants
    assert isinstance(state["constants"], dict)

    # 7. Boundary conditions
    bc = state["boundary_conditions"]
    assert bc is None or callable(bc)

    # 8. Empty containers Step 1 must initialize
    assert isinstance(state["operators"], dict)
    assert isinstance(state["ppe"], dict)
    assert isinstance(state["health"], dict)

    # ❌ No history check — Step 1 no longer defines it
