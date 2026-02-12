# src/step4/orchestrate_step4.py

from copy import deepcopy
from pathlib import Path

from src.step4.allocate_extended_fields import allocate_extended_fields
from src.step4.initialize_staggered_fields import initialize_staggered_fields
from src.step4.apply_all_boundary_conditions import apply_all_boundary_conditions
from src.step4.boundary_cell_treatment import apply_boundary_cell_treatment
from src.step4.precompute_rhs_source_terms import precompute_rhs_source_terms
from src.step4.verify_post_bc_state import verify_post_bc_state


def orchestrate_step4(
    state,
    *,
    validate_json_schema=None,
    load_schema=None,
):
    """
    Orchestrate Step 4 of the solver pipeline.

    Responsibilities:
    - Defensive copy of input state
    - Validate Step‑3 → Step‑4 input schema
    - Allocate extended (halo) fields
    - Initialize staggered fields
    - Apply all boundary conditions
    - Apply boundary-fluid treatment
    - Precompute RHS source terms
    - Verify post-BC state integrity
    - Rename fields to match Step‑4 schema
    - Validate Step‑4 → Step‑5 output schema

    This function is intentionally thin: it delegates all real work to
    the subsystem modules. It simply defines the execution order.
    """

    # ---------------------------------------------------------
    # Defensive copy — Step 4 must not mutate caller state
    # ---------------------------------------------------------
    state = deepcopy(state)

    # ---------------------------------------------------------
    # 1. Validate Step‑3 output (production safety)
    # ---------------------------------------------------------
    if validate_json_schema and load_schema:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schema" / "step3_output_schema.json"
        )
        schema = load_schema(str(schema_path))
        try:
            validate_json_schema(state, schema)
        except Exception as exc:
            raise RuntimeError(
                "\n[Step 4] Input schema validation FAILED.\n"
                "The Step‑3 output does not match step3_output_schema.json.\n"
                f"Validation error: {exc}\n"
            ) from exc

    # ---------------------------------------------------------
    # 2. Execute Step‑4 pipeline
    # ---------------------------------------------------------
    state = allocate_extended_fields(state)
    state = initialize_staggered_fields(state)
    state = apply_all_boundary_conditions(state)
    state = apply_boundary_cell_treatment(state)
    state = precompute_rhs_source_terms(state)
    state = verify_post_bc_state(state)

    # ---------------------------------------------------------
    # 3. Rename Step‑4 extended fields to match schema
    # ---------------------------------------------------------
    if "P_ext" in state:
        state["p_ext"] = state.pop("P_ext")

    if "U_ext" in state:
        state["u_ext"] = state.pop("U_ext")

    if "V_ext" in state:
        state["v_ext"] = state.pop("V_ext")

    if "W_ext" in state:
        state["w_ext"] = state.pop("W_ext")

    # Rename RHS → rhs_source (structure expanded later)
    if "RHS" in state:
        state["rhs_source"] = state.pop("RHS")

    # Rename BCApplied → bc_applied (structure expanded later)
    if "BCApplied" in state:
        state["bc_applied"] = state.pop("BCApplied")

    # ---------------------------------------------------------
    # 4. Validate Step‑4 output schema
    # ---------------------------------------------------------
    if validate_json_schema and load_schema:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schema" / "step4_output_schema.json"
        )
        schema = load_schema(str(schema_path))
        try:
            validate_json_schema(state, schema)
        except Exception as exc:
            raise RuntimeError(
                "\n[Step 4] Output schema validation FAILED.\n"
                "The Step‑4 output does not match step4_output_schema.json.\n"
                f"Validation error: {exc}\n"
            ) from exc

    return state
