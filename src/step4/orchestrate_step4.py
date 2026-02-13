# src/step4/orchestrate_step4.py

from copy import deepcopy
from pathlib import Path

from src.step4.allocate_extended_fields import allocate_extended_fields
from src.step4.initialize_staggered_fields import initialize_staggered_fields
from src.step4.apply_all_boundary_conditions import apply_all_boundary_conditions
from src.step4.boundary_cell_treatment import apply_boundary_cell_treatment
from src.step4.precompute_rhs_source_terms import precompute_rhs_source_terms
from src.step4.verify_post_bc_state import verify_post_bc_state

# NEW: domain metadata subsystem
from src.step4.domain_metadata import build_domain_block

# NEW: rhs_source restructuring subsystem
from src.step4.assemble_rhs_source import assemble_rhs_source

# NEW: bc_applied expansion subsystem
from src.step4.assemble_bc_applied import assemble_bc_applied

# NEW: diagnostics subsystem
from src.step4.assemble_diagnostics import assemble_diagnostics

# NEW: history subsystem
from src.step4.assemble_history import assemble_history


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
    - Convert RHS to schema-compliant rhs_source
    - Verify post-BC state integrity
    - Expand bc_applied to schema format
    - Compute diagnostics
    - Initialize empty history block
    - Build full domain metadata block
    - Set final Step‑4 flags
    - Rename fields to match Step‑4 schema
    - Validate Step‑4 → Step‑5 output schema
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

    # Compute RHS in internal format
    state = precompute_rhs_source_terms(state)

    # Convert RHS → schema-compliant rhs_source
    state = assemble_rhs_source(state)

    # Post-BC integrity checks
    state = verify_post_bc_state(state)

    # Expand bc_applied to schema format
    state = assemble_bc_applied(state)

    # Compute diagnostics
    state = assemble_diagnostics(state)

    # Initialize empty history block
    state = assemble_history(state)

    # ---------------------------------------------------------
    # 3. Build full domain metadata block
    # ---------------------------------------------------------
    state = build_domain_block(state)

    # ---------------------------------------------------------
    # 4. Set final Step‑4 flags
    # ---------------------------------------------------------
    state["initialized"] = True
    state["ready_for_time_loop"] = True

    # ---------------------------------------------------------
    # 5. Rename Step‑4 extended fields to match schema
    # ---------------------------------------------------------
    if "P_ext" in state:
        state["p_ext"] = state.pop("P_ext")

    if "U_ext" in state:
        state["u_ext"] = state.pop("U_ext")

    if "V_ext" in state:
        state["v_ext"] = state.pop("V_ext")

    if "W_ext" in state:
        state["w_ext"] = state.pop("W_ext")

    # Rename RHS → rhs_source (already restructured)
    if "RHS" in state:
        state["rhs_source"] = state.pop("RHS")

    # Rename BCApplied → bc_applied (already expanded)
    if "BCApplied" in state:
        state["bc_applied"] = state.pop("BCApplied")

    # ---------------------------------------------------------
    # 6. Validate Step‑4 output schema
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
