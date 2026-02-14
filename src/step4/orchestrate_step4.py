# src/step4/orchestrate_step4.py

import numpy as np
from src.common.json_safe import to_json_safe

from src.step4.initialize_extended_fields import initialize_extended_fields
from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics


# ---------------------------------------------------------
# Global debug flag for Step‑4
# ---------------------------------------------------------
DEBUG_STEP4 = True


# ---------------------------------------------------------
# Structured debug inspector for Step‑4
# (limited to the most relevant keys)
# ---------------------------------------------------------
def debug_state_step4(state):
    print("\n==================== DEBUG: STEP‑4 STATE SUMMARY ====================")

    interesting_keys = [
        "P_ext", "U_ext", "V_ext", "W_ext",
        "diagnostics", "ready_for_time_loop",
        "mask", "fields"
    ]

    for key in interesting_keys:
        if key not in state:
            continue

        value = state[key]
        print(f"\n• {key}: {type(value)}")

        if isinstance(value, np.ndarray):
            print(f"    ndarray shape={value.shape}, dtype={value.dtype}")

        elif isinstance(value, dict):
            print(f"    dict keys={list(value.keys())}")

        else:
            print(f"    value={value}")

    print("====================================================================\n")


def orchestrate_step4(
    state,
    validate_json_schema=None,
    load_schema=None,
):
    """
    Step 4 — Initialization of extended fields + boundary conditions.

    Simplified, production‑oriented responsibilities:
    - Validate Step‑3 output schema (optional).
    - Initialize extended staggered fields (allocate + fill + mask semantics).
    - Apply boundary conditions (velocity + pressure).
    - Assemble diagnostics (fluid cells, max velocity, divergence, BC violations).
    - Mark ready_for_time_loop = True.

    All other metadata/history/RHS/priority/sync layers from the old Step‑4
    architecture have been intentionally removed for clarity and maintainability.
    """

    # ------------------------------------------------------------
    # 0 — Validate Step‑3 output (optional)
    # ------------------------------------------------------------
    if validate_json_schema and load_schema:
        try:
            step3_schema = load_schema("step3_output_schema.json")
            validate_json_schema(
                instance=to_json_safe(state),
                schema=step3_schema,
                context_label="[Step 4] Input schema validation",
            )
        except Exception as exc:
            raise RuntimeError(
                "\n[Step 4] Input schema validation FAILED.\n"
                f"Validation error: {exc}\n"
            ) from exc

    # Defensive shallow copy
    base_state = dict(state)

    # ------------------------------------------------------------
    # 1 — Initialize extended fields (allocate + fill + mask semantics)
    # ------------------------------------------------------------
    base_state = initialize_extended_fields(base_state)

    # ------------------------------------------------------------
    # 2 — Apply boundary conditions
    # ------------------------------------------------------------
    base_state = apply_boundary_conditions(base_state)

    # ------------------------------------------------------------
    # 3 — Assemble diagnostics
    # ------------------------------------------------------------
    base_state = assemble_diagnostics(base_state)

    # ------------------------------------------------------------
    # 4 — Optional structured debug print
    # ------------------------------------------------------------
    if DEBUG_STEP4:
        debug_state_step4(base_state)

    # ------------------------------------------------------------
    # 5 — Output schema validation (optional)
    # ------------------------------------------------------------
    if validate_json_schema and load_schema:
        try:
            step4_schema = load_schema("step4_output_schema.json")
            validate_json_schema(
                instance=to_json_safe(base_state),
                schema=step4_schema,
                context_label="[Step 4] Output schema validation",
            )
        except Exception as exc:
            raise RuntimeError(
                "\n[Step 4] Output schema validation FAILED.\n"
                f"Validation error: {exc}\n"
            ) from exc

    # ------------------------------------------------------------
    # 6 — Mark ready for time loop
    # ------------------------------------------------------------
    base_state["ready_for_time_loop"] = True

    return base_state


def step4(state, **kwargs):
    """Convenience wrapper."""
    return orchestrate_step4(state, **kwargs)
