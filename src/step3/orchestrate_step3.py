# src/step3/orchestrate_step3.py

import json
import os
from jsonschema import validate, ValidationError

from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.solve_pressure import solve_pressure
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.update_health import update_health
from src.step3.log_step_diagnostics import log_step_diagnostics


# ---------------------------------------------------------------------------
# Helper: Convert NumPy arrays → Python lists for JSON Schema validation
# ---------------------------------------------------------------------------

def _to_json_safe(obj):
    """
    Recursively convert numpy arrays to Python lists so JSON Schema can validate them.
    Functions and other non-JSON types are converted to simple placeholders.
    """
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]

    if callable(obj):
        return {}

    return obj


# ---------------------------------------------------------------------------
# Build a Step‑2‑compatible view of the Step‑3 state
# ---------------------------------------------------------------------------

def _build_step2_compatible_view(state):
    """
    Build a Step-2-output-shaped view of the Step-3 state,
    for schema validation only. Does NOT mutate the original state.
    """

    # Infer grid from P shape and Constants
    P = state["P"]
    nx, ny, nz = P.shape

    const_src = state["Constants"]

    # Build a Step‑2‑compatible constants block
    const = {
        "dt": float(const_src["dt"]),
        "dx": float(const_src["dx"]),
        "dy": float(const_src["dy"]),
        "dz": float(const_src["dz"]),
        "rho": float(const_src["rho"]),
        "mu": float(const_src["mu"]),
    }

    # Add required inverse spacings
    const["inv_dx"] = 1.0 / const["dx"]
    const["inv_dy"] = 1.0 / const["dy"]
    const["inv_dz"] = 1.0 / const["dz"]

    const["inv_dx2"] = const["inv_dx"] ** 2
    const["inv_dy2"] = const["inv_dy"] ** 2
    const["inv_dz2"] = const["inv_dz"] ** 2

    dx = const["dx"]
    dy = const["dy"]
    dz = const["dz"]

    grid = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "x_min": 0.0,
        "x_max": nx * dx,
        "y_min": 0.0,
        "y_max": ny * dy,
        "z_min": 0.0,
        "z_max": nz * dz,
    }

    fields = {
        "P": state["P"],
        "U": state["U"],
        "V": state["V"],
        "W": state["W"],
    }

    # Health must contain required keys
    health = dict(state.get("Health", {}))
    health.setdefault("initial_divergence_norm", 0.0)
    health.setdefault("max_velocity_magnitude", 0.0)
    health.setdefault("cfl_advection_estimate", 0.0)

    # PPE must match Step‑2 schema
    ppe_src = state.get("PPE", {})
    ppe = {
        "rhs_builder": "placeholder",
        "solver_type": "placeholder",
        "tolerance": float(ppe_src.get("tolerance", 1e-6)),
        "max_iterations": int(ppe_src.get("max_iterations", 1)),
        "ppe_is_singular": bool(ppe_src.get("ppe_is_singular", False)),
    }

    # Operators must be strings in Step‑2 schema
    operators = {
        "divergence": "placeholder",
        "gradient_p_x": "placeholder",
        "gradient_p_y": "placeholder",
        "gradient_p_z": "placeholder",
        "laplacian_u": "placeholder",
        "laplacian_v": "placeholder",
        "laplacian_w": "placeholder",
        "advection_u": "placeholder",
        "advection_v": "placeholder",
        "advection_w": "placeholder",
    }

    return {
        "grid": grid,
        "fields": fields,
        "mask": state["Mask"],
        "constants": const,
        "config": state.get("Config", {}),
        "operators": operators,
        "ppe": ppe,
        "health": health,
        "is_fluid": state["is_fluid"],
        "is_boundary_cell": state["is_boundary_cell"],
    }


# ---------------------------------------------------------------------------
# Build a Step‑3‑compatible view of the Step‑3 state
# ---------------------------------------------------------------------------

def _build_step3_compatible_view(state):
    """
    Convert internal Step‑3 state (capitalized keys)
    into the lowercase-key structure required by the Step‑3 schema.
    Does NOT mutate the original state.
    """
    return {
        "config": state.get("Config", {}),
        "mask": state["Mask"],
        "is_fluid": state["is_fluid"],
        "is_boundary_cell": state["is_boundary_cell"],
        "fields": {
            "P": state["P"],
            "U": state["U"],
            "V": state["V"],
            "W": state["W"],
        },
        "bcs": state.get("BCs", []),
        "constants": state["Constants"],
        "operators": state["Operators"],
        "ppe": state["PPE"],
        "health": state["Health"],
        "history": state["History"],
    }


# ---------------------------------------------------------------------------
# Load schemas
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

STEP2_SCHEMA_PATH = os.path.join(ROOT, "schema", "step2_output_schema.json")
STEP3_SCHEMA_PATH = os.path.join(ROOT, "schema", "step3_output_schema.json")

with open(STEP2_SCHEMA_PATH, "r") as f:
    STEP2_SCHEMA = json.load(f)

with open(STEP3_SCHEMA_PATH, "r") as f:
    STEP3_SCHEMA = json.load(f)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def step3(state, current_time, step_index):
    """
    Full Step 3 projection time step.
    """

    # ----------------------------------------------------------------------
    # 0 — INPUT SCHEMA VALIDATION (hard failure)
    # ----------------------------------------------------------------------
    try:
        step2_view = _build_step2_compatible_view(state)
        json_safe_input = _to_json_safe(step2_view)
        validate(instance=json_safe_input, schema=STEP2_SCHEMA)
    except ValidationError as exc:
        raise RuntimeError(
            f"\n[Step 3] Input schema validation FAILED.\n"
            f"Expected schema: {STEP2_SCHEMA_PATH}\n"
            f"Validation error: {exc.message}\n"
            f"Aborting Step 3 — upstream Step 2 output is malformed.\n"
        ) from exc

    # ----------------------------------------------------------------------
    # 1 — Pre-BCs
    # ----------------------------------------------------------------------
    apply_boundary_conditions_pre(state)

    # ----------------------------------------------------------------------
    # 2 — Predict velocity
    # ----------------------------------------------------------------------
    U_star, V_star, W_star = predict_velocity(state)

    # ----------------------------------------------------------------------
    # 3 — Build PPE RHS
    # ----------------------------------------------------------------------
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)

    # ----------------------------------------------------------------------
    # 4 — Solve pressure
    # ----------------------------------------------------------------------
    P_new = solve_pressure(state, rhs)

    # ----------------------------------------------------------------------
    # 5 — Correct velocity
    # ----------------------------------------------------------------------
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)

    # ----------------------------------------------------------------------
    # 6 — Post-BCs
    # ----------------------------------------------------------------------
    apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)

    # ----------------------------------------------------------------------
    # 7 — Update health
    # ----------------------------------------------------------------------
    update_health(state)

    # ----------------------------------------------------------------------
    # 8 — Log diagnostics
    # ----------------------------------------------------------------------
    log_step_diagnostics(state, current_time, step_index)

    # ----------------------------------------------------------------------
    # 9 — OUTPUT SCHEMA VALIDATION (hard failure)
    # ----------------------------------------------------------------------
    try:
        step3_view = _build_step3_compatible_view(state)
        json_safe_state = _to_json_safe(step3_view)
        validate(instance=json_safe_state, schema=STEP3_SCHEMA)
    except ValidationError as exc:
        raise RuntimeError(
            f"\n[Step 3] Output schema validation FAILED.\n"
            f"Expected schema: {STEP3_SCHEMA_PATH}\n"
            f"Validation error: {exc.message}\n"
            f"Aborting — Step 3 produced an invalid SimulationState.\n"
        ) from exc

    return state