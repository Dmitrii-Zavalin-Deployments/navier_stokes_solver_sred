# file: src/step3/orchestrate_step3.py

import numpy as np

from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.solve_pressure import solve_pressure
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.update_health import update_health
from src.step3.log_step_diagnostics import log_step_diagnostics

from src.solver_state import SolverState


def orchestrate_step3_state(
    state: SolverState,
    current_time: float,
    step_index: int,
) -> SolverState:
    """
    Step 3 — Pressure projection and velocity correction.
    Mutates state.fields, state.health, and state.history in place.
    """

    # ------------------------------------------------------------
    # 1. Apply pre‑prediction boundary conditions
    # ------------------------------------------------------------
    fields_pre = apply_boundary_conditions_pre(state, state.fields)

    # ------------------------------------------------------------
    # 2. Predict intermediate velocity U*
    # ------------------------------------------------------------
    U_star, V_star, W_star = predict_velocity(state, fields_pre)

    # ------------------------------------------------------------
    # 3. Build PPE right‑hand side
    # ------------------------------------------------------------
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)

    # ------------------------------------------------------------
    # 4. Solve for pressure
    # ------------------------------------------------------------
    P_new, _ppe_meta = solve_pressure(state, rhs)

    # ------------------------------------------------------------
    # 5. Correct velocity using pressure gradient
    # ------------------------------------------------------------
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)

    # ------------------------------------------------------------
    # 6. Apply post‑correction boundary conditions
    # ------------------------------------------------------------
    fields_post = apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)

    # ------------------------------------------------------------
    # 7. Write back fields
    # ------------------------------------------------------------
    state.fields["U"] = fields_post["U"]
    state.fields["V"] = fields_post["V"]
    state.fields["W"] = fields_post["W"]
    state.fields["P"] = P_new

    # ------------------------------------------------------------
    # 8. Update health diagnostics
    # ------------------------------------------------------------
    state.health = update_health(state, state.fields, P_new)

    # ------------------------------------------------------------
    # 9. Log diagnostics
    # ------------------------------------------------------------
    diag = log_step_diagnostics(state, state.fields, current_time, step_index)

    history = getattr(
        state,
        "history",
        {
            "times": [],
            "divergence_norms": [],
            "max_velocity_history": [],
            "ppe_iterations_history": [],
            "energy_history": [],
        },
    )

    history["times"].append(diag["time"])
    history["divergence_norms"].append(diag["divergence_norm"])
    history["max_velocity_history"].append(diag["max_velocity"])
    history["ppe_iterations_history"].append(diag["ppe_iterations"])
    history["energy_history"].append(diag["energy"])

    state.history = history

    return state
