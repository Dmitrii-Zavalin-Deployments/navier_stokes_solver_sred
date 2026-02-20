# src/step3/orchestrate_step3.py

from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.solve_pressure import solve_pressure
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.update_health import update_health
from src.step3.log_step_diagnostics import log_step_diagnostics
from src.solver_state import SolverState

def orchestrate_step3(
    state: SolverState,
    current_time: float,
    step_index: int,
) -> SolverState:
    """
    Step 3 Conductor — Pressure projection and velocity correction.
    Renamed for consistency. Coordinates the star-step and correction.
    """

    # 1. Apply pre‑prediction boundaries (Sets Inflow/Wall velocities)
    fields_pre = apply_boundary_conditions_pre(state, state.fields)

    # 2. Predict intermediate velocity U* (Accepts fields_pre to satisfy updated signature)
    U_star, V_star, W_star = predict_velocity(state, fields_pre)

    # 3. Build PPE RHS (Divergence of U*)
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)

    # 4. Solve for pressure (Handles Dirichlet/Neumann switching)
    P_new, ppe_meta = solve_pressure(state, rhs)

    # 5. Correct velocity (Subtract Pressure Gradient)
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)

    # 6. Apply post‑correction boundaries (Lock physical boundaries)
    fields_post = apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)

    # 7. Write back fields (Synchronize State)
    state.fields["U"] = fields_post["U"]
    state.fields["V"] = fields_post["V"]
    state.fields["W"] = fields_post["W"]
    state.fields["P"] = P_new

    # 8. Update health diagnostics (CFL and Divergence checks)
    state.health = update_health(state, state.fields, P_new)

    # 9. Log diagnostics and update history
    diag = log_step_diagnostics(state, state.fields, current_time, step_index)
    
    # Update state history safely
    if not hasattr(state, "history") or state.history is None:
        state.history = {}
        
    history = state.history
    keys = ["times", "divergence_norms", "max_velocity_history", "ppe_iterations_history", "energy_history"]
    for key in keys:
        if key not in history: 
            history[key] = []
        
    history["times"].append(diag["time"])
    history["divergence_norms"].append(diag["divergence_norm"])
    history["max_velocity_history"].append(diag["max_velocity"])
    
    # Capture iteration count or status from metadata
    history["ppe_iterations_history"].append(ppe_meta.get("iterations", 0))
    history["energy_history"].append(diag["energy"])

    return state