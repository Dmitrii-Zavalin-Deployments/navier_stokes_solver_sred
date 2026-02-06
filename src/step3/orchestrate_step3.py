# src/step3/orchestrate_step3.py

from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.solve_pressure import solve_pressure
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.update_health import update_health
from src.step3.log_step_diagnostics import log_step_diagnostics

def step3(state, current_time, step_index):
    """
    Full Step 3 projection time step.
    """

    apply_boundary_conditions_pre(state)
    U_star, V_star, W_star = predict_velocity(state)
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)
    P_new = solve_pressure(state, rhs)
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)
    apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)
    update_health(state)
    log_step_diagnostics(state, current_time, step_index)

    return state
