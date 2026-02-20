# src/step3/predict_velocity.py

import numpy as np

def predict_velocity(state, fields=None):
    """
    Step-3 velocity prediction (The "Star" step).
    Calculates intermediate velocities U*, V*, W* using sparse operators.
    
    Args:
        state: The SolverState object.
        fields: Optional dictionary containing U, V, W (pre-enforced BCs).
    """
    # 1. Physics and Time Constants
    rho = state.constants["rho"]
    mu  = state.constants.get("mu", 0.001)
    dt  = state.constants["dt"]
    nu  = mu / rho

    # 2. Get Velocity Fields (favoring input fields from BC pre-step)
    f = fields if fields is not None else state.fields
    U = f["U"]
    V = f["V"]
    W = f["W"]

    # 3. Apply Sparse Operators (Diffusion & Advection)
    def apply_op(field, op_key):
        op = state.operators.get(op_key)
        if op is None:
            return np.zeros_like(field)
        # Matrix-vector multiplication for flattened fields
        return (op @ field.ravel()).reshape(field.shape)

    diff_u = apply_op(U, "lap_u")
    diff_v = apply_op(V, "lap_v")
    diff_w = apply_op(W, "lap_w")

    adv_u = apply_op(U, "advection_u")
    adv_v = apply_op(V, "advection_v")
    adv_w = apply_op(W, "advection_w")

    # 4. External Forces
    forces = state.config.get("external_forces", {})
    fx = forces.get("fx", 0.0)
    fy = forces.get("fy", 0.0)
    fz = forces.get("fz", 0.0)

    # 5. Compute Intermediate "Star" Velocities
    # u* = u^n + dt * [ nu * laplacian - advection + forces ]
    U_star = U + dt * (nu * diff_u - adv_u + fx)
    V_star = V + dt * (nu * diff_v - adv_v + fy)
    W_star = W + dt * (nu * diff_w - adv_w + fz)

    # Note: Boundary enforcement and solid masking are now handled 
    # by the orchestrator via apply_boundary_conditions_post.
    
    return U_star, V_star, W_star