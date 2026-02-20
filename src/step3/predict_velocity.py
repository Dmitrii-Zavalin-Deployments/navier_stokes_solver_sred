# src/step3/predict_velocity.py

import numpy as np

def predict_velocity(state) -> None:
    """
    Step-3 velocity prediction (The "Star" step).
    Calculates intermediate velocities U*, V*, W* using sparse operators
    for Diffusion and Advection from Step 2.
    """
    # 1. Physics and Time Constants
    rho = state.constants["rho"]
    mu  = state.constants.get("mu", 0.001)
    dt  = state.constants["dt"]
    nu  = mu / rho

    # 2. Get Current Velocity Fields
    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]

    # 3. Apply Sparse Operators (Diffusion & Advection)
    # Note: Operators act on flattened (raveled) vectors.
    def apply_op(field, op_key):
        op = state.operators.get(op_key)
        if op is None:
            return np.zeros_like(field)
        # Matrix-vector multiplication: A @ x
        return (op @ field.ravel()).reshape(field.shape)

    # Calculate Diffusion: nu * nabla^2(u)
    diff_u = apply_op(U, "lap_u")
    diff_v = apply_op(V, "lap_v")
    diff_w = apply_op(W, "lap_w")

    # Calculate Advection: (u . grad)u
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

    # 6. Enforce No-Slip at Solid Boundaries (Staggered Grid logic)
    if state.is_solid is not None:
        mask = state.is_solid
        # Zero out velocity components on faces shared with solid cells
        U_star[1:-1, :, :][mask[:-1, :, :] | mask[1:, :, :]] = 0.0
        V_star[:, 1:-1, :][mask[:, :-1, :] | mask[:, 1:, :]] = 0.0
        W_star[:, :, 1:-1][mask[:, :, :-1] | mask[:, :, 1:]] = 0.0
        
        # Domain boundary condition enforcement (Static Walls)
        U_star[0, :, :] = 0.0; U_star[-1, :, :] = 0.0
        V_star[:, 0, :] = 0.0; V_star[:, -1, :] = 0.0
        W_star[:, :, 0] = 0.0; W_star[:, :, -1] = 0.0

    # 7. Update State
    state.intermediate_fields["U_star"] = U_star
    state.intermediate_fields["V_star"] = V_star
    state.intermediate_fields["W_star"] = W_star