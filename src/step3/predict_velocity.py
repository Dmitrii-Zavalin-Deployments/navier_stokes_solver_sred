# src/step3/predict_velocity.py

import numpy as np

def predict_velocity(state, fields=None):
    """
    Step-3 velocity prediction (The "Star" step).
    Calculates intermediate velocities U*, V*, W* using sparse operators.
    """
    # 1. Physics and Time Constants
    rho = state.constants["rho"]
    mu  = state.constants.get("mu", 0.001)
    dt  = state.constants["dt"]
    nu  = mu / rho

    # 2. Get Velocity Fields
    f = fields if fields is not None else state.fields
    U, V, W = f["U"], f["V"], f["W"]

    # 3. Apply Sparse Operators (Diffusion & Advection)
    def apply_op(field, op_key):
        op = state.operators.get(op_key)
        if op is None:
            return np.zeros_like(field)
        return (op @ field.ravel()).reshape(field.shape)

    # 4. Compute Intermediate "Star" Velocities
    U_star = U + dt * (nu * apply_op(U, "lap_u") - apply_op(U, "advection_u"))
    V_star = V + dt * (nu * apply_op(V, "lap_v") - apply_op(V, "advection_v"))
    W_star = W + dt * (nu * apply_op(W, "lap_w") - apply_op(W, "advection_w"))

    # 5. Synchronize state for tests/PPE RHS
    state.intermediate_fields["U_star"] = U_star
    state.intermediate_fields["V_star"] = V_star
    state.intermediate_fields["W_star"] = W_star
    
    return U_star, V_star, W_star