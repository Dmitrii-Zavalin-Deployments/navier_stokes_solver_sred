# src/step3/predict_velocity.py

import numpy as np

def predict_velocity(state, fields=None):
    """
    Step-3 velocity prediction (The "Star" step).
    u* = u^n + dt * (nu * nabla^2 u^n - (u^n . nabla) u^n)
    """
    # 1. Physics and Time Constants (Hardened access)
    rho = state.constants.get("rho", 1.0)
    mu  = state.constants.get("mu", 0.001)
    # Using .get with a fallback or checking config to prevent KeyError
    dt  = state.config.get("dt", state.constants.get("dt", 0.01))
    nu  = mu / rho

    # 2. Get Velocity Fields
    f = fields if fields is not None else state.fields
    U, V, W = f["U"], f["V"], f["W"]

    # 3. Helper for Sparse Operator Application
    def apply_op(field, op_key):
        op = state.operators.get(op_key)
        if op is None:
            # If an operator is missing, return zero-change rather than crashing
            return np.zeros_like(field)
        
        # Flattened size check
        expected_size = field.size
        if op.shape[1] != expected_size:
            raise ValueError(
                f"Operator '{op_key}' shape {op.shape} incompatible with field shape {field.shape}. "
                f"Expected size {expected_size}."
            )
            
        return (op @ field.ravel()).reshape(field.shape)

    # 4. Compute Intermediate "Star" Velocities
    # Diffusion (nu * L) and Advection (A)
    U_star = U + dt * (nu * apply_op(U, "lap_u") - apply_op(U, "advection_u"))
    V_star = V + dt * (nu * apply_op(V, "lap_v") - apply_op(V, "advection_v"))
    W_star = W + dt * (nu * apply_op(W, "lap_w") - apply_op(W, "advection_w"))

    # 5. Synchronize state for PPE RHS calculation
    state.intermediate_fields["U_star"] = U_star
    state.intermediate_fields["V_star"] = V_star
    state.intermediate_fields["W_star"] = W_star
    
    return U_star, V_star, W_star