# src/step3/build_ppe_rhs.py

import numpy as np

def build_ppe_rhs(state, U_star, V_star, W_star):
    """
    Step‑3 PPE RHS builder.
    Computes:
        rhs = (rho/dt) * divergence(U*, V*, W*)
    
    This uses the sparse divergence operator from Step 2.
    """
    rho = state.constants["rho"]
    dt = state.constants["dt"]

    # 1. Get the divergence operator (Sparse Matrix)
    div_op = state.operators["divergence"]

    # 2. Flatten velocity fields and concatenate for the divergence matrix
    # The divergence operator expects a vector [u1, u2... v1, v2... w1, w2...]
    u_flat = U_star.ravel()
    v_flat = V_star.ravel()
    w_flat = W_star.ravel()
    velocity_vector = np.concatenate([u_flat, v_flat, w_flat])

    # 3. Compute divergence: div = D @ u_star
    div_flat = div_op @ velocity_vector
    
    # 4. Reshape back to the Pressure field shape
    div = div_flat.reshape(state.fields["P"].shape)

    # 5. Scale by rho/dt
    rhs = (rho / dt) * div

    # 6. Zero RHS inside solid cells
    # Pressure is only solved in fluid/boundary cells
    if state.is_fluid is not None:
        rhs = np.where(state.is_fluid, rhs, 0.0)

    return rhs