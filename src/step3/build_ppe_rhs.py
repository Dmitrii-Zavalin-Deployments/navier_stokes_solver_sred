import numpy as np

def build_ppe_rhs(state, U_star, V_star, W_star):
    """Computes: rhs = (rho/dt) * divergence(U_star) using staggered fields."""
    rho = state.constants.get("rho", 1.0)
    dt = state.config.get("dt", state.constants.get("dt", 0.01))
    div_op = state.operators["divergence"]

    # Flatten and concatenate staggered fields: (nx+1)ny*nz + nx(ny+1)nz + nx*ny(nz+1)
    velocity_vector = np.concatenate([U_star.ravel(), V_star.ravel(), W_star.ravel()])

    # Shape Guard: Ensure the operator matches the concatenated staggered vector
    if div_op.shape[1] != velocity_vector.size:
        raise ValueError(f"Divergence operator columns ({div_op.shape[1]}) "
                         f"do not match staggered vector size ({velocity_vector.size}).")

    # Matrix multiplication for divergence -> Result is cell-centered (nx, ny, nz)
    div_flat = div_op @ velocity_vector
    div = div_flat.reshape(state.fields["P"].shape)

    rhs = (rho / dt) * div
    
    # Apply fluid mask: zero out RHS inside solids
    if state.is_fluid is not None:
        rhs = np.where(state.is_fluid, rhs, 0.0)

    return rhs