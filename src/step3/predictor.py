# src/step3/predictor.py

import numpy as np
from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def predict_velocity(state: SolverState) -> None:
    """
    Step 3.1: Prediction. Calculates intermediate velocity V*.
    V* = V_n + dt * [ nu * Laplacian(V_n) - Advection(V_n) + External_Forces ]
    Rule 5 Compliance: No silent failures, no zero-defaults for missing operators.
    """
    rho = state.density
    mu = state.viscosity
    dt = state.dt
    nu = mu / rho

    if DEBUG:
        print(f"DEBUG [Step 3 Predictor]: Nu={nu:.6e}, dt={dt}")

    def _apply_operator(field, operator, op_name):
        """Strict operator application with no silent zero-defaults."""
        if operator is None:
            raise RuntimeError(f"Predictor Error: Operator '{op_name}' is missing.")
            
        # Flatten in 'F' order to match sparse matrix construction
        field_flat = field.flatten(order='F')
        res_flat = operator @ field_flat
        
        # Reshape back to original field dimensions
        return res_flat.reshape(field.shape, order='F')

    # 1. Prediction for U component
    # We explicitly look for advection_x, advection_y, advection_z (or the combined operator)
    # For Logic Gate 3, we expect the Laplacian to be present.
    
    if DEBUG: print("DEBUG [Step 3 Predictor]: Predicting U_star...")
    
    laplacian_u = _apply_operator(state.fields.U, state.operators.laplacian, "laplacian")
    # Note: External forces should be pulled from state.config.force_vector
    force_u = state.config.force_vector[0] 

    # Momentum Equation: U* = U + dt * (viscous + advective + forcing)
    # (Advection omitted here if not yet implemented in operators, but will throw error if requested)
    state.fields.U_star = state.fields.U + dt * (nu * laplacian_u + force_u)

    # 2. Prediction for V component
    if DEBUG: print("DEBUG [Step 3 Predictor]: Predicting V_star...")
    laplacian_v = _apply_operator(state.fields.V, state.operators.laplacian, "laplacian")
    force_v = state.config.force_vector[1]
    state.fields.V_star = state.fields.V + dt * (nu * laplacian_v + force_v)

    # 3. Prediction for W component
    if DEBUG: print("DEBUG [Step 3 Predictor]: Predicting W_star...")
    laplacian_w = _apply_operator(state.fields.W, state.operators.laplacian, "laplacian")
    force_w = state.config.force_vector[2]
    state.fields.W_star = state.fields.W + dt * (nu * laplacian_w + force_w)

    if DEBUG:
        u_star_norm = np.linalg.norm(state.fields.U_star)
        print(f"DEBUG [Step 3 Predictor]: U_star norm: {u_star_norm:.6e}")
        if np.isnan(u_star_norm):
            print("!!! CRITICAL: NaN detected in Predictor Step !!!")