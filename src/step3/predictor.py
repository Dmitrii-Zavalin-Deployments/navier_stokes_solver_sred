# src/step3/predictor.py

import numpy as np
from src.solver_state import SolverState

def predict_velocity(state: SolverState) -> None:
    """
    Step 3.1: Prediction. Calculates intermediate velocity V*.
    Rule 5: Explicit or Error. No .get() or fallbacks for rho, mu, or dt.
    """
    rho = state.constants.rho
    mu = state.constants.mu
    dt = state.config.dt
    nu = mu / rho

    # Local helper for sparse application to maintain O(N^3) Scale Guard
    def _apply(field, operator):
        return (operator @ field.ravel()).reshape(field.shape)

    # Calculate U*, V*, W* using Laplacian and Advection operators
    state.fields.U_star = state.fields.U + dt * (
        nu * _apply(state.fields.U, state.operators.laplacian) - 
        _apply(state.fields.U, state.operators.advection_u)
    )
    
    state.fields.V_star = state.fields.V + dt * (
        nu * _apply(state.fields.V, state.operators.laplacian) - 
        _apply(state.fields.V, state.operators.advection_v)
    )
    
    state.fields.W_star = state.fields.W + dt * (
        nu * _apply(state.fields.W, state.operators.laplacian) - 
        _apply(state.fields.W, state.operators.advection_w)
    )