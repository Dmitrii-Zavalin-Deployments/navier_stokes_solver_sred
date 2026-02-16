# src/step2/orchestrate_step2.py
from __future__ import annotations
from src.solver_state import SolverState

from .enforce_mask_semantics import enforce_mask_semantics
from .create_fluid_mask import create_fluid_mask
from .build_divergence_operator import build_divergence_operator
from .build_gradient_operators import build_gradient_operators
from .build_laplacian_operators import build_laplacian_operators
from .build_advection_structure import build_advection_structure
from .prepare_ppe_structure import prepare_ppe_structure
from .compute_initial_health import compute_initial_health


def orchestrate_step2(state: SolverState) -> SolverState:
    enforce_mask_semantics(state)
    create_fluid_mask(state)
    build_divergence_operator(state)
    build_gradient_operators(state)
    build_laplacian_operators(state)
    build_advection_structure(state)
    prepare_ppe_structure(state)
    compute_initial_health(state)
    return state
