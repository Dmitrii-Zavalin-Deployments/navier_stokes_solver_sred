# file: step2/orchestrate_step2.py
from __future__ import annotations

from typing import Any

# Import all Step 2 functions
from .enforce_mask_semantics import enforce_mask_semantics
from .precompute_constants import precompute_constants
from .create_fluid_mask import create_fluid_mask
from .build_divergence_operator import build_divergence_operator
from .build_gradient_operators import build_gradient_operators
from .build_laplacian_operators import build_laplacian_operators
from .build_advection_structure import build_advection_structure
from .prepare_ppe_structure import prepare_ppe_structure
from .compute_initial_health import compute_initial_health


def orchestrate_step2(state: Any) -> Any:
    """
    High-level orchestrator for Step 2.

    This function takes a fully validated SimulationState from Step 1 and
    enhances it with:
      - semantic mask validation
      - boolean fluid masks
      - discrete operators (div, grad, laplacian, advection)
      - PPE structure (rhs builder, solver config, singularity flag)
      - initial solver health diagnostics

    Parameters
    ----------
    state : Any
        SimulationState produced by Step 1.

    Returns
    -------
    state : Any
        Enhanced SimulationState ready for Step 3.
    """

    # ------------------------------------------------------------
    # 1. Enforce CFD mask semantics
    # ------------------------------------------------------------
    enforce_mask_semantics(state)

    # ------------------------------------------------------------
    # 2. Precompute constants (dx, dy, dz, rho, mu, dt, etc.)
    # ------------------------------------------------------------
    precompute_constants(state)

    # ------------------------------------------------------------
    # 3. Create boolean fluid masks
    # ------------------------------------------------------------
    create_fluid_mask(state)

    # ------------------------------------------------------------
    # 4. Build discrete operators
    # ------------------------------------------------------------
    build_divergence_operator(state)
    build_gradient_operators(state)
    build_laplacian_operators(state)
    build_advection_structure(state)

    # ------------------------------------------------------------
    # 5. Prepare PPE structure (rhs builder, solver config, singularity flag)
    # ------------------------------------------------------------
    prepare_ppe_structure(state)

    # ------------------------------------------------------------
    # 6. Compute initial solver health diagnostics
    # ------------------------------------------------------------
    compute_initial_health(state)

    # ------------------------------------------------------------
    # Done — state is now fully Step 2–enhanced
    # ------------------------------------------------------------
    return state
