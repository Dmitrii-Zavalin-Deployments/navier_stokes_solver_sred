# src/step2/orchestrate_step2.py

from __future__ import annotations
from src.solver_state import SolverState

from .load_numerical_config import load_numerical_config
from .enforce_mask_semantics import enforce_mask_semantics
from .create_fluid_mask import create_fluid_mask
from .build_divergence_operator import build_divergence_operator
from .build_gradient_operators import build_gradient_operators
from .build_laplacian_operators import build_laplacian_operators
from .build_advection_structure import build_advection_structure
from .prepare_ppe_structure import prepare_ppe_structure
from .compute_initial_health import compute_initial_health

def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Finalized Orchestrator for Step 2.
    Ensures mask logic, operator construction, and health checks are performed.
    """
    
    # 0. Load Numerical Tuning (from config.json)
    load_numerical_config(state)
    # Phase C: Step 1 -> Step 2 Handover Validation
    if "rho" not in state.constants:
        raise KeyError("Data Integrity Error: rho missing from state.constants")
    if "dt" not in state.constants:
        raise KeyError("Data Integrity Error: dt missing from state.constants")
    # 0.5 Physical Parameter Guard (Phase C: Data Integrity)
    if "physics" not in state.config or "rho" not in state.config["physics"]:
        raise KeyError("Data Integrity Error: Physics configuration (rho) missing.")
    state.constants["rho"] = state.config["physics"]["rho"]

    if "temporal" not in state.config or "dt" not in state.config["temporal"]:
        raise KeyError("Data Integrity Error: Temporal configuration (dt) missing.")
    state.config["dt"] = state.config["temporal"]["dt"]
    
    # 1. Derived Constants (Required by test_orchestrate_step2)
    # We pre-calculate inverse spacings to speed up operator applications later.
    state.constants["inv_dx"] = 1.0 / state.grid['dx']
    state.constants["inv_dy"] = 1.0 / state.grid['dy']
    state.constants["inv_dz"] = 1.0 / state.grid['dz']
    
    # 2. Geometry & Masking
    enforce_mask_semantics(state)
    create_fluid_mask(state)
    
    # 3. Sparse Operator Construction
    # build_divergence_operator creates the 'divergence' key in state.operators
    build_divergence_operator(state)
    build_gradient_operators(state)
    build_laplacian_operators(state)
    
    # 4. Numerical Structures
    build_advection_structure(state)
    prepare_ppe_structure(state)
    
    # 5. Initialization Health Check
    compute_initial_health(state)
    
    state.ready_for_time_loop = True
    return state