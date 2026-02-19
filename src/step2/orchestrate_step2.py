# src/step2/orchestrate_step2.py

from __future__ import annotations
from src.solver_state import SolverState

# New internal dependency for numerical tuning
from .load_numerical_config import load_numerical_config

from .enforce_mask_semantics import enforce_mask_semantics
from .create_fluid_mask import create_fluid_mask
from .build_divergence_operator import build_divergence_operator
from .build_gradient_operators import build_gradient_operators
from .build_laplacian_operators import build_laplacian_operators
from .build_advection_structure import build_advection_structure
from .prepare_ppe_structure import prepare_ppe_structure
from .compute_initial_health import compute_initial_health


def orchestrate_step2_state(state: SolverState) -> SolverState:
    """
    Step 2: Operators & PPE Structure.
    
    Transforms the geometric description from Step 1 into computational 
    operators (Sparse Matrices) and prepares the Pressure Poisson Equation.
    
    All calculations are derived from state.grid, state.constants, 
    state.mask, and the numerical config.
    """
    
    # 0. Configuration Loading
    # Independent parser: loads solver_type, tolerance, etc., from config.json
    load_numerical_config(state)
    
    # 1. Geometry & Masking
    # Sets is_fluid, is_boundary_cell, and is_solid based on state.mask
    enforce_mask_semantics(state)
    create_fluid_mask(state)
    
    # 2. Sparse Operator Construction (Scale Guard Active)
    # These use state.grid and state.constants (dx, dy, dz)
    build_divergence_operator(state)
    build_gradient_operators(state)
    build_laplacian_operators(state)
    
    # 3. Advection & PPE Setup
    # prepare_ppe_structure now utilizes the data loaded in Step 0
    build_advection_structure(state)
    prepare_ppe_structure(state)
    
    # 4. Initialization of Step 2 diagnostics
    compute_initial_health(state)
    
    # Final check: Step 2 doesn't trigger the time loop yet; 
    # it only prepares the tools for Step 3.
    state.ready_for_time_loop = False
    
    return state