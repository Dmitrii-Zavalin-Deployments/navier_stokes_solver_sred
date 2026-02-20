# src/step2/orchestrate_step2.py

from __future__ import annotations
from src.solver_state import SolverState

# Internal dependencies for configuration and geometry
from .load_numerical_config import load_numerical_config
from .enforce_mask_semantics import enforce_mask_semantics
from .create_fluid_mask import create_fluid_mask

# Operator builders
from .build_divergence_operator import build_divergence_operator
from .build_gradient_operators import build_gradient_operators
from .build_laplacian_operators import build_laplacian_operators

# Numerical structures
from .build_advection_structure import build_advection_structure
from .prepare_ppe_structure import prepare_ppe_structure
from .compute_initial_health import compute_initial_health

def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Step 2: Operators & PPE Structure.
    
    Transforms the geometric description from Step 1 into computational 
    operators (Sparse Matrices) and prepares the Pressure Poisson Equation.
    
    This orchestration ensures all tools are sharpened and verified 
    before the main time-integration loop in Step 3.
    """
    
    # 0. Configuration Loading
    # Extracts numerical parameters (tolerances, solver types) into state.config
    load_numerical_config(state)
    
    # 1. Geometry & Masking
    # Converts the integer mask into boolean arrays (is_fluid, is_boundary_cell)
    enforce_mask_semantics(state)
    create_fluid_mask(state)
    
    # 2. Sparse Operator Construction
    # Generates the mathematical matrices based on the grid and fluid masks.
    # build_divergence_operator is placed first as it is required by health checks.
    build_divergence_operator(state)
    build_gradient_operators(state)
    build_laplacian_operators(state)
    
    # 3. Advection & PPE Setup
    # Prepares the metadata for (u · ∇)u and the RHS coefficient (rho/dt) for the PPE.
    build_advection_structure(state)
    prepare_ppe_structure(state)
    
    # 4. Initialization of Step 2 diagnostics
    # PRODUCTION PATH: We pass no override, forcing the use of the 
    # newly built sparse divergence operator to check initial field health.
    compute_initial_health(state)
    
    # Final check: Step 2 confirms the tools are ready. 
    # Step 3 will be responsible for flipping this to True during the loop.
    state.ready_for_time_loop = False
    
    return state