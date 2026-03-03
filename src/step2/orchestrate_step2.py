# src/step2/orchestrate_step2.py

import json
from src.solver_state import SolverState
from .operators import build_numerical_operators
from .advection import build_advection_stencils

def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Step 2 Orchestrator: Mathematical Readiness.
    """
    
    # --- Point 2: Calculate & Prepare ---
    try:
        with open("config.json", "r") as f:
            external_config = json.load(f)
        
        settings = external_config["solver_settings"]
        
        # MANDATORY TRIAD: All three must be initialized to satisfy _get_safe
        state.config.ppe_atol = settings["ppe_atol"]
        state.config.ppe_tolerance = settings["ppe_tolerance"] # <--- ADD THIS LINE
        state.config.ppe_max_iter = settings["ppe_max_iter"]
        
    except FileNotFoundError:
        raise FileNotFoundError("Critical Error: 'config.json' not found.")
    except KeyError as e:
        raise KeyError(f"Critical Error: Missing required solver setting {e} in 'config.json'.")
    except json.JSONDecodeError:
        raise ValueError("Critical Error: 'config.json' is not a valid JSON file.")

    # Delegate math to worker files
    build_numerical_operators(state)
    build_advection_stencils(state)

    # --- Point 3: Insertion & State Baseline ---
    state.ppe._A = state.operators.laplacian
    state.ppe.preconditioner = None 

    # Initialize Health Vitals
    state.health.max_u = 0.0
    state.health.divergence_norm = 0.0
    state.health.is_stable = True
    state.health.post_correction_divergence_norm = 0.0

    state.ready_for_time_loop = True 
    
    return state