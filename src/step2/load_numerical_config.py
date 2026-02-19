# src/step2/load_numerical_config.py
import json
import os
from src.solver_state import SolverState

def load_numerical_config(state: SolverState) -> None:
    """
    Independent parser for Step 2: Loads numerical tuning for linear solvers.
    Ensures Step 2 is self-contained and portable.
    """
    config_path = "config.json"
    
    # Default settings: The 'Safety Net'
    defaults = {
        "solver_type": "PCG",
        "ppe_tolerance": 1e-6,
        "ppe_max_iter": 1000
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                # Pull only the section Step 2 cares about
                state.config['solver_settings'] = data.get('solver_settings', defaults)
        except (json.JSONDecodeError, IOError):
            # If the file is corrupted, use defaults to prevent simulation crash
            state.config['solver_settings'] = defaults
    else:
        # If the file is missing, Step 2 still functions with standard values
        state.config['solver_settings'] = defaults