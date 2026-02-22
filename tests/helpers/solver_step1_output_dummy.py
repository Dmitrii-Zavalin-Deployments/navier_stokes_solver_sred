# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.schema.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Initializes the SolverState with all mandatory departments.
    """
    state = SolverState()
    
    # 1. Grid Department
    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0,
        "dx": 1.0/nx, "dy": 1.0/ny, "dz": 1.0/nz,
        "total_cells": nx * ny * nz
    }

    # 2. Constants Department
    state.constants = {
        "dt": 0.01,
        "nu": 0.001,
        "reynolds_number": 1000.0
    }

    # 3. Fields Department (Initializes with zeros)
    state.fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz))
    }

    # 4. Fluid Properties Department (The Missing Link)
    # Initialized as empty; populated with values in Step 3
    state.fluid_properties = {}

    # 5. PPE & Operators (Foundation for Step 2)
    state.ppe = {"dimension": nx * ny * nz}
    state.operators = {}
    
    # 6. Metadata
    state.time = 0.0
    state.iteration = 0
    state.history = {
        "times": [], 
        "divergence_norms": [], 
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": []
    }
    state.health = {}

    return state