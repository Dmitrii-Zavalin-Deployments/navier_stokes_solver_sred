# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    state = SolverState()

    # 1. Config & Schema-Compliant Boundary Conditions
    state.config.case_name = "dummy_verification"
    state.config.method = "jacobi"
    
    # FIX: Must be an ARRAY of OBJECTS with 'location' and 'type'
    # FIX: 'type' must match the schema enum (e.g., 'no-slip' with a hyphen)
    state.config.boundary_conditions = [
        {"location": "x_min", "type": "no-slip"},
        {"location": "x_max", "type": "outflow"},
        {"location": "y_min", "type": "no-slip"},
        {"location": "y_max", "type": "no-slip"},
        {"location": "z_min", "type": "no-slip"},
        {"location": "z_max", "type": "no-slip"}
    ]
    state.config.constants = {"dt": 0.001, "rho": 1000.0, "mu": 0.001}

    # 2. Grid & Fields
    state.grid.nx, state.grid.ny, state.grid.nz = nx, ny, nz
    state.fields.P = np.zeros((nx, ny, nz))
    state.fields.U = np.zeros((nx + 1, ny, nz))
    state.fields.V = np.zeros((nx, ny + 1, nz))
    state.fields.W = np.zeros((nx, ny, nz + 1))

    # FIX: Initialize 'ext' fields as None or empty (Schema requires them)
    state.fields.P_ext = None
    state.fields.U_ext = None
    state.fields.V_ext = None
    state.fields.W_ext = None

    # 3. Topology
    state.masks.mask = np.ones((nx, ny, nz), dtype=int)

    # 4. Global State
    state.iteration = 0
    state.time = 0.0
    state.ready_for_time_loop = False

    # FIX: Ensure these departments are initialized so to_json_safe sees them
    # Even if empty, they must be 'objects' (dicts) in the final JSON.
    state.health.is_stable = True 
    state.ppe.status = "initialized"

    return state