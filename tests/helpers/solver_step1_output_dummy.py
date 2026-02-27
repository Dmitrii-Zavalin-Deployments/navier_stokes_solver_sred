# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    state = SolverState()

    # 1. Config & Schema-Compliant Boundary Conditions
    state.config.case_name = "dummy_verification"
    state.config.method = "jacobi"
    
    state.config.boundary_definitions = [
        {"location": "x_min", "type": "no-slip"},
        {"location": "x_max", "type": "outflow"},
        {"location": "y_min", "type": "no-slip"},
        {"location": "y_max", "type": "no-slip"},
        {"location": "z_min", "type": "no-slip"},
        {"location": "z_max", "type": "no-slip"}
    ]
    state.config.simulation_parameters = {"time_step": 0.001, "total_time": 1.0, "output_interval": 1}

    # 2. Grid Initialization (UNLOCKED)
    state.grid.nx, state.grid.ny, state.grid.nz = nx, ny, nz
    state.grid.x_min, state.grid.x_max = 0.0, 1.0
    state.grid.y_min, state.grid.y_max = 0.0, 1.0
    state.grid.z_min, state.grid.z_max = 0.0, 1.0

    # 3. Field Allocation
    state.fields.P = np.zeros((nx, ny, nz))
    state.fields.U = np.zeros((nx + 1, ny, nz))
    state.fields.V = np.zeros((nx, ny + 1, nz))
    state.fields.W = np.zeros((nx, ny, nz + 1))

    state.fields.P_ext = None
    state.fields.U_ext = None
    state.fields.V_ext = None
    state.fields.W_ext = None

    # 4. Topology
    state.masks.mask = np.ones((nx, ny, nz), dtype=int)

    # 5. Global State
    state.iteration = 0
    state.time = 0.0
    state.ready_for_time_loop = False

    state.health.is_stable = True 
    state.ppe.status = "initialized"

    return state
