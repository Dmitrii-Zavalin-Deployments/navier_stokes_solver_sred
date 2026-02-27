# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Dummy: Mimics the exact output of orchestrate_step1.
    
    Constitutional Role: 
    Provides a State object where only the 'Step 1' safes are filled.
    Accessing state.ppe or state.health from this object WILL trigger 
    a RuntimeError, as they have not been initialized yet.
    """
    state = SolverState()

    # 1. Parsing & Config (from parse_config)
    state.config.case_name = "dummy_verification"
    state.config.method = "jacobi"
    # Note: ppe_tolerance etc. are left as None if not in Step 1 JSON

    # 2. Grid Initialization (from initialize_grid)
    state.grid.nx, state.grid.ny, state.grid.nz = nx, ny, nz
    state.grid.x_min, state.grid.x_max = 0.0, 1.0
    state.grid.y_min, state.grid.y_max = 0.0, 1.0
    state.grid.z_min, state.grid.z_max = 0.0, 1.0

    # 3. Field Allocation & Initial Conditions (from allocate_fields/apply_initial_conditions)
    state.fields.P = np.zeros((nx, ny, nz))
    state.fields.U = np.zeros((nx + 1, ny, nz))
    state.fields.V = np.zeros((nx, ny + 1, nz))
    state.fields.W = np.zeros((nx, ny, nz + 1))

    # 4. Topology (from map_geometry_mask)
    mask_shape = (nx, ny, nz)
    state.masks.mask = np.ones(mask_shape, dtype=int)
    state.masks.is_fluid = np.ones(mask_shape, dtype=bool)
    state.masks.is_boundary = np.zeros(mask_shape, dtype=bool)

    # 5. Physics & Constants (from compute_derived_constants)
    state.fluid.rho = 1000.0
    state.fluid.mu = 0.001
    
    # We populate the 'config' with dt as Step 1 usually extracts it 
    # from simulation_parameters for the time-loop readiness.
    state.config.ppe_tolerance = 1e-6 

    # 6. Global Odometers (Initialized to zero by assemble_simulation_state)
    state.iteration = 0
    state.time = 0.0
    state.ready_for_time_loop = False

    return state