# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Dummy: Mimics the exact output of orchestrate_step1.
    
    Constitutional Role: 
    Provides a State object where only the 'Step 1' safes are filled.
    Satisfies the 'solver_output_schema.json' contract by populating 
    required legacy fields in the config safe.
    """
    state = SolverState()

    # ------------------------------------------------------------------
    # 1. Parsing & Config (from parse_config)
    # ------------------------------------------------------------------
    state.config.case_name = "dummy_verification"
    state.config.method = "jacobi"
    state.config.precision = "float64"
    
    # CONTRACT REQUIREMENT: Schema expects 'constants' and 'boundary_conditions'
    # even if they are empty at this stage.
    state.config.constants = {
        "dt": 0.001,
        "rho": 1000.0,
        "mu": 0.001
    }
    state.config.boundary_conditions = {
        "x_min": "noslip",
        "x_max": "outflow",
        "y_min": "noslip",
        "y_max": "noslip",
        "z_min": "noslip",
        "z_max": "noslip"
    }

    # ------------------------------------------------------------------
    # 2. Grid Initialization (from initialize_grid)
    # ------------------------------------------------------------------
    state.grid.nx, state.grid.ny, state.grid.nz = nx, ny, nz
    state.grid.x_min, state.grid.x_max = 0.0, 1.0
    state.grid.y_min, state.grid.y_max = 0.0, 1.0
    state.grid.z_min, state.grid.z_max = 0.0, 1.0

    # ------------------------------------------------------------------
    # 3. Field Allocation (from allocate_fields)
    # ------------------------------------------------------------------
    # These base fields are required by the root of the legacy schema
    state.fields.P = np.zeros((nx, ny, nz))
    state.fields.U = np.zeros((nx + 1, ny, nz))
    state.fields.V = np.zeros((nx, ny + 1, nz))
    state.fields.W = np.zeros((nx, ny, nz + 1))

    # ------------------------------------------------------------------
    # 4. Topology (from map_geometry_mask)
    # ------------------------------------------------------------------
    mask_shape = (nx, ny, nz)
    # CONTRACT REQUIREMENT: The schema expects 'mask' at the root level.
    # We populate it in the 'masks' safe, and to_json_safe will lift it up.
    state.masks.mask = np.ones(mask_shape, dtype=int)
    state.masks.is_fluid = np.ones(mask_shape, dtype=bool)
    state.masks.is_boundary = np.zeros(mask_shape, dtype=bool)

    # ------------------------------------------------------------------
    # 5. Global Odometers
    # ------------------------------------------------------------------
    state.iteration = 0
    state.time = 0.0
    state.ready_for_time_loop = False
    
    # Solver settings required for PPE setup later
    state.config.ppe_tolerance = 1e-6 
    state.config.ppe_max_iter = 1000

    return state