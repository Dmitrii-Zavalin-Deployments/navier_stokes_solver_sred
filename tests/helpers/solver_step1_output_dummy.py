# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Implementation: Initialization & Allocation.
    
    Ensures the state is fully populated with Step 1 data, 
    satisfying both the JSON Schema contract and physics property tests.
    """
    state = SolverState()

    # 1. Grid Definition
    # Dynamically calculated based on input resolution to keep Theory tests passing.
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    z_min, z_max = 0.0, 1.0
    
    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": (x_max - x_min) / nx,
        "dy": (y_max - y_min) / ny,
        "dz": (z_max - z_min) / nz,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "total_cells": nx * ny * nz
    }

    # 2. Fluid Physics (Step 1 Department)
    # This prevents the AttributeError in property integrity tests.
    state.fluid_properties = {
        "rho": 1000.0,      # Density
        "mu": 0.001,        # Dynamic Viscosity
        "nu": 0.000001      # Kinematic Viscosity
    }

    # 3. Field Allocation (Staggered Grid Layout)
    state.fields = {
        "P": np.zeros((nx, ny, nz)),           # Cell-centered
        "U": np.zeros((nx + 1, ny, nz)),       # X-faces
        "V": np.zeros((nx, ny + 1, nz)),       # Y-faces
        "W": np.zeros((nx, ny, nz + 1)),       # Z-faces
    }

    # 4. Masking (Type Compliance)
    # Converts NumPy arrays to nested lists to satisfy Schema 'array' type.
    mask_shape = (nx, ny, nz)
    fluid_mask_arr = np.ones(mask_shape, dtype=int)
    
    state.mask = fluid_mask_arr.tolist()
    state.is_fluid = fluid_mask_arr.tolist()
    state.is_solid = np.zeros(mask_shape, dtype=int).tolist()
    state.is_boundary_cell = np.zeros(mask_shape, dtype=int).tolist()

    # 5. Department Initialization & PPE Intent
    # PPE 'dimension' is required for lifecycle allocation tests.
    state.ppe = {"dimension": nx * ny * nz}
    
    state.config = {
        "solver_type": "projection", 
        "precision": "float64",
        "case_name": "dummy_verification"
    }
    
    state.constants = {
        "nu": 0.01, 
        "dt": 0.001,
        "g": 9.81
    }
    
    state.boundary_conditions = {
        "west": "noslip",
        "east": "noslip",
        "north": "moving_wall",
        "south": "noslip",
        "top": "noslip",
        "bottom": "noslip"
    }

    # 6. Global Health & History
    state.health = {"status": "initialized", "errors": []}
    state.iteration = 0
    state.time = 0.0

    return state