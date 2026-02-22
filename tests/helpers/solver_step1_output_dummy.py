# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Implementation: Initialization & Allocation.
    
    Updated Feb 2026:
    - Standardized fluid_properties keys (density, viscosity).
    - Added initial_conditions department for lifecycle tracking.
    - Ensured mask types remain JSON-safe lists.
    """
    state = SolverState()

    # 1. Grid Definition
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
    # Keys standardized for test_physics_fluid_constants.py
    state.fluid_properties = {
        "density": 1000.0,
        "viscosity": 0.001,
        "kinematic_viscosity": 1e-6
    }

    # 3. Initial Conditions (NEW: Property Tracking Matrix Element)
    # This represents the user's intent: u(t=0) = u0
    state.initial_conditions = {
        "velocity": [0.0, 0.0, 0.0],
        "pressure": 0.0
    }

    # 4. Field Allocation (Staggered Grid Layout)
    # Staggering: Faces (nx+1) for primary direction vectors
    state.fields = {
        "P": np.zeros((nx, ny, nz)),           
        "U": np.zeros((nx + 1, ny, nz)),       
        "V": np.zeros((nx, ny + 1, nz)),       
        "W": np.zeros((nx, ny, nz + 1)),       
    }

    # 5. Masking (Type Compliance)
    mask_shape = (nx, ny, nz)
    fluid_mask_arr = np.ones(mask_shape, dtype=int)
    
    state.mask = fluid_mask_arr.tolist()
    state.is_fluid = fluid_mask_arr.tolist()
    state.is_solid = np.zeros(mask_shape, dtype=int).tolist()
    state.is_boundary_cell = np.zeros(mask_shape, dtype=int).tolist()

    # 6. Department Initialization & PPE Intent
    state.ppe = {"dimension": nx * ny * nz}
    
    state.config = {
        "solver_type": "projection", 
        "precision": "float64",
        "case_name": "dummy_verification"
    }
    
    state.constants = {
        "dt": 0.001,
        "g": 9.81
    }
    
    state.boundary_conditions = {
        "west": "noslip", "east": "noslip",
        "north": "moving_wall", "south": "noslip",
        "top": "noslip", "bottom": "noslip"
    }

    # 7. Global Health & History
    state.health = {"status": "initialized", "errors": []}
    state.history = {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": []
    }
    state.iteration = 0
    state.time = 0.0

    return state