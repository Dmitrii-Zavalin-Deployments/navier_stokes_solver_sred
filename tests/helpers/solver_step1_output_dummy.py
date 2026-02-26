# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Implementation: Canonical Initialization.
    
    Zero-Debt Architecture (Feb 2026):
    - No Shadowing: Redundant dt, rho, mu, dx, dy, dz removed from constants.
    - Physics Truth: kinematic_viscosity is now a derived property (state.nu).
    - Memory Purity: is_fluid is a direct reference to the mask.
    - Departmental Integrity: Data lives exactly where it is defined.
    """
    state = SolverState()

    # 1. Grid Definition (The ONLY home for spatial deltas)
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

    # 2. Fluid Physics (The Primary Physics Home)
    state.fluid_properties = {
        "density": 1000.0,
        "viscosity": 0.001
        # kinematic_viscosity is calculated via state.nu properties
    }

    # 3. Initial Conditions
    state.initial_conditions = {
        "velocity": [0.0, 0.0, 0.0],
        "pressure": 0.0
    }

    # 4. Simulation Parameters (Source of Truth for state.dt)
    state.simulation_parameters = {
        "time_step": 0.001,
        "total_time": 1.0,
        "output_interval": 10
    }

    # 5. Field Allocation (Staggered Grid Layout)
    state.fields = {
        "P": np.zeros((nx, ny, nz)),           
        "U": np.zeros((nx + 1, ny, nz)),       
        "V": np.zeros((nx, ny + 1, nz)),       
        "W": np.zeros((nx, ny, nz + 1)),       
    }

    # 6. Masking (Flattened 1D - Linked Source)
    size = state.grid["total_cells"]
    # We create the list once and link is_fluid to it to avoid memory drift
    ones_mask = [1] * size
    zeros_mask = [0] * size
    
    state.mask = ones_mask
    state.is_fluid = ones_mask
    state.is_solid = zeros_mask
    state.is_boundary_cell = zeros_mask

    # 7. Global Constants (Non-departmental only)
    # rho, mu, dt, dx, dy, dz are REMOVED to prevent shadowing
    state.constants = {
        "g": 9.81
    }
    
    # 8. Boundary Conditions
    velocity_only = {"u": 0.0, "v": 0.0, "w": 0.0}
    state.boundary_conditions = [
        {"location": loc, "type": "no-slip", "values": velocity_only.copy()}
        for loc in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    ]

    # 9. Global Health, History, and Metadata
    state.config = {
        "solver_type": "projection", 
        "precision": "float64",
        "case_name": "dummy_verification"
    }
    state.ppe = {"dimension": size}
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