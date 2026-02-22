# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Implementation: Initialization & Allocation.
    
    Updated Feb 2026:
    - Maintained all Health/History departments to prevent test regressions.
    - Explicitly mapped simulation_parameters.output_interval for Archivist logic.
    - Added 'values' sub-dictionary to boundary_conditions for Step 2/3 parity.
    - Physical Logic Fix: Removed 'p' from no-slip values to satisfy src/step1 parser.
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

    # 2. Fluid Physics
    state.fluid_properties = {
        "density": 1000.0,
        "viscosity": 0.001,
        "kinematic_viscosity": 1e-6
    }

    # 3. Initial Conditions
    state.initial_conditions = {
        "velocity": [0.0, 0.0, 0.0],
        "pressure": 0.0
    }

    # 4. Simulation Parameters (Source of Truth for the Archivist)
    dt_val = 0.001
    interval_val = 10
    state.simulation_parameters = {
        "time_step": dt_val,
        "total_time": 1.0,
        "output_interval": interval_val
    }

    # 5. Field Allocation (Staggered Grid Layout)
    state.fields = {
        "P": np.zeros((nx, ny, nz)),           
        "U": np.zeros((nx + 1, ny, nz)),       
        "V": np.zeros((nx, ny + 1, nz)),       
        "W": np.zeros((nx, ny, nz + 1)),       
    }

    # 6. Masking (Type Compliance)
    mask_shape = (nx, ny, nz)
    fluid_mask_arr = np.ones(mask_shape, dtype=int)
    
    state.mask = fluid_mask_arr.tolist()
    state.is_fluid = fluid_mask_arr.tolist()
    state.is_solid = np.zeros(mask_shape, dtype=int).tolist()
    state.is_boundary_cell = np.zeros(mask_shape, dtype=int).tolist()

    # 7. Department Initialization & PPE Intent
    state.ppe = {"dimension": nx * ny * nz}
    
    state.config = {
        "solver_type": "projection", 
        "precision": "float64",
        "case_name": "dummy_verification"
    }
    
    # 8. External Forces (Physical Intent - Established in Step 1)
    state.external_forces = {
        "force_vector": [0.0, 0.0, -9.81],
        "type": "constant_acceleration"
    }

    # Internal math constants (Synced with simulation_parameters)
    state.constants = {
        "dt": state.simulation_parameters["time_step"], 
        "g": 9.81
    }
    
    # Boundary Conditions (Updated with 'values' for Step 2/3 numerical roles)
    # Physical Logic: 'no-slip' allows velocity (u,v,w) but forbids pressure (p).
    velocity_only = {"u": 0.0, "v": 0.0, "w": 0.0}
    state.boundary_conditions = [
        {"location": "x_min", "type": "no-slip", "values": velocity_only.copy()},
        {"location": "x_max", "type": "no-slip", "values": velocity_only.copy()},
        {"location": "y_min", "type": "no-slip", "values": velocity_only.copy()},
        {"location": "y_max", "type": "no-slip", "values": velocity_only.copy()},
        {"location": "z_min", "type": "no-slip", "values": velocity_only.copy()},
        {"location": "z_max", "type": "no-slip", "values": velocity_only.copy()}
    ]

    # 9. Global Health & History (CRITICAL: Do not delete, used by Step 2-5)
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