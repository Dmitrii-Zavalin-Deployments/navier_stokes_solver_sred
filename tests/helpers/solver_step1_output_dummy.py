# tests/helpers/solver_step1_output_dummy.py

import numpy as np

class MockState:
    """
    Decoupled State Container.
    Definition is local to the helper to prevent ImportErrors during 
    CI collection while the src/ directory is under construction.
    """
    def __init__(self):
        self.grid = {}
        self.constants = {}
        self.fields = {}
        self.fluid_properties = {}  # Department for Step 3+ Physics Invariants
        self.ppe = {}
        self.operators = {}
        self.intermediate_fields = {}
        self.mask = None
        self.is_fluid = None
        self.is_solid = None
        self.is_boundary_cell = None
        self.history = {}
        self.health = {}
        self.time = 0.0
        self.iteration = 0
        self.ready_for_time_loop = False

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Dummy: The Physical Foundation.
    Follows the 'No Aliasing' rule:
    - Geometry metrics (dx, nx, bounds) live ONLY in .grid
    - Physics constants (rho, mu, dt) live ONLY in .constants
    - Invariants (density) live ONLY in .fluid_properties
    """
    # Initialize the Decoupled 'Empty Slate'
    state = MockState()

    # --- Step 1 Responsibility: Geometry Department ---
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    z_min, z_max = 0.0, 1.0

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "total_cells": nx * ny * nz
    }

    # --- Step 1 Responsibility: Constants (Pipeline Parameters) ---
    state.constants = {
        "nu": 0.001,
        "dt": 0.01,
        "reynolds_number": 1000.0
    }

    # --- Step 1 Responsibility: Fluid Properties (The Invariant Gate) ---
    # Initialized as empty in Step 1; populated with strictly positive 
    # density values in Step 3 to support projection scaling.
    state.fluid_properties = {}

    # --- Step 1 Responsibility: Basic Staggered Fields ---
    # 
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # --- Step 1 Responsibility: PPE Department Plan ---
    state.ppe = {
        "dimension": nx * ny * nz
    }

    # --- Step 1 Responsibility: Basic Masking ---
    state.mask = np.ones((nx, ny, nz), dtype=int)
    state.is_fluid = (state.mask == 1)
    state.is_solid = ~state.is_fluid
    state.is_boundary_cell = np.zeros((nx, ny, nz), dtype=bool)

    # Initialize history to avoid Step 3 attribute errors
    state.history = {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": [],
    }
    
    state.health = {}

    return state