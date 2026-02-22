# tests/helpers/solver_step1_output_dummy.py

import numpy as np

class MockState:
    """
    Decoupled State Container.
    Maintains departmental isolation while satisfying JSON schema requirements.
    """
    def __init__(self):
        self.grid = {}
        self.constants = {}
        self.fields = {}
        self.fluid_properties = {}  # Added to fix AttributeError
        self.config = {}
        self.mask = None
        self.boundary_conditions = {}
        self.ppe = {}
        self.history = {}
        self.health = {}
        self.U_ext = None
        self.V_ext = None
        self.W_ext = None
        self.P_ext = None
        self.time = 0.0
        self.iteration = 0
        self.ready_for_time_loop = False

    def to_json_safe(self) -> dict:
        """Type-casting bridge for JSON Schema validation."""
        def serialize(obj):
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        return {
            "grid": serialize(self.grid),
            "constants": serialize(self.constants),
            "fields": serialize(self.fields),
            "fluid_properties": serialize(self.fluid_properties), # Added to map
            "config": serialize(self.config),
            "mask": serialize(self.mask),
            "boundary_conditions": serialize(self.boundary_conditions),
            "ppe": serialize(self.ppe),
            "history": serialize(self.history),
            "health": serialize(self.health),
            "U_ext": serialize(self.U_ext),
            "V_ext": serialize(self.V_ext),
            "W_ext": serialize(self.W_ext),
            "P_ext": serialize(self.P_ext),
            "time": float(self.time),
            "iteration": int(self.iteration)
        }

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    state = MockState()

    # --- GEOMETRY THEORY CONFORMANCE ---
    # Bounds defined first to satisfy Δx = (max - min) / N
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    z_min, z_max = 0.0, 1.0
    
    dx, dy, dz = x_max/nx, y_max/ny, z_max/nz

    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "total_cells": nx * ny * nz
    }

    # --- DEPARTMENTAL DATA ---
    state.constants = {"nu": 0.001, "dt": 0.01}
    state.fluid_properties = {"density": 1.0, "viscosity": 0.001}
    state.config = {"solver_type": "projection", "precision": "float64"}
    state.boundary_conditions = {"type": "lid_driven_cavity"}
    
    # Primary Staggered Fields
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }
    
    # JSON-Safe Mask (List instead of Array)
    state.mask = np.ones((nx, ny, nz)).tolist()
    
    # Extended Fields (Ghost Nodes)
    state.U_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.V_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.W_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.P_ext = np.zeros((nx + 2, ny + 2, nz + 2))

    state.ppe = {"dimension": nx * ny * nz}
    state.history = {"times": [], "divergence_norms": []}
    state.health = {"status": "initialized"}

    return state