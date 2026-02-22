# tests/helpers/solver_step1_output_dummy.py

import numpy as np

class MockState:
    """
    Decoupled State Container with Serialization Bridge.
    Restored with to_json_safe() to satisfy external contract tests.
    """
    def __init__(self):
        self.grid = {}
        self.constants = {}
        self.fields = {}
        self.fluid_properties = {} 
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

    def to_json_safe(self) -> dict:
        """
        Recursively converts the state into a JSON-serializable dictionary.
        Specifically handles the conversion of NumPy types.
        """
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

        # Capture all departmental attributes
        data = {
            "grid": serialize(self.grid),
            "constants": serialize(self.constants),
            "fields": serialize(self.fields),
            "fluid_properties": serialize(self.fluid_properties),
            "ppe": serialize(self.ppe),
            "history": serialize(self.history),
            "health": serialize(self.health),
            "time": float(self.time),
            "iteration": int(self.iteration),
            "ready_for_time_loop": bool(self.ready_for_time_loop)
        }
        return data

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Factory remains unchanged, but now produces a 'MockState' 
    capable of serialization.
    """
    state = MockState()

    # Geometry
    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": 1.0/nx, "dy": 1.0/ny, "dz": 1.0/nz,
        "total_cells": nx * ny * nz
    }

    # Constants & Fluid Properties
    state.constants = {"nu": 0.001, "dt": 0.01}
    state.fluid_properties = {} 

    # Staggered Fields
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    state.ppe = {"dimension": nx * ny * nz}
    state.history = {
        "times": [], "divergence_norms": [], "max_velocity_history": [],
        "ppe_iterations_history": [], "energy_history": [],
    }
    state.health = {}

    return state