# tests/helpers/step1_schema_dummy_state.py

import numpy as np

class Step1DummyConfig:
    def __init__(self):
        self.domain = {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 2, "ny": 2, "nz": 2
        }
        self.fluid = {"density": 1.0, "viscosity": 0.1}
        self.simulation = {"time_step": 0.1, "total_time": 1.0, "output_interval": 1}
        self.forces = {"force_vector": [0, 0, 0], "units": "N"}
        self.boundary_conditions = [
            {
                "role": "wall",
                "type": "dirichlet",
                "faces": ["x_min"],
                "apply_to": ["velocity"],
                "velocity": [0, 0, 0],
                "pressure": 0.0,
                "pressure_gradient": 0.0,
                "no_slip": True
            }
        ]
        self.geometry_definition = {
            "geometry_mask_flat": [1, 1, 1, 1, 1, 1, 1, 1],
            "geometry_mask_shape": [2, 2, 2],
            "mask_encoding": {"fluid": 1, "solid": -1},
            "flattening_order": "C"
        }


class Step1DummyGrid:
    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = 1.0
        self.dy = 1.0
        self.dz = 1.0
        self.x_min = 0.0
        self.y_min = 0.0
        self.z_min = 0.0
        self.x_max = nx * 1.0
        self.y_max = ny * 1.0
        self.z_max = nz * 1.0


class Step1DummyFields:
    def __init__(self, nx, ny, nz):
        self.P = np.zeros((nx, ny, nz))
        self.U = np.zeros((nx, ny, nz))
        self.V = np.zeros((nx, ny, nz))
        self.W = np.zeros((nx, ny, nz))
        self.Mask = np.ones((nx, ny, nz), dtype=int)


class Step1DummyConstants:
    def __init__(self):
        self.rho = 1.0
        self.mu = 0.1
        self.dt = 0.1
        self.dx = 1.0
        self.dy = 1.0
        self.dz = 1.0
        self.inv_dx = 1.0
        self.inv_dy = 1.0
        self.inv_dz = 1.0
        self.inv_dx2 = 1.0
        self.inv_dy2 = 1.0
        self.inv_dz2 = 1.0


class Step1SchemaDummyState:
    def __init__(self, nx, ny, nz):
        self.config = Step1DummyConfig()
        self.grid = Step1DummyGrid(nx, ny, nz)
        self.fields = Step1DummyFields(nx, ny, nz)
        self.mask_3d = np.ones((nx, ny, nz), dtype=int)
        self.boundary_table = {
            "x_min": [],
            "x_max": [],
            "y_min": [],
            "y_max": [],
            "z_min": [],
            "z_max": []
        }
        self.constants = Step1DummyConstants()
