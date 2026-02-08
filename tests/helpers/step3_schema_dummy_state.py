# tests/helpers/step3_schema_dummy_state.py

import numpy as np


class Step3SchemaDummyState(dict):
    """
    Dummy state for Step 3 tests.
    Must satisfy the Step 2 output schema.
    """

    def __init__(self, nx, ny, nz):
        super().__init__()

        # -----------------------------
        # Grid (required by Step 2 schema)
        # -----------------------------
        self["grid"] = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "x_min": 0.0,
            "x_max": nx * 1.0,
            "y_min": 0.0,
            "y_max": ny * 1.0,
            "z_min": 0.0,
            "z_max": nz * 1.0,
        }

        # -----------------------------
        # Config (required)
        # -----------------------------
        self["config"] = {
            "simulation": {"dt": 0.1},
            "fluid": {"density": 1.0, "viscosity": 0.1},
            "boundary_conditions": [],
        }

        # -----------------------------
        # Fields (required)
        # -----------------------------
        self["fields"] = {
            "P": np.zeros((nx, ny, nz)),
            "U": np.zeros((nx + 1, ny, nz)),
            "V": np.zeros((nx, ny + 1, nz)),
            "W": np.zeros((nx, ny, nz + 1)),
            "Mask": np.ones((nx, ny, nz), dtype=int),
        }

        # -----------------------------
        # Constants (required)
        # -----------------------------
        self["constants"] = {
            "rho": 1.0,
            "mu": 0.1,
            "dt": 0.1,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "inv_dx": 1.0,
            "inv_dy": 1.0,
            "inv_dz": 1.0,
            "inv_dx2": 1.0,
            "inv_dy2": 1.0,
            "inv_dz2": 1.0,
        }

        # -----------------------------
        # Operators (required)
        # Step 3 only checks presence, not correctness
        # -----------------------------
        self["operators"] = {
            "divergence": lambda U, V, W: np.zeros((nx, ny, nz)),
            "gradient_p": lambda P: (np.zeros_like(self["fields"]["U"]),
                                     np.zeros_like(self["fields"]["V"]),
                                     np.zeros_like(self["fields"]["W"])),
        }

        # -----------------------------
        # PPE structure (required)
        # -----------------------------
        self["ppe"] = {
            "rhs_builder": lambda div: np.zeros((nx, ny, nz)),
            "solver_type": "PCG",
            "tolerance": 1e-6,
            "max_iterations": 1000,
            "ppe_is_singular": True,
        }

        # -----------------------------
        # Health (required)
        # -----------------------------
        self["health"] = {
            "divergence_norm": 0.0,
            "cfl_number": 0.0,
        }
