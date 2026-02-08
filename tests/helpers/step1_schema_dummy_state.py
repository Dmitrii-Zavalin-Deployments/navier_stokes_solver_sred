# tests/helpers/step1_schema_dummy_state.py

import numpy as np


class Step1SchemaDummyState(dict):
    """
    Fully schema‑compliant Step‑1 output dummy.
    Matches the Step 1 Output Schema exactly.
    """

    def __init__(self, nx, ny, nz):
        super().__init__()

        # -----------------------------
        # grid (required)
        # -----------------------------
        self["grid"] = {
            "x_min": 0.0,
            "x_max": nx * 1.0,
            "y_min": 0.0,
            "y_max": ny * 1.0,
            "z_min": 0.0,
            "z_max": nz * 1.0,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
        }

        # -----------------------------
        # fields (required)
        # -----------------------------
        self["fields"] = {
            "P": np.zeros((nx, ny, nz)),
            "U": np.zeros((nx, ny, nz)),
            "V": np.zeros((nx, ny, nz)),
            "W": np.zeros((nx, ny, nz)),
            "Mask": np.ones((nx, ny, nz), dtype=int),  # values ∈ {-1,0,1}
        }

        # -----------------------------
        # mask_3d (required)
        # -----------------------------
        self["mask_3d"] = np.ones((nx, ny, nz), dtype=int)

        # -----------------------------
        # boundary_table (required)
        # -----------------------------
        self["boundary_table"] = {
            "x_min": [],
            "x_max": [],
            "y_min": [],
            "y_max": [],
            "z_min": [],
            "z_max": [],
        }

        # -----------------------------
        # constants (required)
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
        # config (required)
        # -----------------------------
        self["config"] = {
            "domain": {
                "x_min": 0.0, "x_max": 1.0,
                "y_min": 0.0, "y_max": 1.0,
                "z_min": 0.0, "z_max": 1.0,
                "nx": nx, "ny": ny, "nz": nz,
            },
            "fluid": {
                "density": 1.0,
                "viscosity": 0.1,
            },
            "simulation": {
                "time_step": 0.1,
                "total_time": 1.0,
                "output_interval": 1,
            },
            "forces": {
                "force_vector": [0.0, 0.0, 0.0],
                "units": "N",
            },
            "boundary_conditions": [
                {
                    "role": "wall",
                    "type": "dirichlet",
                    "faces": ["x_min"],
                    "apply_to": ["velocity", "pressure"],
                    "velocity": [0.0, 0.0, 0.0],
                    "pressure": 0.0,
                    "pressure_gradient": 0.0,
                    "no_slip": True,
                }
            ],
            "geometry_definition": {
                "geometry_mask_flat": [1] * (nx * ny * nz),
                "geometry_mask_shape": [nx, ny, nz],
                "mask_encoding": {"fluid": 1, "solid": -1},
                "flattening_order": "C",
            },
        }
