# tests/helpers/step3_schema_dummy_state.py

import numpy as np


class Step3SchemaDummyState(dict):
    """
    Fully schema‑compliant Step‑3 output dummy.
    Matches the Step 3 Output Schema exactly.
    """

    def __init__(self, nx, ny, nz):
        super().__init__()

        # -----------------------------
        # config (required)
        # -----------------------------
        self["config"] = {
            "simulation": {"dt": 0.1},
            "fluid": {"density": 1.0, "viscosity": 0.1},
            "boundary_conditions": [],
            "forces": {"force_vector": [0, 0, 0], "units": "N"},
            "geometry_definition": {
                "geometry_mask_flat": [1] * (nx * ny * nz),
                "geometry_mask_shape": [nx, ny, nz],
                "mask_encoding": {"fluid": 1, "solid": -1},
                "flattening_order": "C",
            },
            "domain": {
                "x_min": 0.0, "x_max": nx * 1.0,
                "y_min": 0.0, "y_max": ny * 1.0,
                "z_min": 0.0, "z_max": nz * 1.0,
                "nx": nx, "ny": ny, "nz": nz,
            },
        }

        # -----------------------------
        # mask (required)
        # -----------------------------
        mask = np.ones((nx, ny, nz), dtype=int)
        self["mask"] = mask

        # -----------------------------
        # is_fluid (required)
        # -----------------------------
        self["is_fluid"] = (mask == 1)

        # -----------------------------
        # is_boundary_cell (required)
        # -----------------------------
        self["is_boundary_cell"] = np.zeros((nx, ny, nz), dtype=bool)

        # -----------------------------
        # fields (required)
        # -----------------------------
        self["fields"] = {
            "P": np.zeros((nx, ny, nz)),
            "U": np.zeros((nx + 1, ny, nz)),
            "V": np.zeros((nx, ny + 1, nz)),
            "W": np.zeros((nx, ny, nz + 1)),
        }

        # -----------------------------
        # bcs (required)
        # -----------------------------
        self["bcs"] = []  # normalized BC table

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
        # operators (required)
        # Step‑3 schema requires OBJECTS, not callables.
        # -----------------------------
        self["operators"] = {
            "divergence": {},
            "gradient_p_x": {},
            "gradient_p_y": {},
            "gradient_p_z": {},
            "laplacian_u": {},
            "laplacian_v": {},
            "laplacian_w": {},
            "advection_u": {},
            "advection_v": {},
            "advection_w": {},
        }

        # -----------------------------
        # ppe (required)
        # -----------------------------
        self["ppe"] = {
            "rhs_builder": {},
            "solver_type": "PCG",
            "tolerance": 1e-6,
            "max_iterations": 100,
            "ppe_is_singular": False,
            # ppe_converged is optional
        }

        # -----------------------------
        # health (required)
        # -----------------------------
        self["health"] = {
            "post_correction_divergence_norm": 0.0,
            "max_velocity_magnitude": 0.0,
            "cfl_advection_estimate": 0.0,
        }

        # -----------------------------
        # history (required)
        # -----------------------------
        self["history"] = {
            "times": [],
            "divergence_norms": [],
            "max_velocity_history": [],
            "ppe_iterations_history": [],
            "energy_history": [],
        }

        # -----------------------------
        # advection_meta (optional)
        # -----------------------------
        self["advection_meta"] = None
