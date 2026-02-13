# tests/helpers/step2_schema_dummy_state.py

import numpy as np


class Step2SchemaDummyState(dict):
    """
    Fully schema‑compliant Step‑1 output dummy.
    Used as input to Step‑2 orchestrator.
    """

    def __init__(
        self,
        nx,
        ny,
        nz,
        *,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        dt=0.1,
        rho=1.0,
        mu=0.1,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # grid (required)
        # ------------------------------------------------------------
        self["grid"] = {
            "x_min": 0.0,
            "x_max": nx * dx,
            "y_min": 0.0,
            "y_max": ny * dy,
            "z_min": 0.0,
            "z_max": nz * dz,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
        }

        # ------------------------------------------------------------
        # mask_3d (required by Step‑2)
        # ------------------------------------------------------------
        mask = np.ones((nx, ny, nz), dtype=int)
        self["mask_3d"] = mask.tolist()

        # ------------------------------------------------------------
        # fields (required by Step‑1 schema)
        # ------------------------------------------------------------
        self["fields"] = {
            "P": np.zeros((nx, ny, nz)),
            "U": np.zeros((nx + 1, ny, nz)),
            "V": np.zeros((nx, ny + 1, nz)),
            "W": np.zeros((nx, ny, nz + 1)),
            "Mask": mask.tolist(),
        }

        # ------------------------------------------------------------
        # boundary_table (required)
        # ------------------------------------------------------------
        self["boundary_table"] = {
            "x_min": [],
            "x_max": [],
            "y_min": [],
            "y_max": [],
            "z_min": [],
            "z_max": [],
        }

        # ------------------------------------------------------------
        # constants (required)
        # ------------------------------------------------------------
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy
        inv_dz = 1.0 / dz

        self["constants"] = {
            "rho": float(rho),
            "mu": float(mu),
            "dt": float(dt),
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            "inv_dx": inv_dx,
            "inv_dy": inv_dy,
            "inv_dz": inv_dz,
            "inv_dx2": inv_dx * inv_dx,
            "inv_dy2": inv_dy * inv_dy,
            "inv_dz2": inv_dz * inv_dz,
        }

        # ------------------------------------------------------------
        # config (required)
        # ------------------------------------------------------------
        self["config"] = {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
            "fluid": {"density": rho, "viscosity": mu},
            "simulation": {"time_step": dt},
            "forces": {"force_vector": [0.0, 0.0, 0.0]},
            "boundary_conditions": [],
            "geometry_definition": {
                "geometry_mask_flat": mask.flatten().tolist(),
                "geometry_mask_shape": [nx, ny, nz],
                "mask_encoding": {"fluid": 1, "solid": -1},
                "flattening_order": "C",
            },
        }

        # ------------------------------------------------------------
        # state_as_dict (required)
        # ------------------------------------------------------------
        self["state_as_dict"] = {}
