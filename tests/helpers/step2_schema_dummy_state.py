# tests/helpers/step2_schema_dummy_state.py

import numpy as np


class Step2SchemaDummyState(dict):
    """
    A Step‑2 test fixture that mimics the *structure* of Step‑1 output
    but uses NumPy arrays for all numerical fields, because Step‑2 tests
    and operators require NumPy semantics (.shape, slicing, broadcasting).

    Structure = Step‑1
    Types     = Step‑2
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
        mask=None,
        boundary_table=None,
        scheme="upwind",
    ):
        super().__init__()

        # -----------------------------
        # Grid block (Step‑1 structure)
        # -----------------------------
        self["grid"] = {
            "x_min": 0.0,
            "x_max": dx * nx,
            "y_min": 0.0,
            "y_max": dy * ny,
            "z_min": 0.0,
            "z_max": dz * nz,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
        }

        # -----------------------------
        # Config block (Step‑1 structure)
        # -----------------------------
        self["config"] = {
            "boundary_conditions": [],
            "domain": {},
            "fluid": {
                "density": rho,
                "viscosity": mu,
            },
            "forces": {},
            "geometry_definition": {},
            "simulation": {
                "dt": dt,
                "advection_scheme": scheme,
            },
        }

        # -----------------------------
        # Mask (NumPy array for Step‑2)
        # -----------------------------
        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=int)

        # -----------------------------
        # Fields block
        # Step‑1 structure, Step‑2 types (NumPy arrays)
        # -----------------------------
        self["fields"] = {
            "P": np.zeros((nx, ny, nz), float),
            "U": np.zeros((nx + 1, ny, nz), float),
            "V": np.zeros((nx, ny + 1, nz), float),
            "W": np.zeros((nx, ny, nz + 1), float),
            "Mask": mask,
        }

        # JSON‑friendly version of mask (Step‑1 compatibility)
        self["mask_3d"] = mask.tolist()

        # -----------------------------
        # Boundary table (Step‑1 structure)
        # Must be an object, not a list
        # -----------------------------
        self["boundary_table"] = (
            boundary_table
            if boundary_table is not None
            else {
                "x_min": [],
                "x_max": [],
                "y_min": [],
                "y_max": [],
                "z_min": [],
                "z_max": [],
            }
        )

        # -----------------------------
        # Constants block (Step‑2 will overwrite)
        # -----------------------------
        self["constants"] = {
            "rho": rho,
            "mu": mu,
            "dt": dt,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "inv_dx": 1.0 / dx,
            "inv_dy": 1.0 / dy,
            "inv_dz": 1.0 / dz,
            "inv_dx2": 1.0 / (dx * dx),
            "inv_dy2": 1.0 / (dy * dy),
            "inv_dz2": 1.0 / (dz * dz),
        }

    # -----------------------------
    # Allow dict‑like and attribute‑like access
    # -----------------------------
    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
