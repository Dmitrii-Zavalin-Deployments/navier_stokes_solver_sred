# tests/helpers/step2_schema_dummy_state.py

import numpy as np


class Step2SchemaDummyState(dict):
    """
    Step‑2 dummy state:
    - Structure matches Step‑1 output
    - Types match Step‑2 numerical operators
    """

    PROTECTED_KEYS = {
        "grid",
        "config",
        "fields",
        "mask_3d",
        "boundary_table",
        "constants",
    }

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

        # Validate dt
        if dt <= 0:
            raise ValueError("dt must be positive")

        # Grid block
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

        # Config block
        self["config"] = {
            "boundary_conditions": [],
            "domain": {},
            "fluid": {"density": rho, "viscosity": mu},
            "forces": {},
            "geometry_definition": {},
            "simulation": {"dt": dt, "advection_scheme": scheme},
        }

        # Mask
        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=int)

        # Fields
        self["fields"] = {
            "P": np.zeros((nx, ny, nz), float),
            "U": np.zeros((nx + 1, ny, nz), float),
            "V": np.zeros((nx, ny + 1, nz), float),
            "W": np.zeros((nx, ny, nz + 1), float),
            "Mask": mask,
        }

        self["mask_3d"] = mask.tolist()

        # Boundary table must be an object
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

        # Constants
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

    # Protect structured blocks
    def __setitem__(self, key, value):
        if key in self.PROTECTED_KEYS and not isinstance(value, dict):
            raise TypeError(f"Cannot overwrite structured block '{key}' with non-dict value")
        super().__setitem__(key, value)
