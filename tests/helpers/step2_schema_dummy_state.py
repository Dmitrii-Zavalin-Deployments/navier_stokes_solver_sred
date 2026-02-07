# tests/helpers/step2_schema_dummy_state.py

import numpy as np


class Step2SchemaDummyState(dict):
    """
    Step‑2 dummy state:
    - Must satisfy Step‑1 schema (for orchestrator validation)
    - Must provide Step‑2‑friendly structures (for numerical operators)
    """

    PROTECTED_KEYS = {
        "grid",
        "config",
        "fields",
        # constants is intentionally NOT protected
        # boundary_table is protected because Step‑1 schema requires an object
        # mask_3d is intentionally NOT protected
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
        boundary_table=None,   # Step‑2 list of BC dicts
        scheme="upwind",
    ):
        super().__init__()

        # -----------------------------
        # Grid block (Step‑1 schema)
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
        # Config block (Step‑1 schema)
        # -----------------------------
        self["config"] = {
            "boundary_conditions": [],
            "domain": {},
            "fluid": {"density": rho, "viscosity": mu},
            "forces": {},
            "geometry_definition": {},
            "simulation": {"dt": dt, "advection_scheme": scheme},
        }

        # -----------------------------
        # Mask (NumPy for Step‑2)
        # -----------------------------
        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=int)

        # -----------------------------
        # Fields (Step‑1 structure, Step‑2 types)
        # -----------------------------
        self["fields"] = {
            "P": np.zeros((nx, ny, nz), float),
            "U": np.zeros((nx + 1, ny, nz), float),
            "V": np.zeros((nx, ny + 1, nz), float),
            "W": np.zeros((nx, ny, nz + 1), float),
            "Mask": mask,
        }

        self["mask_3d"] = mask.tolist()

        # -----------------------------
        # Boundary table
        #
        # Step‑1 schema requires an OBJECT:
        #   { "x_min": [], ... }
        #
        # Step‑2 operators require a LIST of BC dicts:
        #   [ {"face": ..., "type": ...}, ... ]
        #
        # So we store BOTH.
        # -----------------------------
        self["boundary_table_list"] = boundary_table if boundary_table is not None else []

        self["boundary_table"] = {
            "x_min": [],
            "x_max": [],
            "y_min": [],
            "y_max": [],
            "z_min": [],
            "z_max": [],
        }

        # -----------------------------
        # Constants block
        #
        # Step‑1 output includes constants as an object.
        # We initialize it as an EMPTY dict so:
        # - Step‑1 schema (type: object) is satisfied
        # - Step‑2 precompute_constants() can detect "empty" and recompute
        # -----------------------------
        self["constants"] = {}

    # -----------------------------
    # Protect structured blocks
    # -----------------------------
    def __setitem__(self, key, value):
        if key in self.PROTECTED_KEYS and not isinstance(value, dict):
            raise TypeError(f"Cannot overwrite structured block '{key}' with non-dict value")
        super().__setitem__(key, value)
