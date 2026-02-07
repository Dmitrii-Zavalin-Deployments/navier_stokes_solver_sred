# tests/helpers/schema_dummy_state.py

import numpy as np

class SchemaDummyState(dict):
    """
    A fully Step‑1‑schema‑compliant dummy state for Step‑2 tests.

    This fixture produces:
    - config: full Step‑1 config block
    - grid: full Step‑1 grid block
    - fields: P, U, V, W, Mask with correct shapes (as Python lists)
    - mask_3d: Python list version of Mask
    - boundary_table: list of BC entries
    - constants: None (to be filled by precompute_constants)

    It is intentionally minimal but 100% schema‑accurate.
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
        # Grid block (Step‑1 schema)
        # -----------------------------
        grid = {
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
        config = {
            "boundary_conditions": {},
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
        # Mask
        # -----------------------------
        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=int)

        # -----------------------------
        # Fields block (Step‑1 schema)
        # MUST be Python lists, not NumPy arrays
        # -----------------------------
        fields = {
            "P": np.zeros((nx, ny, nz), float).tolist(),
            "U": np.zeros((nx + 1, ny, nz), float).tolist(),
            "V": np.zeros((nx, ny + 1, nz), float).tolist(),
            "W": np.zeros((nx, ny, nz + 1), float).tolist(),
            "Mask": mask.tolist(),
        }

        # -----------------------------
        # Final Step‑1 state
        # -----------------------------
        self["config"] = config
        self["grid"] = grid
        self["fields"] = fields
        self["mask_3d"] = mask.tolist()
        self["boundary_table"] = boundary_table or []
        self["constants"] = None  # Filled by precompute_constants
