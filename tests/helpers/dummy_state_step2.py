import numpy as np

class DummyState(dict):
    """
    Dict-backed test state that matches Step 2 expectations:
    - state["Grid"]["nx"], state["Grid"]["dx"], ...
    - state["Config"]["simulation_parameters"]["dt"]
    - state["Mask"]
    - state["Constants"] (filled by precompute_constants)
    - state["BoundaryTable"]

    Also exposes U, V, W, P as attributes for convenience.
    """

    def __init__(
        self,
        nx,
        ny,
        nz,
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

        # Grid dictionary (Step 2 expects dict-like access)
        grid = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
        }

        # Config dictionary (Step 2 expects dict-like access)
        config = {
            "fluid_properties": {
                "density": rho,
                "viscosity": mu,
            },
            "simulation_parameters": {
                "dt": dt,
                "advection_scheme": scheme,
            },
        }

        # Mask
        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=int)

        # Populate dict entries
        self["Grid"] = grid
        self["Config"] = config
        self["Mask"] = mask
        self["BoundaryTable"] = boundary_table or []
        self["Constants"] = None  # Filled by precompute_constants

        # Also expose attribute-style access (optional convenience)
        self.Grid = self["Grid"]
        self.Config = self["Config"]
        self.Mask = self["Mask"]
        self.BoundaryTable = self["BoundaryTable"]
        self.Constants = self["Constants"]

        # Fields used by Step 2 operators
        self.P = np.zeros((nx, ny, nz), dtype=float)
        self.U = np.zeros((nx + 1, ny, nz), dtype=float)
        self.V = np.zeros((nx, ny + 1, nz), dtype=float)
        self.W = np.zeros((nx, ny, nz + 1), dtype=float)