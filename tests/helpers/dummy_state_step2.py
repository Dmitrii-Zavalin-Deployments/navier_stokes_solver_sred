import numpy as np

class DummyState:
    """
    A test double that mimics the real Step 1 → Step 2 SimulationState structure.
    Step 2 code expects capitalized attributes: Grid, Config, Constants, Mask, etc.
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
        # Match Step 1 naming conventions (capitalized)
        self.Grid = type("Grid", (), dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz))()

        # Mask
        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=int)
        self.Mask = mask

        # Config (capitalized)
        self.Config = type(
            "Config",
            (),
            dict(
                fluid_properties={"density": rho, "viscosity": mu},
                simulation_parameters={"dt": dt, "advection_scheme": scheme},
            ),
        )()

        # Boundary table
        self.BoundaryTable = boundary_table or []

        # Constants (Step 2 fills this)
        self.Constants = None

        # Fields expected by Step 2
        self.P = np.zeros((nx, ny, nz), dtype=float)
        self.U = np.zeros((nx + 1, ny, nz), dtype=float)
        self.V = np.zeros((nx, ny + 1, nz), dtype=float)
        self.W = np.zeros((nx, ny, nz + 1), dtype=float)
