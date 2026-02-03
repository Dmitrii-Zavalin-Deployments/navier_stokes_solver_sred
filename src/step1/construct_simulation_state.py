import numpy as np


class SimulationState:
    """
    Minimal placeholder SimulationState object.
    Only includes fields required by test_01_happy_path.py.
    """

    def __init__(self, P, U, V, W, mask, grid, constants):
        self.P = P
        self.U = U
        self.V = V
        self.W = W
        self.mask = mask
        self.grid = grid
        self.constants = constants


class GridConfig:
    """Minimal placeholder grid config."""
    def __init__(self, dx):
        self.dx = dx


def construct_simulation_state(json_input):
    """
    Minimal stub implementation that satisfies:
    - test_01_happy_path.py
    - test_02_schema_validation.py
    - test_03_physical_constraints.py

    This is NOT the final Step 1 implementation.
    """

    # ----------------------------------------------------------------------
    # 1. Required top-level keys
    # ----------------------------------------------------------------------
    required_keys = ["domain", "fluid", "simulation", "geometry_mask_flat"]
    for key in required_keys:
        if key not in json_input:
            raise KeyError(f"Missing required key: {key}")

    domain = json_input["domain"]
    fluid = json_input["fluid"]
    simulation = json_input["simulation"]
    mask_flat = json_input["geometry_mask_flat"]

    # ----------------------------------------------------------------------
    # 2. Required domain keys
    # ----------------------------------------------------------------------
    for key in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]:
        if key not in domain:
            raise KeyError(f"Missing domain key: {key}")

    # Type check for nx, ny, nz
    if not isinstance(domain["nx"], int):
        raise TypeError("nx must be an integer")
    if not isinstance(domain["ny"], int):
        raise TypeError("ny must be an integer")
    if not isinstance(domain["nz"], int):
        raise TypeError("nz must be an integer")

    nx = domain["nx"]
    ny = domain["ny"]
    nz = domain["nz"]

    # ----------------------------------------------------------------------
    # 3. Simulation keys
    # ----------------------------------------------------------------------
    if "flattening_order" not in simulation:
        raise KeyError("Missing flattening_order")

    # initial_velocity must be a list of length 3
    iv = simulation.get("initial_velocity")
    if not isinstance(iv, list):
        raise TypeError("initial_velocity must be a list")
    if len(iv) != 3:
        raise ValueError("initial_velocity must have length 3")

    # force_vector must be a list of length 3
    fv = simulation.get("force_vector")
    if not isinstance(fv, list):
        raise TypeError("force_vector must be a list")
    if len(fv) != 3:
        raise ValueError("force_vector must have length 3")

    # ----------------------------------------------------------------------
    # 4. Mask length check
    # ----------------------------------------------------------------------
    expected_len = nx * ny * nz
    if len(mask_flat) != expected_len:
        raise ValueError("geometry_mask_flat has incorrect length")

    # ----------------------------------------------------------------------
    # 5. Physical constraints
    # ----------------------------------------------------------------------

    # Density must be strictly positive
    if fluid["density"] <= 0:
        raise ValueError("fluid density must be > 0")

    # Viscosity must be non-negative
    if fluid["viscosity"] < 0:
        raise ValueError("fluid viscosity must be >= 0")

    # Grid resolution must be >= 1
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, nz must be >= 1")

    # Domain extents must be valid
    if domain["x_max"] <= domain["x_min"]:
        raise ValueError("x_max must be > x_min")
    if domain["y_max"] <= domain["y_min"]:
        raise ValueError("y_max must be > y_min")
    if domain["z_max"] <= domain["z_min"]:
        raise ValueError("z_max must be > z_min")

    # CFL pre-check (very loose)
    dt = simulation["dt"]
    vel = simulation["initial_velocity"]
    max_vel = max(abs(vel[0]), abs(vel[1]), abs(vel[2]))

    dx_tmp = abs(domain["x_max"] - domain["x_min"]) / nx

    if dt * max_vel > dx_tmp:
        raise ValueError("CFL pre-check failed: dt * |u| > dx")

    # ----------------------------------------------------------------------
    # 6. Minimal stub logic (unchanged)
    # ----------------------------------------------------------------------
    x_min = domain["x_min"]
    x_max = domain["x_max"]
    dx = abs(x_max - x_min) / max(nx, 1)

    P = np.zeros((nx, ny, nz))
    U = np.zeros((nx + 1, ny, nz))
    V = np.zeros((nx, ny + 1, nz))
    W = np.zeros((nx, ny, nz + 1))
    mask = np.zeros((nx, ny, nz))

    grid = GridConfig(dx=dx)
    constants = {"dx": dx, "mu": fluid["viscosity"]}

    return SimulationState(P, U, V, W, mask, grid, constants)
