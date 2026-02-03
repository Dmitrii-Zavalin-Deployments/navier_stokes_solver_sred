import numpy as np
import math

# NEW: import schema validator
from src.step1.schema_validator import validate_input_schema


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
    - test_04_grid_initialization.py
    - test_05_field_allocation.py
    - test_06_initial_conditions.py
    - test_07_geometry_mask.py

    This is NOT the final Step 1 implementation.
    """

    # ----------------------------------------------------------------------
    # 0. Schema validation (NEW)
    # ----------------------------------------------------------------------
    # If the input JSON does not match the schema, this will raise.
    validate_input_schema(json_input)

    # After schema validation, we can safely assume the structure is correct.
    # Now we extract the fields using the new schema structure.

    domain = json_input["domain_definition"]
    fluid = json_input["fluid_properties"]
    init = json_input["initial_conditions"]
    sim = json_input["simulation_parameters"]
    geom = json_input["geometry_definition"]
    forces = json_input["external_forces"]

    mask_flat = geom["geometry_mask_flat"]
    nx, ny, nz = geom["geometry_mask_shape"]

    # ----------------------------------------------------------------------
    # 1. Extract simulation parameters
    # ----------------------------------------------------------------------
    dt = sim["time_step"]
    iv = init["initial_velocity"]
    ip = init["initial_pressure"]
    forces["force_vector"]
    flattening_order = geom["flattening_order"]

    # ----------------------------------------------------------------------
    # 2. Physical constraints
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

    # Reject NaN / Inf in initial velocity
    for v in iv:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("initial_velocity contains NaN or Inf")

    # initial_pressure must be finite
    if math.isnan(ip) or math.isinf(ip):
        raise ValueError("initial_pressure contains NaN or Inf")

    # CFL pre-check (very loose)
    max_vel = max(abs(iv[0]), abs(iv[1]), abs(iv[2]))
    dx_tmp = abs(domain["x_max"] - domain["x_min"]) / nx
    if dt * max_vel > dx_tmp:
        raise ValueError("CFL pre-check failed: dt * |u| > dx")

    # ----------------------------------------------------------------------
    # 3. Geometry mask validation
    # ----------------------------------------------------------------------
    expected_len = nx * ny * nz
    if len(mask_flat) != expected_len:
        raise ValueError("geometry_mask_flat has incorrect length")

    # All mask values must be integers
    for m in mask_flat:
        if not isinstance(m, int):
            raise ValueError("geometry_mask_flat must contain integers only")

    unique_vals = set(mask_flat)

    # Case 1: binary mask (only 0/1)
    if unique_vals.issubset({0, 1}):
        pass

    # Case 2: pattern mask (contains no 0/1)
    elif unique_vals.isdisjoint({0, 1}):
        pass

    # Case 3: mixed mask → invalid
    else:
        raise ValueError("geometry_mask_flat contains invalid values")

    # ----------------------------------------------------------------------
    # 4. Geometry mask reshaping
    # ----------------------------------------------------------------------
    flat = np.array(mask_flat)

    order = flattening_order.strip()

    if order == "i + nx*(j + ny*k)":
        mask = flat.reshape((nx, ny, nz), order="C")

    elif order == "j + ny*(i + nx*k)":
        mask = np.zeros((nx, ny, nz), dtype=flat.dtype)
        idx = 0
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    mask[i, j, k] = flat[idx]
                    idx += 1
    else:
        raise ValueError("Unsupported flattening_order")

    # ----------------------------------------------------------------------
    # 5. Allocate fields
    # ----------------------------------------------------------------------
    x_min = domain["x_min"]
    x_max = domain["x_max"]
    dx = abs(x_max - x_min) / max(nx, 1)

    P = np.zeros((nx, ny, nz))
    U = np.zeros((nx + 1, ny, nz))
    V = np.zeros((nx, ny + 1, nz))
    W = np.zeros((nx, ny, nz + 1))

    grid = GridConfig(dx=dx)
    constants = {"dx": dx, "mu": fluid["viscosity"]}

    return SimulationState(P, U, V, W, mask, grid, constants)
