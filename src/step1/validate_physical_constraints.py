import math
from jsonschema import ValidationError


def validate_physical_constraints(config: dict) -> None:
    dom = config["domain_definition"]
    fluid = config["fluid_properties"]
    ic = config["initial_conditions"]
    sim = config["simulation_parameters"]
    geom = config["geometry_definition"]
    forces = config["external_forces"]

    density = fluid["density"]
    viscosity = fluid["viscosity"]
    nx = dom["nx"]
    ny = dom["ny"]
    nz = dom["nz"]
    x_min, x_max = dom["x_min"], dom["x_max"]

    # -----------------------------
    # 1. Basic physical constraints
    # -----------------------------
    if density <= 0.0:
        raise ValueError("Density must be positive")

    if viscosity < 0.0:
        raise ValueError("Viscosity must be non-negative")

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Grid resolution must be positive")

    if x_max <= x_min:
        raise ValueError("Invalid domain extent in x")

    # -----------------------------
    # 2. NaN / Inf checks
    # -----------------------------
    u0 = ic["initial_velocity"]
    if any(math.isnan(v) or math.isinf(v) for v in u0):
        raise ValueError("initial_velocity contains NaN or Inf")

    p0 = ic["initial_pressure"]
    if math.isnan(p0) or math.isinf(p0):
        raise ValueError("initial_pressure contains NaN or Inf")

    fvec = forces["force_vector"]
    if any(math.isnan(f) or math.isinf(f) for f in fvec):
        raise ValueError("force_vector contains NaN or Inf")

    # -----------------------------
    # 3. Geometry mask structural constraint
    # -----------------------------
    shape = geom["geometry_mask_shape"]

    # Only structural check here; all other mask logic is in map_geometry_mask.
    if len(shape) != 3:
        raise ValidationError("geometry_mask_shape must have length 3")

    # -----------------------------
    # 4. CFL-like precheck
    # -----------------------------
    dt = sim["time_step"]
    speed = math.sqrt(u0[0] ** 2 + u0[1] ** 2 + u0[2] ** 2)
    dx = abs(x_max - x_min) / nx

    if speed * dt > dx:
        raise ValueError("CFL precheck failed")