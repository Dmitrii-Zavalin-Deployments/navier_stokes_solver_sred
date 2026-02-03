import math


def validate_physical_constraints(config: dict) -> None:
    dom = config["domain_definition"]
    fluid = config["fluid_properties"]
    ic = config["initial_conditions"]
    sim = config["simulation_parameters"]

    density = fluid["density"]
    viscosity = fluid["viscosity"]
    nx = dom["nx"]
    ny = dom["ny"]
    nz = dom["nz"]
    x_min, x_max = dom["x_min"], dom["x_max"]

    if density <= 0.0:
        raise ValueError("Density must be positive")

    if viscosity < 0.0:
        raise ValueError("Viscosity must be non-negative")

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Grid resolution must be positive")

    if x_max <= x_min:
        raise ValueError("Invalid domain extent in x")

    # Simple CFL-like precheck: dt * |u| < dx
    dt = sim["time_step"]
    u0 = ic["initial_velocity"]
    speed = math.sqrt(u0[0] ** 2 + u0[1] ** 2 + u0[2] ** 2)
    dx = abs(x_max - x_min) / nx
    if speed * dt > dx:
        raise ValueError("CFL precheck failed")
