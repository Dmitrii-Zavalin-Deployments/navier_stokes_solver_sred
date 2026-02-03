from .simulation_state import Grid


def compute_derived_constants(config: dict, grid: Grid) -> dict:
    fluid = config["fluid_properties"]
    mu = fluid["viscosity"]
    dx = grid.dx
    return {
        "mu": float(mu),
        "dx": float(dx),
    }
