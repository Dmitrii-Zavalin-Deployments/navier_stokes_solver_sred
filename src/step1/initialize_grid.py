from .simulation_state import Grid


def initialize_grid(config: dict) -> Grid:
    dom = config["domain_definition"]
    x_min, x_max = dom["x_min"], dom["x_max"]
    y_min, y_max = dom["y_min"], dom["y_max"]
    z_min, z_max = dom["z_min"], dom["z_max"]
    nx, ny, nz = dom["nx"], dom["ny"], dom["nz"]

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    return Grid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
    )
