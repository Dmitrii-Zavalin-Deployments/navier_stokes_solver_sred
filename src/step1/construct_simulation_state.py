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


# vulture: ignore
def construct_simulation_state(json_input):
    """
    Minimal stub implementation that satisfies test_01_happy_path.py.
    Does NOT implement real Step 1 logic yet.
    """

    nx = json_input["domain"]["nx"]
    ny = json_input["domain"]["ny"]
    nz = json_input["domain"]["nz"]

    x_min = json_input["domain"]["x_min"]
    x_max = json_input["domain"]["x_max"]

    dx = abs(x_max - x_min) / max(nx, 1)

    P = np.zeros((nx, ny, nz))
    U = np.zeros((nx + 1, ny, nz))
    V = np.zeros((nx, ny + 1, nz))
    W = np.zeros((nx, ny, nz + 1))

    mask = np.zeros((nx, ny, nz))

    grid = GridConfig(dx=dx)
    constants = {"dx": dx, "mu": json_input["fluid"]["viscosity"]}

    return SimulationState(P, U, V, W, mask, grid, constants)
