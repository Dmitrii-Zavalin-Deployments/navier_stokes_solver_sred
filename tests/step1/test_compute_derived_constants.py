# tests/step1/test_compute_derived_constants.py

from src.step1.compute_derived_constants import compute_derived_constants
from src.step1.types import GridConfig


def test_compute_constants():
    grid = GridConfig(
        nx=2, ny=2, nz=2,
        dx=0.5, dy=1.0, dz=2.0,
        x_min=0, x_max=1,
        y_min=0, y_max=2,
        z_min=0, z_max=4,
    )

    fluid = {"density": 1.2, "viscosity": 0.05}
    sim = {"time_step": 0.01}

    c = compute_derived_constants(grid, fluid, sim)

    # Basic passthrough values
    assert c.rho == 1.2
    assert c.mu == 0.05
    assert c.dt == 0.01

    # Derived inverse spacings
    assert c.inv_dx == 2.0
    assert c.inv_dy == 1.0
    assert c.inv_dz == 0.5
