# tests/step1/test_compute_derived_constants_math.py

import math
import pytest

from src.step1.compute_derived_constants import compute_derived_constants
from src.step1.types import GridConfig


def make_grid(dx=1.0, dy=1.0, dz=1.0):
    return GridConfig(
        nx=4, ny=4, nz=4,
        dx=dx, dy=dy, dz=dz,
        x_min=0.0, y_min=0.0, z_min=0.0,
        x_max=4.0, y_max=4.0, z_max=4.0,
    )


def test_correct_computation_of_derived_constants():
    grid = make_grid(dx=2.0, dy=4.0, dz=0.5)

    constants = compute_derived_constants(
        grid,
        fluid_properties={"density": 10.0, "viscosity": 2.0},
        simulation_parameters={"time_step": 0.1},
    )

    assert constants.inv_dx == pytest.approx(1 / 2.0)
    assert constants.inv_dy == pytest.approx(1 / 4.0)
    assert constants.inv_dz == pytest.approx(1 / 0.5)

    assert constants.inv_dx2 == pytest.approx(1 / (2.0 * 2.0))
    assert constants.inv_dy2 == pytest.approx(1 / (4.0 * 4.0))
    assert constants.inv_dz2 == pytest.approx(1 / (0.5 * 0.5))


def test_missing_density_raises():
    grid = make_grid()
    with pytest.raises(KeyError):
        compute_derived_constants(
            grid,
            fluid_properties={"viscosity": 1.0},
            simulation_parameters={"time_step": 0.1},
        )


def test_missing_viscosity_raises():
    grid = make_grid()
    with pytest.raises(KeyError):
        compute_derived_constants(
            grid,
            fluid_properties={"density": 1.0},
            simulation_parameters={"time_step": 0.1},
        )


def test_missing_time_step_raises():
    grid = make_grid()
    with pytest.raises(KeyError):
        compute_derived_constants(
            grid,
            fluid_properties={"density": 1.0, "viscosity": 1.0},
            simulation_parameters={},
        )


def test_density_must_be_finite_and_positive():
    grid = make_grid()
    bad_values = [0.0, -1.0, float("inf"), float("nan")]

    for rho in bad_values:
        with pytest.raises(ValueError):
            compute_derived_constants(
                grid,
                fluid_properties={"density": rho, "viscosity": 1.0},
                simulation_parameters={"time_step": 0.1},
            )


def test_viscosity_must_be_finite_and_non_negative():
    grid = make_grid()
    bad_values = [-1.0, float("inf"), float("nan")]

    for mu in bad_values:
        with pytest.raises(ValueError):
            compute_derived_constants(
                grid,
                fluid_properties={"density": 1.0, "viscosity": mu},
                simulation_parameters={"time_step": 0.1},
            )


def test_time_step_must_be_finite_and_positive():
    grid = make_grid()
    bad_values = [0.0, -1.0, float("inf"), float("nan")]

    for dt in bad_values:
        with pytest.raises(ValueError):
            compute_derived_constants(
                grid,
                fluid_properties={"density": 1.0, "viscosity": 1.0},
                simulation_parameters={"time_step": dt},
            )


def test_grid_spacing_must_be_finite_and_positive():
    bad_values = [0.0, -1.0, float("inf"), float("nan")]

    for bad_dx in bad_values:
        grid = make_grid()
        object.__setattr__(grid, "dx", bad_dx)

        with pytest.raises(ValueError):
            compute_derived_constants(
                grid,
                fluid_properties={"density": 1.0, "viscosity": 1.0},
                simulation_parameters={"time_step": 0.1},
            )
