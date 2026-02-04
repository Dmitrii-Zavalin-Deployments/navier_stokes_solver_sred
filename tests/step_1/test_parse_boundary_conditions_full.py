import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from src.step1.types import GridConfig


GRID = GridConfig(
    nx=4, ny=4, nz=4,
    dx=1, dy=1, dz=1,
    x_min=0, x_max=1,
    y_min=0, y_max=1,
    z_min=0, z_max=1,
)


def test_invalid_role():
    bc = [{"role": "rocket", "faces": ["x_min"], "apply_to": ["pressure"], "pressure": 1}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


def test_invalid_face():
    bc = [{"role": "wall", "faces": ["diagonal"], "apply_to": ["pressure"], "pressure": 1}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


def test_duplicate_face():
    bc = [
        {"role": "wall", "faces": ["x_min"], "apply_to": ["pressure"], "pressure": 1},
        {"role": "wall", "faces": ["x_min"], "apply_to": ["pressure"], "pressure": 2},
    ]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


def test_velocity_wrong_length():
    bc = [{"role": "inlet", "faces": ["x_min"], "apply_to": ["velocity"], "velocity": [1, 2]}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


def test_pressure_missing():
    bc = [{"role": "outlet", "faces": ["x_max"], "apply_to": ["pressure"]}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)
