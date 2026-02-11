# tests/step1/test_parse_boundary_conditions_math.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from src.step1.types import GridConfig


def make_grid():
    return GridConfig(
        nx=4, ny=4, nz=4,
        dx=1.0, dy=1.0, dz=1.0,
        x_min=0.0, x_max=4.0,
        y_min=0.0, y_max=4.0,
        z_min=0.0, z_max=4.0,
    )


# ---------------------------------------------------------
# Role validation
# ---------------------------------------------------------

def test_invalid_role_raises():
    grid = make_grid()
    bc = [{"role": "invalid", "faces": ["x_min"], "apply_to": []}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, grid)


# ---------------------------------------------------------
# Face validation
# ---------------------------------------------------------

def test_missing_faces_raises():
    grid = make_grid()
    bc = [{"role": "wall", "apply_to": []}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, grid)


def test_invalid_face_raises():
    grid = make_grid()
    bc = [{"role": "wall", "faces": ["not_a_face"], "apply_to": []}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, grid)


def test_duplicate_face_raises():
    grid = make_grid()
    bc = [
        {"role": "wall", "faces": ["x_min"], "apply_to": []},
        {"role": "wall", "faces": ["x_min"], "apply_to": []},
    ]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, grid)


# ---------------------------------------------------------
# apply_to validation
# ---------------------------------------------------------

def test_apply_to_must_be_list():
    grid = make_grid()
    bc = [{"role": "wall", "faces": ["x_min"], "apply_to": "not_a_list"}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, grid)


def test_invalid_apply_to_entry_raises():
    grid = make_grid()
    bc = [{"role": "wall", "faces": ["x_min"], "apply_to": ["invalid"]}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, grid)


# ---------------------------------------------------------
# Velocity BC validation
# ---------------------------------------------------------

def test_velocity_bc_requires_three_finite_components():
    grid = make_grid()

    bad_velocities = [
        [1, 2],          # wrong length
        [1, 2, float("inf")],
        [1, 2, float("nan")],
        ["a", 2, 3],
    ]

    for vel in bad_velocities:
        bc = [{
            "role": "inlet",
            "faces": ["x_min"],
            "apply_to": ["velocity"],
            "velocity": vel,
        }]
        with pytest.raises(ValueError):
            parse_boundary_conditions(bc, grid)


# ---------------------------------------------------------
# Pressure BC validation
# ---------------------------------------------------------

def test_pressure_bc_must_be_finite_scalar():
    grid = make_grid()

    bad_values = [float("inf"), float("nan"), "x"]

    for p in bad_values:
        bc = [{
            "role": "inlet",
            "faces": ["x_min"],
            "apply_to": ["pressure"],
            "pressure": p,
        }]
        with pytest.raises(ValueError):
            parse_boundary_conditions(bc, grid)


# ---------------------------------------------------------
# Pressure gradient BC validation
# ---------------------------------------------------------

def test_pressure_gradient_bc_must_be_finite_scalar():
    grid = make_grid()

    bad_values = [float("inf"), float("nan"), "x"]

    for pg in bad_values:
        bc = [{
            "role": "inlet",
            "faces": ["x_min"],
            "apply_to": ["pressure_gradient"],
            "pressure_gradient": pg,
        }]
        with pytest.raises(ValueError):
            parse_boundary_conditions(bc, grid)


# ---------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------

def test_unknown_key_raises():
    grid = make_grid()
    bc = [{
        "role": "wall",
        "faces": ["x_min"],
        "apply_to": [],
        "unknown_key": 123,
    }]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, grid)


# ---------------------------------------------------------
# Valid BC normalization
# ---------------------------------------------------------

def test_valid_bc_is_normalized_correctly():
    grid = make_grid()
    bc = [{
        "role": "inlet",
        "faces": ["x_min", "y_max"],
        "apply_to": ["velocity"],
        "velocity": [1.0, 2.0, 3.0],
        "comment": "test comment",
    }]

    table = parse_boundary_conditions(bc, grid)

    assert "x_min" in table
    assert "y_max" in table
    assert len(table["x_min"]) == 1
    assert len(table["y_max"]) == 1

    entry = table["x_min"][0]
    assert entry["role"] == "inlet"
    assert entry["velocity"] == [1.0, 2.0, 3.0]
    assert entry["comment"] == "test comment"


# ---------------------------------------------------------
# Comment key allowed
# ---------------------------------------------------------

def test_comment_key_is_allowed():
    grid = make_grid()
    bc = [{
        "role": "wall",
        "faces": ["x_min"],
        "apply_to": [],
        "comment": "ok",
    }]

    table = parse_boundary_conditions(bc, grid)
    assert table["x_min"][0]["comment"] == "ok"
