# tests/step4/test_step4_utilities.py

import numpy as np
from src.step4.utilities import (
    get_face_slices,
    get_normal_direction,
    get_tangential_directions,
    get_face_coordinates,
    apply_priority_rule,
)


# ----------------------------------------------------------------------
# 8.1 get_face_slices(face)
# ----------------------------------------------------------------------
def test_get_face_slices():
    # x_min
    sl = get_face_slices("x_min")
    assert sl == (slice(0, 1), slice(None), slice(None))

    # y_max
    sl = get_face_slices("y_max")
    assert sl == (slice(None), slice(-1, None), slice(None))

    # z_min
    sl = get_face_slices("z_min")
    assert sl == (slice(None), slice(None), slice(0, 1))


# ----------------------------------------------------------------------
# 8.2 get_normal_direction / get_tangential_directions
# ----------------------------------------------------------------------
def test_get_normal_and_tangential_directions():
    # x_min → normal = x, tangential = (y, z)
    n = get_normal_direction("x_min")
    t = get_tangential_directions("x_min")
    assert n == "x"
    assert set(t) == {"y", "z"}

    # y_max → normal = y, tangential = (x, z)
    n = get_normal_direction("y_max")
    t = get_tangential_directions("y_max")
    assert n == "y"
    assert set(t) == {"x", "z"}

    # z_min → normal = z, tangential = (x, y)
    n = get_normal_direction("z_min")
    t = get_tangential_directions("z_min")
    assert n == "z"
    assert set(t) == {"x", "y"}


# ----------------------------------------------------------------------
# 8.3 get_face_coordinates(face_type)
# ----------------------------------------------------------------------
def test_get_face_coordinates():
    # Synthetic coordinate field
    coords = np.zeros((5, 5, 5, 3))
    for i in range(5):
        for j in range(5):
            for k in range(5):
                coords[i, j, k] = [i, j, k]

    # x_min face should have x = 0 everywhere
    face = get_face_coordinates(coords, "x_min")
    assert np.all(face[..., 0] == 0)

    # z_max face should have z = 4 everywhere
    face = get_face_coordinates(coords, "z_max")
    assert np.all(face[..., 2] == 4)


# ----------------------------------------------------------------------
# 8.4 apply_priority_rule(values_at_corner)
# ----------------------------------------------------------------------
def test_apply_priority_rule():
    # Priority: no-slip > inlet > outlet > symmetry > pressure
    vals = {
        "no-slip": 111,
        "inlet": 222,
        "outlet": 333,
        "symmetry": 444,
        "pressure_dirichlet": 555,
    }

    result = apply_priority_rule(vals)
    assert result == 111  # highest priority wins

    # If no-slip absent → inlet wins
    vals2 = {
        "inlet": 10,
        "outlet": 20,
        "symmetry": 30,
    }
    assert apply_priority_rule(vals2) == 10


# ----------------------------------------------------------------------
# 8.5 Minimal Grid Utility Behavior
# ----------------------------------------------------------------------
def test_utilities_minimal_grid():
    coords = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                coords[i, j, k] = [i, j, k]

    # Minimal grid still must return valid slices and coordinates
    sl = get_face_slices("x_min")
    assert isinstance(sl, tuple)

    face = get_face_coordinates(coords, "x_min")
    assert face.shape[0] == 1  # x_min slice

    # Priority rule must still work
    assert apply_priority_rule({"symmetry": 1, "pressure_neumann": 2}) == 1
