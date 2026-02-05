# tests/step_1/test_verify_staggered_shapes_full.py

import pytest
import numpy as np

from src.step1.verify_staggered_shapes import verify_staggered_shapes
from src.step1.types import GridConfig, Fields, SimulationState, DerivedConstants, Config


def make_state():
    grid = GridConfig(
        nx=2, ny=2, nz=2,
        dx=1, dy=1, dz=1,
        x_min=0, x_max=1,
        y_min=0, y_max=1,
        z_min=0, z_max=1,
    )

    fields = Fields(
        P=np.zeros((2,2,2)),
        U=np.zeros((3,2,2)),
        V=np.zeros((2,3,2)),
        W=np.zeros((2,2,3)),
        Mask=np.zeros((2,2,2)),
    )

    cfg = Config({}, {}, {}, {}, [], {})
    constants = DerivedConstants(1,1,1,1,1,1,1,1,1,1,1,1)

    return SimulationState(
        config=cfg,
        grid=grid,
        fields=fields,
        mask_3d=np.zeros((2,2,2)),
        boundary_table={},
        constants=constants,
    )


def test_bad_P_shape():
    state = make_state()
    state.fields.P = np.zeros((1,1,1))
    with pytest.raises(ValueError):
        verify_staggered_shapes(state)


def test_bad_V_shape():
    state = make_state()
    state.fields.V = np.zeros((2,2,2))
    with pytest.raises(ValueError):
        verify_staggered_shapes(state)


def test_bad_W_shape():
    state = make_state()
    state.fields.W = np.zeros((2,2,2))
    with pytest.raises(ValueError):
        verify_staggered_shapes(state)


def test_bad_Mask_shape():
    state = make_state()
    state.fields.Mask = np.zeros((1,1,1))
    with pytest.raises(ValueError):
        verify_staggered_shapes(state)
