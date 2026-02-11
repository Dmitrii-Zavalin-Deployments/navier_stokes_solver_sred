# tests/step1/test_types_math.py

import numpy as np
import pytest

from src.step1.types import (
    GridConfig,
    Fields,
    DerivedConstants,
    SimulationState,
    Config,
)


# ---------------------------------------------------------
# GridConfig tests
# ---------------------------------------------------------

def test_gridconfig_invalid_dimensions():
    with pytest.raises(ValueError):
        GridConfig(
            nx=0, ny=1, nz=1,
            dx=1.0, dy=1.0, dz=1.0,
            x_min=0, y_min=0, z_min=0,
            x_max=1, y_max=1, z_max=1,
        )

    with pytest.raises(ValueError):
        GridConfig(
            nx=-1, ny=1, nz=1,
            dx=1.0, dy=1.0, dz=1.0,
            x_min=0, y_min=0, z_min=0,
            x_max=1, y_max=1, z_max=1,
        )


def test_gridconfig_invalid_spacings():
    bad_values = [0.0, -1.0, float("inf"), float("nan")]

    for bad in bad_values:
        with pytest.raises(ValueError):
            GridConfig(
                nx=1, ny=1, nz=1,
                dx=bad, dy=1.0, dz=1.0,
                x_min=0, y_min=0, z_min=0,
                x_max=1, y_max=1, z_max=1,
            )


def test_gridconfig_invalid_extents():
    bad_values = [float("inf"), float("nan"), "x"]

    for bad in bad_values:
        with pytest.raises(ValueError):
            GridConfig(
                nx=1, ny=1, nz=1,
                dx=1.0, dy=1.0, dz=1.0,
                x_min=bad, y_min=0, z_min=0,
                x_max=1, y_max=1, z_max=1,
            )


# ---------------------------------------------------------
# Fields tests
# ---------------------------------------------------------

def test_fields_requires_numpy_arrays():
    with pytest.raises(TypeError):
        Fields(P=[1], U=np.zeros((1,)), V=np.zeros((1,)), W=np.zeros((1,)), Mask=np.zeros((1,)))


def test_fields_mask_must_have_valid_values():
    with pytest.raises(ValueError):
        Fields(
            P=np.zeros((1,)),
            U=np.zeros((1,)),
            V=np.zeros((1,)),
            W=np.zeros((1,)),
            Mask=np.array([2]),  # invalid
        )


# ---------------------------------------------------------
# DerivedConstants tests
# ---------------------------------------------------------

def test_derivedconstants_requires_finite_values():
    with pytest.raises(ValueError):
        DerivedConstants(
            rho=float("inf"), mu=1.0, dt=1.0,
            dx=1.0, dy=1.0, dz=1.0,
            inv_dx=1.0, inv_dy=1.0, inv_dz=1.0,
            inv_dx2=1.0, inv_dy2=1.0, inv_dz2=1.0,
        )


def test_derivedconstants_physical_constraints():
    # rho must be positive
    with pytest.raises(ValueError):
        DerivedConstants(
            rho=0.0, mu=1.0, dt=1.0,
            dx=1.0, dy=1.0, dz=1.0,
            inv_dx=1.0, inv_dy=1.0, inv_dz=1.0,
            inv_dx2=1.0, inv_dy2=1.0, inv_dz2=1.0,
        )

    # mu must be non-negative
    with pytest.raises(ValueError):
        DerivedConstants(
            rho=1.0, mu=-1.0, dt=1.0,
            dx=1.0, dy=1.0, dz=1.0,
            inv_dx=1.0, inv_dy=1.0, inv_dz=1.0,
            inv_dx2=1.0, inv_dy2=1.0, inv_dz2=1.0,
        )

    # dt must be positive
    with pytest.raises(ValueError):
        DerivedConstants(
            rho=1.0, mu=1.0, dt=0.0,
            dx=1.0, dy=1.0, dz=1.0,
            inv_dx=1.0, inv_dy=1.0, inv_dz=1.0,
            inv_dx2=1.0, inv_dy2=1.0, inv_dz2=1.0,
        )


def test_derivedconstants_spacings_must_be_positive():
    with pytest.raises(ValueError):
        DerivedConstants(
            rho=1.0, mu=1.0, dt=1.0,
            dx=0.0, dy=1.0, dz=1.0,
            inv_dx=1.0, inv_dy=1.0, inv_dz=1.0,
            inv_dx2=1.0, inv_dy2=1.0, inv_dz2=1.0,
        )


# ---------------------------------------------------------
# SimulationState tests
# ---------------------------------------------------------

def test_simulationstate_requires_numpy_mask():
    cfg = Config({}, {}, {}, {}, [], {})
    grid = GridConfig(
        nx=1, ny=1, nz=1,
        dx=1.0, dy=1.0, dz=1.0,
        x_min=0, y_min=0, z_min=0,
        x_max=1, y_max=1, z_max=1,
    )
    fields = Fields(
        P=np.zeros((1,)),
        U=np.zeros((1,)),
        V=np.zeros((1,)),
        W=np.zeros((1,)),
        Mask=np.zeros((1,), dtype=int),
    )
    constants = DerivedConstants(
        rho=1.0, mu=1.0, dt=1.0,
        dx=1.0, dy=1.0, dz=1.0,
        inv_dx=1.0, inv_dy=1.0, inv_dz=1.0,
        inv_dx2=1.0, inv_dy2=1.0, inv_dz2=1.0,
    )

    with pytest.raises(TypeError):
        SimulationState(
            config=cfg,
            grid=grid,
            fields=fields,
            mask_3d=[1, 2, 3],  # not numpy array
            boundary_table={},
            constants=constants,
        )
