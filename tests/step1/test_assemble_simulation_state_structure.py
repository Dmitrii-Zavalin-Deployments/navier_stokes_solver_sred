# tests/step1/test_assemble_simulation_state_structure.py

import numpy as np

from src.step1.allocate_staggered_fields import allocate_staggered_fields
from src.step1.assemble_simulation_state import assemble_simulation_state
from src.step1.types import GridConfig, DerivedConstants, Config


def make_cfg(nx, ny, nz):
    return GridConfig(
        nx=nx, ny=ny, nz=nz,
        dx=1.0, dy=1.0, dz=1.0,
        x_min=0.0, y_min=0.0, z_min=0.0,
        x_max=float(nx), y_max=float(ny), z_max=float(nz),
    )


def make_constants():
    return DerivedConstants(
        rho=1.0, mu=1.0, dt=0.1,
        dx=1.0, dy=1.0, dz=1.0,
        inv_dx=1.0, inv_dy=1.0, inv_dz=1.0,
        inv_dx2=1.0, inv_dy2=1.0, inv_dz2=1.0,
    )


def make_config():
    return Config(
        domain={}, fluid={}, simulation={},
        forces={}, boundary_conditions={},
        geometry_definition={},
    )


def test_assemble_simulation_state_structure():
    cfg = make_cfg(2, 2, 2)
    fields = allocate_staggered_fields(cfg)
    constants = make_constants()
    config = make_config()
    mask_3d = np.ones((2, 2, 2), dtype=int)
    bc_table = {}

    state = assemble_simulation_state(
        config, cfg, fields, mask_3d, bc_table, constants
    )

    assert "grid" in state
    assert "fields" in state
    assert "constants" in state
    assert "config" in state
    assert "mask_3d" in state
    assert "boundary_table" in state
