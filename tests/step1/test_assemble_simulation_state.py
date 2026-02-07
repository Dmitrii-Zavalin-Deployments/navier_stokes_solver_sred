# tests/step1/test_assemble_simulation_state.py
import numpy as np
from src.step1.assemble_simulation_state import assemble_simulation_state
from src.step1.types import GridConfig, Fields, Config, DerivedConstants


def test_assemble_simulation_state():
    grid = GridConfig(
        nx=1, ny=1, nz=1,
        dx=1, dy=1, dz=1,
        x_min=0, x_max=1,
        y_min=0, y_max=1,
        z_min=0, z_max=1,
    )

    fields = Fields(
        P=np.zeros((1,1,1)),
        U=np.zeros((2,1,1)),
        V=np.zeros((1,2,1)),
        W=np.zeros((1,1,2)),
        Mask=np.zeros((1,1,1)),
    )

    cfg = Config({}, {}, {}, {}, [], {})
    constants = DerivedConstants(1,1,1,1,1,1,1,1,1,1,1,1)

    state = assemble_simulation_state(cfg, grid, fields, fields.Mask, {}, constants)

    assert state.grid.nx == 1
    assert state.fields.P.shape == (1,1,1)
