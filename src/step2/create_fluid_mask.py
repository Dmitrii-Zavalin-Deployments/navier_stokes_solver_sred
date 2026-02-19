# src/step2/create_fluid_mask.py

from __future__ import annotations
import numpy as np
from src.solver_state import SolverState


def create_fluid_mask(state: SolverState) -> None:
    """
    Convert integer mask into boolean masks for operator use.

    In the input schema:
     1  = Pure Fluid
     0  = Solid
    -1  = Boundary-Fluid (Fluid cell adjacent to a wall)

    Updates:
      state.is_fluid: True for any cell where flow is calculated (1, -1).
      state.is_boundary_cell: True for specialized boundary fluid cells (-1).
      state.is_solid: True for cells where no flow occurs (0).
    """

    mask = np.asarray(state.mask)

    # Logic: Fluid is anything that is NOT solid (values 1 and -1)
    state.is_fluid = (mask != 0)
    
    # Logic: Specifically identify the boundary-fluid interface
    state.is_boundary_cell = (mask == -1)
    
    # Logic: Identify solid cells to satisfy the Schema and Scale Guard
    # This prevents state.is_solid from being None, which caused the test failure.
    state.is_solid = (mask == 0)