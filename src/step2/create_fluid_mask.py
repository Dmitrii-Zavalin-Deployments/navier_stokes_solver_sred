# src/step2/create_fluid_mask.py
from __future__ import annotations
import numpy as np
from src.solver_state import SolverState


def create_fluid_mask(state: SolverState) -> None:
    """
    Convert integer mask into boolean masks for operator use.

    Updates:
      state.is_fluid
      state.is_boundary_cell
    """

    mask = np.asarray(state.mask)

    state.is_fluid = (mask != 0)
    state.is_boundary_cell = (mask == -1)
