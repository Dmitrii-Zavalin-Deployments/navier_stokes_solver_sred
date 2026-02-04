# file: step1/verify_staggered_shapes.py
from __future__ import annotations

from .types import Fields, GridConfig, SimulationState


def verify_staggered_shapes(state: SimulationState) -> None:
    """
    Ensure all arrays match expected MAC-grid shapes before entering Step 2.
    """
    grid: GridConfig = state.grid
    fields: Fields = state.fields

    nx, ny, nz = grid.nx, grid.ny, grid.nz

    if fields.P.shape != (nx, ny, nz):
        raise ValueError(f"P shape mismatch: expected {(nx, ny, nz)}, got {fields.P.shape}")
    if fields.U.shape != (nx + 1, ny, nz):
        raise ValueError(
            f"U shape mismatch: expected {(nx+1, ny, nz)}, got {fields.U.shape}"
        )
    if fields.V.shape != (nx, ny + 1, nz):
        raise ValueError(
            f"V shape mismatch: expected {(nx, ny+1, nz)}, got {fields.V.shape}"
        )
    if fields.W.shape != (nx, ny, nz + 1):
        raise ValueError(
            f"W shape mismatch: expected {(nx, ny, nz+1)}, got {fields.W.shape}"
        )
    if fields.Mask.shape != (nx, ny, nz):
        raise ValueError(
            f"Mask shape mismatch: expected {(nx, ny, nz)}, got {fields.Mask.shape}"
        )
