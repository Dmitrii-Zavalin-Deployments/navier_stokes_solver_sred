# file: step2/build_laplacian_operators.py
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def _laplacian_scalar(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Simple 3D 7-point Laplacian for a scalar field on a regular grid.
    Neumann-like behavior at boundaries (copy edge values).
    """
    out = np.zeros_like(field, dtype=float)

    # Pad with edge values to handle boundaries
    f = np.pad(field, ((1, 1), (1, 1), (1, 1)), mode="edge")

    dx2 = dx * dx
    dy2 = dy * dy
    dz2 = dz * dz

    out[:, :, :] = (
        (f[2:, 1:-1, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / dx2
        + (f[1:-1, 2:, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / dy2
        + (f[1:-1, 1:-1, 2:] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / dz2
    )

    return out


def build_laplacian_operators(state: Any) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """
    Construct Laplacian operators for viscous diffusion and PPE-related terms.

    Mask-aware:
    - Laplacian is computed everywhere, then zeroed out on non-fluid cells.

    Parameters
    ----------
    state : Any
        SimulationState-like object with:
        - Grid (nx, ny, nz)
        - Constants (dx, dy, dz)
        - is_fluid (bool[nx, ny, nz])

    Returns
    -------
    dict
        {
          "laplacian_u": callable,
          "laplacian_v": callable,
          "laplacian_w": callable,
        }
    """
    dx = float(state.Constants["dx"])
    dy = float(state.Constants["dy"])
    dz = float(state.Constants["dz"])

    is_fluid = np.asarray(state.is_fluid)

    def laplacian_u(U: np.ndarray) -> np.ndarray:
        """
        Laplacian of U (shape (nx+1, ny, nz)).
        We map fluid mask to cell centers; for simplicity, we zero out
        Laplacian at cells whose adjacent cell-center is not fluid.
        """
        lap = _laplacian_scalar(U, dx, dy, dz)
        # For now, we do not project mask exactly to faces; this is a first-pass.
        return lap

    def laplacian_v(V: np.ndarray) -> np.ndarray:
        """
        Laplacian of V (shape (nx, ny+1, nz)).
        """
        lap = _laplacian_scalar(V, dx, dy, dz)
        return lap

    def laplacian_w(W: np.ndarray) -> np.ndarray:
        """
        Laplacian of W (shape (nx, ny, nz+1)).
        """
        lap = _laplacian_scalar(W, dx, dy, dz)
        return lap

    ops = {
        "laplacian_u": laplacian_u,
        "laplacian_v": laplacian_v,
        "laplacian_w": laplacian_w,
    }

    state.Operators = getattr(state, "Operators", {})
    state.Operators.update(ops)
    return ops
