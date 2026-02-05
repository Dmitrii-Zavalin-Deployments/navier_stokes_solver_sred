# file: step2/build_advection_structure.py
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def build_advection_structure(state: Any) -> Dict[str, Callable[..., np.ndarray]]:
    """
    Prepare interpolation and upwind logic for the nonlinear advection term.

    This is a first-pass implementation using simple central differences to
    approximate u · ∇u. It is intentionally conservative and can be refined
    later.

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
          "advection_u": callable,
          "advection_v": callable,
          "advection_w": callable,
          "interpolation_scheme": str,
          "interpolation_stencils": None (placeholder),
        }
    """
    dx = float(state.Constants["dx"])
    dy = float(state.Constants["dy"])
    dz = float(state.Constants["dz"])

    def _central_diff_x(F: np.ndarray, dx: float) -> np.ndarray:
        d = np.zeros_like(F)
        d[1:-1, :, :] = (F[2:, :, :] - F[:-2, :, :]) / (2.0 * dx)
        d[0, :, :] = (F[1, :, :] - F[0, :, :]) / dx
        d[-1, :, :] = (F[-1, :, :] - F[-2, :, :]) / dx
        return d

    def _central_diff_y(F: np.ndarray, dy: float) -> np.ndarray:
        d = np.zeros_like(F)
        d[:, 1:-1, :] = (F[:, 2:, :] - F[:, :-2, :]) / (2.0 * dy)
        d[:, 0, :] = (F[:, 1, :] - F[:, 0, :]) / dy
        d[:, -1, :] = (F[:, -1, :] - F[:, -2, :]) / dy
        return d

    def _central_diff_z(F: np.ndarray, dz: float) -> np.ndarray:
        d = np.zeros_like(F)
        d[:, :, 1:-1] = (F[:, :, 2:] - F[:, :, :-2]) / (2.0 * dz)
        d[:, :, 0] = (F[:, :, 1] - F[:, :, 0]) / dz
        d[:, :, -1] = (F[:, :, -1] - F[:, :, -2]) / dz
        return d

    def advection_u(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Approximate (u · ∇)u on U-grid.
        For now, we treat U as the advected quantity and use central differences.
        """
        du_dx = _central_diff_x(U, dx)
        du_dy = _central_diff_y(U, dy)
        du_dz = _central_diff_z(U, dz)

        # Interpolate velocities to U locations crudely by reusing U, V, W as-is.
        # This is a placeholder; a proper MAC interpolation can be added later.
        adv = U * du_dx  # dominant term along x
        adv += 0.5 * (du_dy + du_dz)
        return adv

    def advection_v(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        dv_dx = _central_diff_x(V, dx)
        dv_dy = _central_diff_y(V, dy)
        dv_dz = _central_diff_z(V, dz)
        adv = V * dv_dy
        adv += 0.5 * (dv_dx + dv_dz)
        return adv

    def advection_w(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        dw_dx = _central_diff_x(W, dx)
        dw_dy = _central_diff_y(W, dy)
        dw_dz = _central_diff_z(W, dz)
        adv = W * dw_dz
        adv += 0.5 * (dw_dx + dw_dy)
        return adv

    ops = {
        "advection_u": advection_u,
        "advection_v": advection_v,
        "advection_w": advection_w,
        "interpolation_scheme": "central",
        "interpolation_stencils": None,
    }

    state.Operators = getattr(state, "Operators", {})
    state.Operators.update(
        {k: v for k, v in ops.items() if k.startswith("advection_")}
    )
    # Non-callable metadata can be stored separately if desired.
    state.AdvectionMeta = {
        "interpolation_scheme": ops["interpolation_scheme"],
        "interpolation_stencils": ops["interpolation_stencils"],
    }

    return ops
