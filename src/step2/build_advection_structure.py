# src/step2/build_advection_structure.py
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
        - Mask / is_fluid (optional)

    Returns
    -------
    dict
        {
          "advection_u": callable,
          "advection_v": callable,
          "advection_w": callable,
          "interpolation_scheme": str,
          "interpolation_stencils": None,
        }
    """

    # Use dict-style access (DummyState and Step 2 both expect this)
    const = state["Constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    # ------------------------------------------------------------------
    # Central difference helpers
    # ------------------------------------------------------------------

    def _central_diff_x(F: np.ndarray, dx: float) -> np.ndarray:
        d = np.zeros_like(F)
        if F.shape[0] > 1:
            d[1:-1, :, :] = (F[2:, :, :] - F[:-2, :, :]) / (2.0 * dx)
            d[0, :, :] = (F[1, :, :] - F[0, :, :]) / dx
            d[-1, :, :] = (F[-1, :, :] - F[-2, :, :]) / dx
        return d

    def _central_diff_y(F: np.ndarray, dy: float) -> np.ndarray:
        d = np.zeros_like(F)
        if F.shape[1] > 1:
            d[:, 1:-1, :] = (F[:, 2:, :] - F[:, :-2, :]) / (2.0 * dy)
            d[:, 0, :] = (F[:, 1, :] - F[:, 0, :]) / dy
            d[:, -1, :] = (F[:, -1, :] - F[:, -2, :]) / dy
        return d

    def _central_diff_z(F: np.ndarray, dz: float) -> np.ndarray:
        d = np.zeros_like(F)
        if F.shape[2] > 1:
            d[:, :, 1:-1] = (F[:, :, 2:] - F[:, :, :-2]) / (2.0 * dz)
            d[:, :, 0] = (F[:, :, 1] - F[:, :, 0]) / dz
            d[:, :, -1] = (F[:, :, -1] - F[:, :, -2]) / dz
        return d

    # ------------------------------------------------------------------
    # Advection operators
    # ------------------------------------------------------------------

    def advection_u(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Approximate (u · ∇)u on U-grid.
        """
        du_dx = _central_diff_x(U, dx)
        du_dy = _central_diff_y(U, dy)
        du_dz = _central_diff_z(U, dz)

        adv = U * du_dx
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

    # ------------------------------------------------------------------
    # Package operators
    # ------------------------------------------------------------------

    ops = {
        "advection_u": advection_u,
        "advection_v": advection_v,
        "advection_w": advection_w,
        "interpolation_scheme": "central",
        "interpolation_stencils": None,
    }

    # Store callables in state["Operators"]
    if "Operators" not in state:
        state["Operators"] = {}

    for k, v in ops.items():
        if k.startswith("advection_"):
            state["Operators"][k] = v

    # Metadata
    state["AdvectionMeta"] = {
        "interpolation_scheme": ops["interpolation_scheme"],
        "interpolation_stencils": ops["interpolation_stencils"],
    }

    return ops
