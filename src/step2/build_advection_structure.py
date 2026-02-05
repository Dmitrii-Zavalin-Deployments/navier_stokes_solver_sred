# src/step2/build_advection_structure.py
from __future__ import annotations

from typing import Any, Callable, Dict
import numpy as np


def build_advection_structure(state: Any) -> Dict[str, Callable[..., np.ndarray]]:
    """
    Build advection operators for U, V, W using either central or upwind scheme.

    Parameters
    ----------
    state : Any
        SimulationState-like object with:
        - state["Constants"] (dx, dy, dz)
        - state["Config"]["simulation_parameters"]["advection_scheme"]

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

    const = state["Constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    scheme = state["Config"]["simulation_parameters"].get("advection_scheme", "central")

    # ------------------------------------------------------------------
    # Central differences
    # ------------------------------------------------------------------
    def _central_x(F):
        d = np.zeros_like(F)
        if F.shape[0] > 1:
            d[1:-1] = (F[2:] - F[:-2]) / (2 * dx)
            d[0] = (F[1] - F[0]) / dx
            d[-1] = (F[-1] - F[-2]) / dx
        return d

    def _central_y(F):
        d = np.zeros_like(F)
        if F.shape[1] > 1:
            d[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * dy)
            d[:, 0] = (F[:, 1] - F[:, 0]) / dy
            d[:, -1] = (F[:, -1] - F[:, -2]) / dy
        return d

    def _central_z(F):
        d = np.zeros_like(F)
        if F.shape[2] > 1:
            d[:, :, 1:-1] = (F[:, :, 2:] - F[:, :, :-2]) / (2 * dz)
            d[:, :, 0] = (F[:, :, 1] - F[:, :, 0]) / dz
            d[:, :, -1] = (F[:, :, -1] - F[:, :, -2]) / dz
        return d

    # ------------------------------------------------------------------
    # Upwind differences (simple first-order)
    # ------------------------------------------------------------------
    def _upwind_x(F, U):
        d = np.zeros_like(F)
        if F.shape[0] > 1:
            # backward if U>0, forward if U<=0
            d[1:] = np.where(U[1:] > 0,
                             (F[1:] - F[:-1]) / dx,
                             (F[2:] - F[1:-1]) / dx)
            d[0] = (F[1] - F[0]) / dx
        return d

    def _upwind_y(F, V):
        d = np.zeros_like(F)
        if F.shape[1] > 1:
            d[:, 1:] = np.where(V[:, 1:] > 0,
                                (F[:, 1:] - F[:, :-1]) / dy,
                                (F[:, 2:] - F[:, 1:-1]) / dy)
            d[:, 0] = (F[:, 1] - F[:, 0]) / dy
        return d

    def _upwind_z(F, W):
        d = np.zeros_like(F)
        if F.shape[2] > 1:
            d[:, :, 1:] = np.where(W[:, :, 1:] > 0,
                                   (F[:, :, 1:] - F[:, :, :-1]) / dz,
                                   (F[:, :, 2:] - F[:, :, 1:-1]) / dz)
            d[:, :, 0] = (F[:, :, 1] - F[:, :, 0]) / dz
        return d

    # ------------------------------------------------------------------
    # Advection operators
    # ------------------------------------------------------------------
    def advection_u(U, V, W):
        if scheme == "upwind":
            du_dx = _upwind_x(U, U)
            du_dy = _upwind_y(U, V)
            du_dz = _upwind_z(U, W)
        else:
            du_dx = _central_x(U)
            du_dy = _central_y(U)
            du_dz = _central_z(U)

        return U * du_dx + 0.5 * (du_dy + du_dz)

    def advection_v(U, V, W):
        if scheme == "upwind":
            dv_dx = _upwind_x(V, U)
            dv_dy = _upwind_y(V, V)
            dv_dz = _upwind_z(V, W)
        else:
            dv_dx = _central_x(V)
            dv_dy = _central_y(V)
            dv_dz = _central_z(V)

        return V * dv_dy + 0.5 * (dv_dx + dv_dz)

    def advection_w(U, V, W):
        if scheme == "upwind":
            dw_dx = _upwind_x(W, U)
            dw_dy = _upwind_y(W, V)
            dw_dz = _upwind_z(W, W)
        else:
            dw_dx = _central_x(W)
            dw_dy = _central_y(W)
            dw_dz = _central_z(W)

        return W * dw_dz + 0.5 * (dw_dx + dw_dy)

    # ------------------------------------------------------------------
    # Package operators
    # ------------------------------------------------------------------
    ops = {
        "advection_u": advection_u,
        "advection_v": advection_v,
        "advection_w": advection_w,
        "interpolation_scheme": scheme,
        "interpolation_stencils": None,
    }

    if "Operators" not in state:
        state["Operators"] = {}

    for k, v in ops.items():
        if k.startswith("advection_"):
            state["Operators"][k] = v

    state["AdvectionMeta"] = {
        "interpolation_scheme": scheme,
        "interpolation_stencils": None,
    }

    return ops
