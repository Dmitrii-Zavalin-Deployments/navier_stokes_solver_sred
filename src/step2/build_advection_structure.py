# src/step2/build_advection_structure.py
from __future__ import annotations

from typing import Any, Callable, Dict
import numpy as np


def build_advection_structure(state: Any) -> Dict[str, Callable[..., np.ndarray]]:
    """
    Build advection operators for U, V, W using either central or upwind scheme.
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
    # Upwind differences (shape-safe, loop-based)
    # ------------------------------------------------------------------
    def _upwind_x(F, U):
        nx = F.shape[0]
        d = np.zeros_like(F)

        if nx <= 1:
            return d

        for i in range(1, nx):
            if U[i].mean() > 0:  # backward
                d[i] = (F[i] - F[i - 1]) / dx
            else:  # forward
                if i + 1 < nx:
                    d[i] = (F[i + 1] - F[i]) / dx
                else:
                    d[i] = (F[i] - F[i - 1]) / dx

        d[0] = (F[1] - F[0]) / dx
        return d

    def _upwind_y(F, V):
        ny = F.shape[1]
        d = np.zeros_like(F)

        if ny <= 1:
            return d

        for j in range(1, ny):
            if V[:, j].mean() > 0:
                d[:, j] = (F[:, j] - F[:, j - 1]) / dy
            else:
                if j + 1 < ny:
                    d[:, j] = (F[:, j + 1] - F[:, j]) / dy
                else:
                    d[:, j] = (F[:, j] - F[:, j - 1]) / dy

        d[:, 0] = (F[:, 1] - F[:, 0]) / dy
        return d

    def _upwind_z(F, W):
        nz = F.shape[2]
        d = np.zeros_like(F)

        if nz <= 1:
            return d

        for k in range(1, nz):
            if W[:, :, k].mean() > 0:
                d[:, :, k] = (F[:, :, k] - F[:, :, k - 1]) / dz
            else:
                if k + 1 < nz:
                    d[:, :, k] = (F[:, :, k + 1] - F[:, :, k]) / dz
                else:
                    d[:, :, k] = (F[:, :, k] - F[:, :, k - 1]) / dz

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
    state["Operators"].update({k: v for k, v in ops.items() if k.startswith("advection_")})

    state["AdvectionMeta"] = {
        "interpolation_scheme": scheme,
        "interpolation_stencils": None,
    }

    return ops
