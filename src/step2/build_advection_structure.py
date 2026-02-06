# src/step2/build_advection_structure.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np


def build_advection_structure(state: Any) -> Dict[str, Any]:
    """
    Build advection operators for U, V, W using either central or upwind scheme.
    """

    const = state["constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    scheme = state["config"]["simulation_parameters"].get("advection_scheme", "central")

    # ------------------------------------------------------------------
    # Central differences
    # ------------------------------------------------------------------
    def _central_x(F: np.ndarray) -> np.ndarray:
        d = np.zeros_like(F)
        if F.shape[0] > 1:
            d[1:-1] = (F[2:] - F[:-2]) / (2 * dx)
            d[0] = (F[1] - F[0]) / dx
            d[-1] = (F[-1] - F[-2]) / dx
        return d

    def _central_y(F: np.ndarray) -> np.ndarray:
        d = np.zeros_like(F)
        if F.shape[1] > 1:
            d[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * dy)
            d[:, 0] = (F[:, 1] - F[:, 0]) / dy
            d[:, -1] = (F[:, -1] - F[:, -2]) / dy
        return d

    def _central_z(F: np.ndarray) -> np.ndarray:
        d = np.zeros_like(F)
        if F.shape[2] > 1:
            d[:, :, 1:-1] = (F[:, :, 2:] - F[:, :, :-2]) / (2 * dz)
            d[:, :, 0] = (F[:, :, 1] - F[:, :, 0]) / dz
            d[:, :, -1] = (F[:, :, -1] - F[:, :, -2]) / dz
        return d

    # ------------------------------------------------------------------
    # Upwind advection in flux form (shape-safe, 1D loops)
    # ------------------------------------------------------------------
    def _upwind_flux_x(U: np.ndarray) -> np.ndarray:
        """
        Compute (u · ∂u/∂x) in a simple first-order upwind flux form:
            a_u[i] ≈ (F_i - F_{i-1}) / dx,  F = u^2 / 2
        using the sign of u to pick the upstream side.
        """
        nx = U.shape[0]
        a = np.zeros_like(U)
        if nx <= 1:
            return a

        F = 0.5 * U**2

        for i in range(1, nx):
            ui_mean = U[i].mean()
            if ui_mean > 0:
                a[i] = (F[i] - F[i - 1]) / dx
            else:
                if i + 1 < nx:
                    a[i] = (F[i + 1] - F[i]) / dx
                else:
                    a[i] = (F[i] - F[i - 1]) / dx

        a[0] = a[1]
        return a

    # For now, keep y/z upwind contributions simple and symmetric with x
    def _upwind_flux_y(V: np.ndarray) -> np.ndarray:
        ny = V.shape[1]
        a = np.zeros_like(V)
        if ny <= 1:
            return a

        F = 0.5 * V**2

        for j in range(1, ny):
            vj_mean = V[:, j].mean()
            if vj_mean > 0:
                a[:, j] = (F[:, j] - F[:, j - 1]) / dy
            else:
                if j + 1 < ny:
                    a[:, j] = (F[:, j + 1] - F[:, j]) / dy
                else:
                    a[:, j] = (F[:, j] - F[:, j - 1]) / dy

        a[:, 0] = a[:, 1]
        return a

    def _upwind_flux_z(W: np.ndarray) -> np.ndarray:
        nz = W.shape[2]
        a = np.zeros_like(W)
        if nz <= 1:
            return a

        F = 0.5 * W**2

        for k in range(1, nz):
            wk_mean = W[:, :, k].mean()
            if wk_mean > 0:
                a[:, :, k] = (F[:, :, k] - F[:, :, k - 1]) / dz
            else:
                if k + 1 < nz:
                    a[:, :, k] = (F[:, :, k + 1] - F[:, :, k]) / dz
                else:
                    a[:, :, k] = (F[:, :, k] - F[:, :, k - 1]) / dz

        a[:, :, 0] = a[:, :, 1]
        return a

    # ------------------------------------------------------------------
    # Advection operators
    # ------------------------------------------------------------------
    def advection_u(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        if scheme == "upwind":
            # Flux-form upwind in x, plus simple contributions in y/z
            ax = _upwind_flux_x(U)
            ay = _upwind_flux_y(U)
            az = _upwind_flux_z(U)
            return ax + 0.5 * (ay + az)
        else:
            du_dx = _central_x(U)
            du_dy = _central_y(U)
            du_dz = _central_z(U)
            return U * du_dx + 0.5 * (du_dy + du_dz)

    def advection_v(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        if scheme == "upwind":
            ax = _upwind_flux_x(V)
            ay = _upwind_flux_y(V)
            az = _upwind_flux_z(V)
            return ax + 0.5 * (ay + az)
        else:
            dv_dx = _central_x(V)
            dv_dy = _central_y(V)
            dv_dz = _central_z(V)
            return V * dv_dy + 0.5 * (dv_dx + dv_dz)

    def advection_w(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        if scheme == "upwind":
            ax = _upwind_flux_x(W)
            ay = _upwind_flux_y(W)
            az = _upwind_flux_z(W)
            return ax + 0.5 * (ay + az)
        else:
            dw_dx = _central_x(W)
            dw_dy = _central_y(W)
            dw_dz = _central_z(W)
            return W * dw_dz + 0.5 * (dw_dx + dw_dy)

    # ------------------------------------------------------------------
    # Package operators
    # ------------------------------------------------------------------
    ops: Dict[str, Any] = {
        "advection_u": advection_u,
        "advection_v": advection_v,
        "advection_w": advection_w,
        "interpolation_scheme": scheme,
        "interpolation_stencils": None,
    }

    if "operators" not in state:
        state["operators"] = {}
    state["operators"].update({k: v for k, v in ops.items() if k.startswith("advection_")})

    state["advection_meta"] = {
        "interpolation_scheme": scheme,
        "interpolation_stencils": None,
    }

    return ops