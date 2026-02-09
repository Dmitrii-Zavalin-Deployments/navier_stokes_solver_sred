# src/step2/build_advection_structure.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def _to_numpy(arr):
    return np.array(arr, dtype=float)


def _to_list(arr):
    return arr.tolist()


def build_advection_structure(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build advection contributions for U, V, W using either central or upwind scheme.
    Input: Step‑1 state (dict with lists)
    Output: dict with advection_u, advection_v, advection_w (lists)
    """

    const = state["constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    scheme = state["config"].get("simulation", {}).get("advection_scheme", "central")

    # Convert fields to numpy
    U = _to_numpy(state["fields"]["U"])
    V = _to_numpy(state["fields"]["V"])
    W = _to_numpy(state["fields"]["W"])

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
    # Upwind flux form
    # ------------------------------------------------------------------
    def _upwind_flux_x(F):
        nx = F.shape[0]
        a = np.zeros_like(F)
        if nx <= 1:
            return a

        flux = 0.5 * F**2
        for i in range(1, nx):
            mean_u = F[i].mean()
            if mean_u > 0:
                a[i] = (flux[i] - flux[i - 1]) / dx
            else:
                if i + 1 < nx:
                    a[i] = (flux[i + 1] - flux[i]) / dx
                else:
                    a[i] = (flux[i] - flux[i - 1]) / dx

        a[0] = a[1]
        return a

    def _upwind_flux_y(F):
        ny = F.shape[1]
        a = np.zeros_like(F)
        if ny <= 1:
            return a

        flux = 0.5 * F**2
        for j in range(1, ny):
            mean_v = F[:, j].mean()
            if mean_v > 0:
                a[:, j] = (flux[:, j] - flux[:, j - 1]) / dy
            else:
                if j + 1 < ny:
                    a[:, j] = (flux[:, j + 1] - flux[:, j]) / dy
                else:
                    a[:, j] = (flux[:, j] - flux[:, j - 1]) / dy

        a[:, 0] = a[:, 1]
        return a

    def _upwind_flux_z(F):
        nz = F.shape[2]
        a = np.zeros_like(F)
        if nz <= 1:
            return a

        flux = 0.5 * F**2
        for k in range(1, nz):
            mean_w = F[:, :, k].mean()
            if mean_w > 0:
                a[:, :, k] = (flux[:, :, k] - flux[:, :, k - 1]) / dz
            else:
                if k + 1 < nz:
                    a[:, :, k] = (flux[:, :, k + 1] - flux[:, :, k]) / dz
                else:
                    a[:, :, k] = (flux[:, :, k] - flux[:, :, k - 1]) / dz

        a[:, :, 0] = a[:, :, 1]
        return a

    # ------------------------------------------------------------------
    # Advection operators
    # ------------------------------------------------------------------
    def adv_u():
        if scheme == "upwind":
            return _upwind_flux_x(U) + 0.5 * (_upwind_flux_y(U) + _upwind_flux_z(U))
        return U * _central_x(U) + 0.5 * (_central_y(U) + _central_z(U))

    def adv_v():
        if scheme == "upwind":
            return _upwind_flux_x(V) + 0.5 * (_upwind_flux_y(V) + _upwind_flux_z(V))
        return V * _central_y(V) + 0.5 * (_central_x(V) + _central_z(V))

    def adv_w():
        if scheme == "upwind":
            return _upwind_flux_x(W) + 0.5 * (_upwind_flux_y(W) + _upwind_flux_z(W))
        return W * _central_z(W) + 0.5 * (_central_x(W) + _central_y(W))

    # ------------------------------------------------------------------
    # Return JSON‑serializable output
    # ------------------------------------------------------------------
    return {
        "advection": {
            "u": _to_list(adv_u()),
            "v": _to_list(adv_v()),
            "w": _to_list(adv_w()),
        },
        "advection_meta": {
            "interpolation_scheme": scheme,
        },
    }
