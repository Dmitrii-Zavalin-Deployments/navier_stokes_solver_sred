# src/step2/build_ppe_rhs.py
from __future__ import annotations
import numpy as np


def build_ppe_rhs(
    divergence: np.ndarray,
    mask: np.ndarray,
    rho: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """
    Compute the RHS of the pressure Poisson equation:

        RHS = (rho / dt) * divergence

    Masking rules (from tests):
      • mask == 1   → fluid          → keep RHS
      • mask == 0   → solid          → RHS = 0
      • mask == -1  → boundary-fluid → RHS = 0
    """

    div = np.asarray(divergence, dtype=float)
    mask = np.asarray(mask)

    # Base RHS
    rhs = (rho / dt) * div

    # Apply masking
    rhs = np.where(mask == 1, rhs, 0.0)

    return rhs
