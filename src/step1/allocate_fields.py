# src/step1/allocate_fields.py

from __future__ import annotations
from typing import Dict, Any
import numpy as np


def allocate_fields(grid: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allocate cell-centered and staggered fields, aligned with the frozen Step 1 dummy.

    Produces:
      {
        "P": ndarray (nx, ny, nz),
        "U": ndarray (nx+1, ny, nz),
        "V": ndarray (nx, ny+1, nz),
        "W": ndarray (nx, ny, nz+1),
      }
    """

    nx = grid["nx"]
    ny = grid["ny"]
    nz = grid["nz"]

    return {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }
