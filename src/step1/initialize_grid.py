# src/step1/initialize_grid.py

from __future__ import annotations
from typing import Dict, Any


def initialize_grid(domain: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal grid initializer aligned with the frozen Step 1 dummy.

    Produces:
      {
        "nx": int,
        "ny": int,
        "nz": int,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
      }

    Step 1 does not compute physical spacing from domain extents.
    It simply constructs a uniform grid with dx = dy = dz = 1.0,
    matching the Step 1 dummy and Step 1 schema.
    """

    nx = int(domain["nx"])
    ny = int(domain["ny"])
    nz = int(domain["nz"])

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }
