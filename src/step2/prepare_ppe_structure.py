# src/step2/prepare_ppe_structure.py
from __future__ import annotations
from typing import Any, Dict
import numpy as np


def _to_numpy(arr):
    return np.array(arr, dtype=float)


def _to_list(arr):
    return arr.tolist()


def prepare_ppe_structure(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare the structure and metadata for the Pressure Poisson Equation (PPE).

    Pure Step‑2 function:
    - Does NOT mutate the input state.
    - Returns JSON‑serializable data.
    - Uses canonical Step‑1 mask (mask_3d).
    """

    # ------------------------------------------------------------
    # Physical constants
    # ------------------------------------------------------------
    const = state["constants"]
    rho = float(const["rho"])
    dt = float(const["dt"])

    # ------------------------------------------------------------
    # Canonical mask (Step‑1 schema)
    # ------------------------------------------------------------
    mask = _to_numpy(state["mask_3d"])
    is_fluid = (mask != 0)  # treat -1 as fluid

    # ------------------------------------------------------------
    # Boundary table (Step‑2 schema)
    # ------------------------------------------------------------
    boundary_table = state.get("boundary_table_list", [])

    # ------------------------------------------------------------
    # RHS = -rho/dt * divergence, masked to fluid cells
    # ------------------------------------------------------------
    def rhs_builder_numpy(divergence: np.ndarray) -> np.ndarray:
        rhs = -rho / dt * divergence
        return np.where(is_fluid, rhs, 0.0)

    # JSON‑serializable wrapper
    def rhs_builder(divergence_list):
        div_np = _to_numpy(divergence_list)
        rhs_np = rhs_builder_numpy(div_np)
        return _to_list(rhs_np)

    # ------------------------------------------------------------
    # Singularity detection
    # PPE is non‑singular if ANY boundary has type "pressure_outlet"
    # ------------------------------------------------------------
    has_pressure_outlet = any(
        bc.get("type") == "pressure_outlet" for bc in boundary_table
    )
    ppe_is_singular = not has_pressure_outlet

    # ------------------------------------------------------------
    # Return pure JSON‑serializable PPE structure
    # ------------------------------------------------------------
    return {
        "rhs_builder": rhs_builder,  # JSON‑safe wrapper
        "solver_type": "PCG",
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "ppe_is_singular": ppe_is_singular,
        "meta": {
            "rho": rho,
            "dt": dt,
            "masking": "solid=0, fluid=1, boundary-fluid=-1",
        },
    }
