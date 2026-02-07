# src/step2/prepare_ppe_structure.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np


def prepare_ppe_structure(state: Any) -> Dict[str, Any]:
    """
    Prepare the structure and metadata for the Pressure Poisson Equation (PPE).

    This does NOT build the full sparse matrix A. Instead, it prepares:
    - rhs_builder: how to map divergence to RHS
    - solver_type, tolerance, max_iterations
    - ppe_is_singular: whether the PPE is structurally singular (no pressure Dirichlet)
    """

    # ------------------------------------------------------------------
    # Physical constants (schema-correct)
    # ------------------------------------------------------------------
    const = state["constants"]
    rho = float(const["rho"])
    dt = float(const["dt"])

    # ------------------------------------------------------------------
    # Mask (schema-correct)
    # ------------------------------------------------------------------
    mask = np.asarray(state["fields"]["Mask"])
    is_fluid = (mask != 0)  # treat -1 as fluid

    # ------------------------------------------------------------------
    # Boundary table from Step 1 (schema-correct)
    # ------------------------------------------------------------------
    boundary_table = state.get("boundary_table", [])

    # ------------------------------------------------------------------
    # RHS builder: apply -rho/dt * divergence on fluid cells only
    # ------------------------------------------------------------------
    def rhs_builder(divergence: np.ndarray) -> np.ndarray:
        rhs = -rho / dt * divergence
        rhs = np.where(is_fluid, rhs, 0.0)
        return rhs

    # ------------------------------------------------------------------
    # Singularity detection:
    # PPE is non-singular if ANY boundary has type "pressure_outlet"
    # ------------------------------------------------------------------
    has_pressure_outlet = any(
        bc.get("type") == "pressure_outlet" for bc in boundary_table
    )

    ppe_is_singular = not has_pressure_outlet

    # ------------------------------------------------------------------
    # Package PPE structure
    # ------------------------------------------------------------------
    ppe = {
        "rhs_builder": rhs_builder,
        "solver_type": "PCG",
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "ppe_is_singular": ppe_is_singular,
    }

    # Store in schema-correct location
    state["ppe"] = ppe
    return ppe
