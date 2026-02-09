# src/step2/prepare_ppe_structure.py
from __future__ import annotations
from typing import Any, Dict


def prepare_ppe_structure(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a JSON‑serializable PPE structure descriptor for Step‑2.

    Contract requirements:
      • Step‑2 schema requires ppe.rhs_builder to be a STRING, not a function.
      • Step‑3 expects state["ppe_structure"] to exist.
      • No callables are allowed anywhere in Step‑2 output.
    """

    const = state["constants"]
    rho = float(const["rho"])
    dt = float(const["dt"])

    # Boundary table (Step‑1 → Step‑2)
    boundary_table = state.get("boundary_table_list", [])

    # PPE singularity rule:
    # Non‑singular if ANY boundary has type "pressure_outlet"
    has_pressure_outlet = any(
        bc.get("type") == "pressure_outlet" for bc in boundary_table
    )
    ppe_is_singular = not has_pressure_outlet

    # Return a pure JSON‑serializable descriptor
    # IMPORTANT: rhs_builder MUST be a STRING per schema
    return {
        "rhs_builder": "rhs_builder",   # <-- FIXED: no functions allowed
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
