# src/step2/prepare_ppe_structure.py
from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .build_ppe_rhs import build_ppe_rhs


def prepare_ppe_structure(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a PPE structure descriptor for Step‑2.

    Contract requirements:
      • Step‑2 schema requires ppe.rhs_builder to be a STRING.
      • Internal Step‑2 tests require ppe["rhs_builder"] to be CALLABLE.
      • No callables may appear in the JSON‑safe schema output.
    """

    const = state["constants"]
    rho = float(const["rho"])
    dt = float(const["dt"])

    grid = state["grid"]
    dx = float(grid["dx"])
    dy = float(grid["dy"])
    dz = float(grid["dz"])

    mask = np.asarray(state["mask_3d"])

    # Boundary table (Step‑1 → Step‑2)
    boundary_table = state.get("boundary_table_list", [])

    # PPE singularity rule:
    has_pressure_outlet = any(
        bc.get("type") == "pressure_outlet" for bc in boundary_table
    )
    ppe_is_singular = not has_pressure_outlet

    # ---------------------------------------------------------
    # INTERNAL callable (used by Step‑2 tests)
    # ---------------------------------------------------------
    def rhs_builder(divergence_list):
        div = np.asarray(divergence_list)
        rhs = build_ppe_rhs(
            divergence=div,
            mask=mask,
            rho=rho,
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
        )
        return rhs.tolist()

    # ---------------------------------------------------------
    # Return BOTH:
    #   • rhs_builder      → callable (internal use)
    #   • rhs_builder_name → string (schema‑safe)
    # ---------------------------------------------------------
    return {
        "rhs_builder": rhs_builder,          # callable for tests
        "rhs_builder_name": "rhs_builder",   # schema‑required string
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
