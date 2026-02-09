# file: step1/parse_boundary_conditions.py
from __future__ import annotations

from typing import Any, Dict, List
import math

from .types import GridConfig


_VALID_ROLES = {"inlet", "outlet", "wall", "symmetry"}
_VALID_FACES = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
_VALID_APPLY_TO = {"velocity", "pressure", "pressure_gradient"}


def parse_boundary_conditions(
    bc_list: List[Dict[str, Any]],
    grid_config: GridConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalize BCs into solver-ready structures.

    Enforces:
      • valid roles
      • valid faces
      • valid apply_to entries
      • no duplicate faces
      • velocity BC must have 3 finite components
      • pressure BC must have a finite scalar
      • pressure_gradient BC must have a finite scalar
      • allows optional 'comment' field (schema‑approved)
    """

    table: Dict[str, List[Dict[str, Any]]] = {f: [] for f in _VALID_FACES}
    seen_faces: Dict[str, bool] = {}

    for bc in bc_list:

        # ---------------------------------------------------------
        # Validate role
        # ---------------------------------------------------------
        role = bc.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"Role must be one of {sorted(_VALID_ROLES)}, got {role!r}")

        # ---------------------------------------------------------
        # Validate faces
        # ---------------------------------------------------------
        faces = bc.get("faces")
        if not isinstance(faces, list) or not faces:
            raise ValueError("BC must specify at least one face")

        for face in faces:
            if face not in _VALID_FACES:
                raise ValueError(f"Face must be one of {sorted(_VALID_FACES)}, got {face!r}")
            if face in seen_faces:
                raise ValueError(f"Duplicate BC definition for face {face}")
            seen_faces[face] = True

        # ---------------------------------------------------------
        # Validate apply_to
        # ---------------------------------------------------------
        apply_to = bc.get("apply_to", [])
        if not isinstance(apply_to, list):
            raise ValueError("apply_to must be a list")
        for item in apply_to:
            if item not in _VALID_APPLY_TO:
                raise ValueError(
                    f"apply_to entries must be one of {sorted(_VALID_APPLY_TO)}, got {item!r}"
                )

        # ---------------------------------------------------------
        # Validate velocity BC
        # ---------------------------------------------------------
        if "velocity" in apply_to:
            vel = bc.get("velocity")
            if not isinstance(vel, list) or len(vel) != 3:
                raise ValueError("Velocity BC must provide 3 components")
            for idx, v in enumerate(vel):
                if not (isinstance(v, (int, float)) and math.isfinite(v)):
                    raise ValueError(f"Velocity component {idx} must be finite, got {v}")

        # ---------------------------------------------------------
        # Validate pressure BC
        # ---------------------------------------------------------
        if "pressure" in apply_to:
            p = bc.get("pressure")
            if not isinstance(p, (int, float)) or not math.isfinite(p):
                raise ValueError("Pressure BC must provide a finite scalar")

        # ---------------------------------------------------------
        # Validate pressure gradient BC
        # ---------------------------------------------------------
        if "pressure_gradient" in apply_to:
            pg = bc.get("pressure_gradient")
            if not isinstance(pg, (int, float)) or not math.isfinite(pg):
                raise ValueError("pressure_gradient must be a finite scalar")

        # ---------------------------------------------------------
        # Validate allowed keys (now includes 'comment')
        # ---------------------------------------------------------
        allowed_keys = {
            "role", "faces", "apply_to",
            "velocity", "pressure", "pressure_gradient",
            "no_slip", "type",
            "comment",   # <-- added to match schema
        }

        for key in bc.keys():
            if key not in allowed_keys:
                raise ValueError(f"Unknown BC key: {key!r}")

        # ---------------------------------------------------------
        # Store normalized BC
        # ---------------------------------------------------------
        normalized = {k: v for k, v in bc.items() if k in allowed_keys}

        for face in faces:
            table[face].append(normalized)

    return table
