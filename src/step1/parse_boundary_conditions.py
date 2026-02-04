# file: step1/parse_boundary_conditions.py
from __future__ import annotations

from typing import Any, Dict, List

from .types import GridConfig


_VALID_ROLES = {"inlet", "outlet", "wall", "symmetry"}
_VALID_FACES = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}


def parse_boundary_conditions(
    bc_list: List[Dict[str, Any]],
    grid_config: GridConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalize BCs into solver-ready structures.
    Structural checks only: roles, faces, duplicates.
    """
    table: Dict[str, List[Dict[str, Any]]] = {f: [] for f in _VALID_FACES}
    seen_faces: Dict[str, bool] = {}

    for bc in bc_list:
        role = bc.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(
                f"Role must be one of {sorted(_VALID_ROLES)}, got {role!r}"
            )

        faces = bc.get("faces", [])
        if not isinstance(faces, list) or not faces:
            raise ValueError("BC must specify at least one face")

        for face in faces:
            if face not in _VALID_FACES:
                raise ValueError(
                    f"Face must be one of {sorted(_VALID_FACES)}, got {face!r}"
                )
            if face in seen_faces:
                raise ValueError(f"Duplicate BC definition for face {face}")
            seen_faces[face] = True
            table[face].append(dict(bc))

    return table
