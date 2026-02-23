# src/step1/parse_boundary_conditions.py

from __future__ import annotations
from typing import Any, Dict, List

_VALID_LOCATIONS = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
_VALID_TYPES = {"no-slip", "free-slip", "inflow", "outflow", "pressure"}

def parse_boundary_conditions(
    bc_list: List[Dict[str, Any]],
    grid_config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Translates JSON boundary definitions into a flattened internal table.
    Compliance: Phase F Data Intake (Strict Theory Enforcement).
    """
    parsed_table = {}

    for bc in bc_list:
        loc = bc.get("location")
        if not loc or loc not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid or missing boundary location: {loc}")
        
        if loc in parsed_table:
            raise ValueError(f"Duplicate boundary condition for location: {loc}")

        bc_type = bc.get("type")
        if bc_type not in _VALID_TYPES:
            raise ValueError(f"Invalid boundary type: {bc_type}")

        values = bc.get("values", {})
        
        # 1. Inflow Validation: Requires full velocity vector
        if bc_type == "inflow":
            for comp in ["u", "v", "w"]:
                val = values.get(comp)
                if val is None or not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Inflow at {loc} requires velocity component '{comp}'; "
                        f"must provide numerical u, v, and w."
                    )
        
        # 2. Pressure Validation: Requires scalar p
        elif bc_type == "pressure":
            p_val = values.get("p")
            if p_val is None or not isinstance(p_val, (int, float)):
                raise ValueError(
                    f"Pressure boundary at {loc} requires value 'p'; "
                    f"must provide numerical p."
                )

        # 3. Collision Prevention: No pressure allowed on velocity-specified boundaries
        elif bc_type in ["no-slip", "free-slip", "outflow"]:
            if "p" in values:
                raise ValueError(
                    f"Boundary type {bc_type} at {loc} cannot define pressure 'p'; "
                    f"does not allow pressure values."
                )

        # 4. Canonical Storage
        parsed_table[loc] = {
            "type": bc_type,
            "u": float(values.get("u", 0.0)),
            "v": float(values.get("v", 0.0)),
            "w": float(values.get("w", 0.0)),
            "p": float(values.get("p", 0.0)),
            "comment": bc.get("comment", "")
        }

    return parsed_table