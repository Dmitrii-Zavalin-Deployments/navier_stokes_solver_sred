# src/step1/parse_boundary_conditions.py

from __future__ import annotations
from typing import Any, Dict, List

# Explicitly matching the JSON Schema Enum
_VALID_LOCATIONS = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
_VALID_TYPES = {"no-slip", "free-slip", "inflow", "outflow", "pressure"}

def parse_boundary_conditions(
    bc_list: List[Dict[str, Any]],
    grid_config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Step 1: Parse and validate BCs based on the JSON Schema.
    Returns a dictionary mapping locations to their normalized physical properties.
    """
    parsed_table = {}

    for bc in bc_list:
        # 1. Validate Location
        loc = bc.get("location")
        if not loc or loc not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid or missing boundary location: {loc}")
        
        # 2. Check for Duplicates
        if loc in parsed_table:
            raise ValueError(f"Duplicate boundary condition for location: {loc}")

        # 3. Validate BC Type (Must match Schema: no-slip, etc.)
        bc_type = bc.get("type")
        if bc_type not in _VALID_TYPES:
            raise ValueError(f"Invalid boundary type: {bc_type}")

        values = bc.get("values", {})
        
        # 4. Physical Logic Validations
        if bc_type == "inflow":
            for comp in ["u", "v", "w"]:
                if comp not in values:
                    raise ValueError(f"Inflow at {loc} requires velocity component '{comp}'")
        
        if bc_type == "pressure":
            if "p" not in values:
                raise ValueError(f"Pressure boundary at {loc} requires value 'p'")

        if bc_type == "no-slip":
            # Extra safety check: no-slip implies zero velocity, 
            # and pressure is usually solved, not specified.
            if "p" in values:
                raise ValueError(f"No-slip boundary at {loc} cannot define pressure 'p'")

        # 5. Normalization
        # We cast to float here to ensure the SolverState remains numerically consistent.
        parsed_table[loc] = {
            "type": bc_type,
            "u": float(values.get("u", 0.0)),
            "v": float(values.get("v", 0.0)),
            "w": float(values.get("w", 0.0)),
            "p": float(values.get("p", 0.0)),
            "comment": bc.get("comment", "")
        }

    return parsed_table