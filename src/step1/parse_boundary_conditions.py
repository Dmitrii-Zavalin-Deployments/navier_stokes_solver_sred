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
    Step 1: Parse and validate BCs based on the JSON Schema and Theory of Operations.
    Returns a dictionary mapping locations to their normalized physical properties.
    """
    parsed_table = {}

    for bc in bc_list:
        # 1. Validate Location
        loc = bc.get("location")
        if not loc or loc not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid or missing boundary location: {loc}")
        
        # 2. Check for Duplicates (Collision Prevention)
        if loc in parsed_table:
            raise ValueError(f"Duplicate boundary condition defined for location: {loc}")

        # 3. Validate BC Type (Must match Schema: no-slip, free-slip, inflow, outflow, pressure)
        bc_type = bc.get("type")
        if bc_type not in _VALID_TYPES:
            raise ValueError(f"Invalid boundary type: {bc_type}")

        values = bc.get("values", {})
        
        # 4. Physical Logic Validations (Theory Compliance)
        if bc_type == "inflow":
            # Action: Validates that an inflow BC actually provides numerical values for {u, v, w}
            for comp in ["u", "v", "w"]:
                val = values.get(comp)
                if val is None or not isinstance(val, (int, float)):
                    raise ValueError(f"Inflow boundary at {loc} must provide numerical u, v, and w.")
        
        elif bc_type == "pressure":
            # Action: Validates that a pressure BC provides a numerical value for 'p'
            p_val = values.get("p")
            if p_val is None or not isinstance(p_val, (int, float)):
                raise ValueError(f"Pressure boundary at {loc} must provide numerical p.")

        elif bc_type in ["no-slip", "free-slip", "outflow"]:
            # Action: Prevents ambiguous "Pressure Dirichlet" on velocity-defined boundaries
            if "p" in values:
                raise ValueError(f"Boundary type {bc_type} at {loc} does not allow pressure values.")

        # 5. Normalization & SolverState Compatibility
        # We explicitly cast to float here to prevent integer-division debt in later steps.
        parsed_table[loc] = {
            "type": bc_type,
            "u": float(values.get("u", 0.0)),
            "v": float(values.get("v", 0.0)),
            "w": float(values.get("w", 0.0)),
            "p": float(values.get("p", 0.0)),
            "comment": bc.get("comment", "")
        }

    return parsed_table