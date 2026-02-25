# src/step1/parse_boundary_conditions.py

from __future__ import annotations
from typing import Any, Dict, List

# Immutable Sets for Constitutional Validation
_VALID_LOCATIONS = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
_VALID_TYPES = {"no-slip", "free-slip", "inflow", "outflow", "pressure"}

def parse_boundary_conditions(
    bc_list: List[Dict[str, Any]],
    
) -> Dict[str, Dict[str, Any]]:
    """
    Translates JSON boundary definitions into a structured internal lookup table.

    Constitutional Role: Physics Contract Validator.
    Compliance: Phase F Data Intake (Prevention of Over-determined Systems).
    """
    parsed_table = {}

    for bc in bc_list:
        loc = bc.get("location")
        if loc not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid or missing boundary location: {loc}. Must be in {_VALID_LOCATIONS}")
        
        if loc in parsed_table:
            raise ValueError(f"Constitutional Violation: Duplicate BC for location '{loc}'.")

        bc_type = bc.get("type")
        if bc_type not in _VALID_TYPES:
            raise ValueError(f"Invalid boundary type: {bc_type}. Must be in {_VALID_TYPES}")

        values = bc.get("values", {})
        
        # 1. Inflow Validation: Strict Vector Requirement
        if bc_type == "inflow":
            for comp in ["u", "v", "w"]:
                if comp not in values or not isinstance(values[comp], (int, float)):
                    raise ValueError(f"Inflow at {loc} requires numeric velocity '{comp}'.")
        
        # 2. Pressure Validation: Scalar Requirement
        elif bc_type == "pressure":
            if "p" not in values or not isinstance(values["p"], (int, float)):
                raise ValueError(f"Pressure boundary at {loc} requires numeric 'p'.")

        # 3. Collision Prevention: Prevent inconsistent PDE constraints
        elif bc_type in ["no-slip", "free-slip", "outflow"]:
            if "p" in values:
                raise ValueError(f"BC Type '{bc_type}' at {loc} cannot define pressure. Use 'pressure' type instead.")

        # 4. Canonical Storage (Normalization to float64)
        parsed_table[loc] = {
            "type": bc_type,
            "u": float(values.get("u", 0.0)),
            "v": float(values.get("v", 0.0)),
            "w": float(values.get("w", 0.0)),
            "p": float(values.get("p", 0.0))
        }

    # 5. Final Audit: Verify all 6 faces are defined (Zero-Debt Mandate)
    missing_faces = _VALID_LOCATIONS - set(parsed_table.keys())
    if missing_faces:
         raise ValueError(f"Incomplete Domain: Missing boundary conditions for faces: {missing_faces}")

    return parsed_table