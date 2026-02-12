# src/step4/bc_priority.py

def apply_priority_rule(bc_map):
    """
    Resolve boundary-condition conflicts at corners or edges.

    Parameters
    ----------
    bc_map : dict
        Maps BC type → face name.
        Example:
            {
                "no-slip": "x_min",
                "inlet":   "y_min",
                "slip":    "z_min"
            }

    Returns
    -------
    winning_bc_type : str
        The BC type that wins according to the priority rule.

    Notes
    -----
    This function does NOT modify the state or ghost cells.
    It only returns the winning BC type. The caller (bc_sync.py)
    is responsible for marking overridden faces.
    """

    # ------------------------------------------------------------
    # Priority order (highest → lowest)
    # ------------------------------------------------------------
    PRIORITY = [
        "no-slip",
        "inlet",
        "outlet",
        "slip",
        "symmetry",
        "pressure_dirichlet",
        "pressure_neumann",
    ]

    # Find the highest-priority BC present in bc_map
    for bc_type in PRIORITY:
        if bc_type in bc_map:
            return bc_type

    # If nothing matches (should not happen), return arbitrary BC
    return next(iter(bc_map.keys()))
