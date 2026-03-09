# src/step4/boundary_dispatcher.py

def get_applicable_boundary_configs(block, boundary_cfg: list, grid: GridContext) -> list:
    """
    Unified dispatcher returning a list of dicts:
    {'location': str, 'type': str, 'values': dict}
    """
    mask = block.center.mask
    
    # 1. Internal Boundary Fluid (-1)
    if mask == -1:
        return _find_config(boundary_cfg, "internal_boundary")
        
    # 2. Solid (0)
    if mask == 0:
        return [{'location': 'solid', 'type': 'no-slip', 'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}}]
        
    # 3. Domain Boundaries (1)
    b_type = _get_domain_location_type(block, grid)
    if b_type != "none":
        return _find_config(boundary_cfg, b_type)
        
    # 4. Interior Fluid (1)
    return [{'location': 'interior', 'type': 'fluid_gas', 'values': {}}]

def _find_config(boundary_cfg, location):
    """Returns the config entry including location, type, and values."""
    for bc in boundary_cfg:
        if bc["location"] == location:
            # Return as a list to maintain loop compatibility in orchestrator
            return [bc]
    
    # Fallback if config is missing: return default struct
    return [{'location': location, 'type': 'default', 'values': {}}]