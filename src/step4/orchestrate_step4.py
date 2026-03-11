# src/step4/orchestrate_step4.py

from src.common.stencil_block import StencilBlock
from src.step4.boundary_applier import apply_boundary_values
from src.step4.boundary_dispatcher import get_applicable_boundary_configs


def orchestrate_step4(block: StencilBlock, boundary_cfg: list, grid, domain_cfg: dict) -> StencilBlock:
    """
    Step 4: Boundary Enforcement.
    Coordinates the identification and application of boundary conditions
    to a specific stencil block, using domain configuration to select the 
    physics strategy (Internal vs. External).
    
    Compliance:
    - Acts as the final stage of the pipeline to enforce physical constraints.
    - Operates exclusively on the Foundation through the StencilBlock pointer graph,
      ensuring that modifications are reflected in the global fields_buffer.
    """
    # Ghost cells are identified by the Foundation logic and are typically skipped 
    # to avoid redundant boundary condition overwriting.
    if block.center.is_ghost:
        return block

    # 1. Identify which boundaries apply based on location and the domain strategy
    # The dispatcher uses the schema-compliant block.center.mask to determine strategy.
    rules = get_applicable_boundary_configs(block, boundary_cfg, grid, domain_cfg)
    
    # 2. Apply updates for each rule found (Selective partial updates)
    # The applier uses the schema-locked property setters (e.g., block.center.vx = ...)
    # to perform in-place mutations of the contiguous Foundation buffer.
    for rule in rules:
        apply_boundary_values(block, rule)
        
    return block