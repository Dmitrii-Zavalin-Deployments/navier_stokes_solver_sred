import numpy as np
from .factory import build_core_cell, build_ghost_cell
from .stencil_block import StencilBlock

def assemble_stencil_matrix(state, nx, ny, nz, ctx, physics_params):
    """
    Assembles a flattened matrix of StencilBlocks containing only Core cells.
    Neighbors (including ghost cells) are resolved automatically via get_cell.
    """
    # 1. Size the matrix for Core Cells only
    total_core_cells = nx * ny * nz
    local_stencil_matrix = np.empty((total_core_cells,), dtype=object)
    
    cursor = 0
    
    # 2. Iterate through the full domain (including ghost layer)
    # We loop through the full grid to ensure boundary access, 
    # but only instantiate StencilBlocks for the internal Core domain.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                # Helper to fetch individual cells
                def get_cell(ix, iy, iz):
                    if (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz):
                        return build_core_cell(ix, iy, iz, state, ctx)
                    return build_ghost_cell(ix, iy, iz, ctx)

                # Assemble the 7-point stencil for the current core cell
                block = StencilBlock(
                    center=get_cell(i, j, k),
                    i_minus=get_cell(i-1, j, k), i_plus=get_cell(i+1, j, k),
                    j_minus=get_cell(i, j-1, k), j_plus=get_cell(i, j+1, k),
                    k_minus=get_cell(i, j, k-1), k_plus=get_cell(i, j, k+1),
                    **physics_params
                )
                
                local_stencil_matrix[cursor] = block
                cursor += 1
                
    return local_stencil_matrix