# src/step2/stencil_assembler.py

from .factory import build_core_cell, build_ghost_cell
from .stencil_block import StencilBlock

def assemble_stencil_matrix(state, nx, ny, nz, ctx, physics_params):
    """
    Assembles the 3D grid into a flattened matrix of StencilBlocks.
    Handles boundary/ghost cell logic seamlessly.
    """
    total_cells = (nx + 2) * (ny + 2) * (nz + 2)
    # StencilBlock is an object, so the matrix is now an Object Array
    local_stencil_matrix = np.empty((total_cells,), dtype=object)
    
    cursor = 0
    # Loop range including ghost boundaries
    for i in range(-1, nx + 1):
        for j in range(-1, ny + 1):
            for k in range(-1, nz + 1):
                
                # Helper to fetch individual cells
                def get_cell(ix, iy, iz):
                    # Clamp indices to handle boundaries for neighbor lookup
                    # Or use build_ghost_cell if out of bounds
                    if (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz):
                        return build_core_cell(ix, iy, iz, state, ctx)
                    return build_ghost_cell(ix, iy, iz, ctx)

                # Assemble the 7-point stencil
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