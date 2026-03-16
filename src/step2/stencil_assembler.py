# src/step2/stencil_assembler.py

from src.common.field_schema import FI
from src.common.grid_math import get_flat_index
from src.common.solver_state import SolverState
from src.common.stencil_block import StencilBlock

from .factory import get_cell

# Rule 7: Granular Traceability
DEBUG = True

class CellRegistry:
    def __init__(self, nx: int, ny: int, nz: int):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # 2-Tier Architecture: Ghost layer at -1 and nx 
        # Total dimension: (nx + 1) - (-1) + 1 = nx + 2
        self.nx_dim = nx + 2
        self.ny_dim = ny + 2
        self.nz_dim = nz + 2
        self._cache = [None] * (self.nx_dim * self.ny_dim * self.nz_dim)

    def _get_idx(self, i: int, j: int, k: int) -> int:
        # Per Section 7: Valid coordinate range is [-1, nx]
        # Coordinates must be >= -1 AND <= nx
        if not (-1 <= i <= self.nx and -1 <= j <= self.ny and -1 <= k <= self.nz):
             raise IndexError(f"Stencil accessing out-of-bounds: ({i}, {j}, {k})")
        # Offset 1 maps coordinate -1 to index 0
        return get_flat_index(i, j, k, self.nx_dim, self.ny_dim, offset=1)

    def get_or_create(self, i: int, j: int, k: int, state: SolverState):
        idx = self._get_idx(i, j, k)
        if self._cache[idx] is None:
            self._cache[idx] = get_cell(i, j, k, state)
        
        return self._cache[idx]

def assemble_stencil_matrix(state: SolverState) -> list:
    """
    Assembles a flattened list of StencilBlocks restricted to the Core Domain
    [0, nx-1] while maintaining access to Ghost buffers [-1, nx].
    """
    if state.fields.data.shape[-1] != FI.num_fields():
        raise RuntimeError(f"Foundation Mismatch: Buffer width {state.fields.data.shape[-1]} "
                           f"!= Schema requirement {FI.num_fields()}.")

    grid = state.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    registry = CellRegistry(nx, ny, nz)
    
    physics_params = {
        "dx": grid.dx,
        "dy": grid.dy,
        "dz": grid.dz,
        "dt": state.simulation_parameters.time_step,
        "rho": state.fluid_properties.density,
        "mu": state.fluid_properties.viscosity,
        "f_vals": tuple(state.external_forces.force_vector)
    }

    if DEBUG:
        print(f"DEBUG [Step 2.2]: Stencil Assembly Started for {nx}x{ny}x{nz} Core Domain")

    local_stencil_list = []
    
    # Iterate ONLY over the Core Domain [0, nx-1]
    # Registry lookups safely resolve neighbors in Ghost layer [-1, nx]
    for k in range(0, nz):
        for j in range(0, ny):
            for i in range(0, nx):
                block = StencilBlock(
                    center=registry.get_or_create(i, j, k, state),
                    i_minus=registry.get_or_create(i - 1, j, k, state),
                    i_plus=registry.get_or_create(i + 1, j, k, state),
                    j_minus=registry.get_or_create(i, j - 1, k, state),
                    j_plus=registry.get_or_create(i, j + 1, k, state),
                    k_minus=registry.get_or_create(i, j, k - 1, state),
                    k_plus=registry.get_or_create(i, j, k + 1, state),
                    **physics_params
                )
                local_stencil_list.append(block)
    
    if DEBUG:
        print(f"DEBUG [Step 2.2]: Successfully assembled {len(local_stencil_list)} Core StencilBlocks.")
    
    return local_stencil_list