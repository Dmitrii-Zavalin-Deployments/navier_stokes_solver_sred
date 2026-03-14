# tests/helpers/solver_step2_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 2).

Compliance:
- Rule 6: Zero-Redundancy (This is a snapshot, not an orchestration).
- Rule 7: Atomic Numerical Truth (Use fixed values for verification).
"""

from src.common.stencil_block import StencilBlock
from src.step2.factory import build_core_cell
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def make_step2_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns a 'frozen' prototype representing a successful Step 2 completion.
    The stencil_matrix is populated with static, verified StencilBlocks, 
    accounting for ghost cell layers.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Define Fixed Physics Baseline (Atomic Numerical Truth)
    physics_params = {
        "dx": 0.25, "dy": 0.25, "dz": 0.25,
        "dt": 0.001, "rho": 1000.0, "mu": 0.001,
        "f_vals": (0.0, 0.0, -9.81)
    }
    
    # 2. Populate with Static Blocks (Accounting for Ghost Cells)
    # We iterate over (N+2) dimensions to match the buffer allocation
    state.stencil_matrix = []
    for i in range(nx + 2):
        for j in range(ny + 2):
            for k in range(nz + 2):
                # Using the factory to build the cell
                cell = build_core_cell(i, j, k, state)
                
                # Mocking the block connectivity for the test baseline
                block = StencilBlock(
                    center=cell,
                    i_minus=cell, i_plus=cell,
                    j_minus=cell, j_plus=cell,
                    k_minus=cell, k_plus=cell,
                    **physics_params
                )
                state.stencil_matrix.append(block)
    
    # 3. Explicitly mark as ready
    state.ready_for_time_loop = True
    
    return state