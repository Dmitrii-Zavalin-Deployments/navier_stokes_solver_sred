# tests/helpers/solver_step2_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 2).

Compliance:
- Rule 6: Zero-Redundancy (Independent of Factory logic).
- Rule 7: Atomic Numerical Truth (Fixed data for verification).
"""

from src.common.stencil_block import StencilBlock
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

# Define a lightweight Mock Cell that satisfies the StencilBlock interface
class SimpleCellMock:
    def __init__(self, index, is_ghost):
        self.index = index
        self.is_ghost = is_ghost
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.p = 0.0
        self.mask = 0

def make_step2_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns a 'frozen' prototype representing a successful Step 2 completion.
    Uses independent mocking to ensure validation is decoupled from Step 2 logic.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    physics_params = {
        "dx": 0.25, "dy": 0.25, "dz": 0.25,
        "dt": 0.001, "rho": 1000.0, "mu": 0.001,
        "f_vals": (0.0, 0.0, -9.81)
    }
    
    state.stencil_matrix = []
    nx_buf, ny_buf = nx + 2, ny + 2
    
    for k in range(nz + 2):
        for j in range(ny + 2):
            for i in range(nx + 2):
                # Manual index calculation independent of Factory logic
                index = i + nx_buf * (j + ny_buf * k)
                is_ghost = (i == 0 or i == nx + 1 or 
                            j == 0 or j == ny + 1 or 
                            k == 0 or k == nz + 1)
                
                # Mock cell construction
                cell = SimpleCellMock(index=index, is_ghost=is_ghost)
                
                # Manual StencilBlock construction for ground truth
                block = StencilBlock(
                    center=cell,
                    i_minus=cell, i_plus=cell,
                    j_minus=cell, j_plus=cell,
                    k_minus=cell, k_plus=cell,
                    **physics_params
                )
                state.stencil_matrix.append(block)
    
    state.ready_for_time_loop = True
    return state