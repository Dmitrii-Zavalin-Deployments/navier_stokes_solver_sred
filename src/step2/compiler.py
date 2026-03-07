import numpy as np
from .cell import Cell
from src.solver_state import SolverState

class TopologyCompiler:
    def __init__(self, shape: tuple[int, int, int]):
        self.shape = shape
        # Internal buffers (The "Local" Memory)
        self.mask_buffer = np.zeros(shape, dtype=np.int8)
        self.is_fluid_buffer = np.zeros(shape, dtype=bool)
        self.is_boundary_buffer = np.zeros(shape, dtype=bool)

    def compile_cell(self, cell: Cell):
        # Update buffers based on cell data
        i, j, k = cell.x, cell.y, cell.z
        self.mask_buffer[i, j, k] = cell.mask
        self.is_fluid_buffer[i, j, k] = (cell.mask == 1)
        self.is_boundary_buffer[i, j, k] = (cell.mask == -1)

    def commit_to_state(self, state: SolverState):
        # Final atomic transfer to the Source of Truth
        state.masks.mask = self.mask_buffer
        state.masks.is_fluid = self.is_fluid_buffer
        state.masks.is_boundary = self.is_boundary_buffer
