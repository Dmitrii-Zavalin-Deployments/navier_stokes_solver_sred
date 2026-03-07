from src.solver_input import SolverInput
from src.solver_state import SolverState

from .compiler import TopologyCompiler
from .factory import CellBuilder


def orchestrate_step2(state: SolverState, input_data: SolverInput) -> SolverState:
    # 1. Initialize Builder and Compiler
    builder = CellBuilder(input_data)
    compiler = TopologyCompiler(shape=(state.grid.nx, state.grid.ny, state.grid.nz))
    
    # 2. Loop through the grid
    for i in range(state.grid.nx):
        for j in range(state.grid.ny):
            for k in range(state.grid.nz):
                # Build transient object
                cell = builder.build(i, j, k)
                # Compile to local buffers
                compiler.compile_cell(cell)
                
    # 3. Final atomic update
    compiler.commit_to_state(state)
    return state
