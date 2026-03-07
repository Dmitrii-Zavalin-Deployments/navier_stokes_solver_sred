# src/step2/orchestrate_step2.py

def orchestrate_step2(state: SolverState, input_data: SolverInput) -> SolverState:
    # 1. Initialize Builder and Compiler
    builder = CellBuilder(input_data)
    compiler = TopologyCompiler(shape=(state.grid.nx, state.grid.ny, state.grid.nz))
    
    # 2. Loop
    for i in range(state.grid.nx):
        for j in range(state.grid.ny):
            for k in range(state.grid.nz):
                cell = builder.build(i, j, k)      # Factory creates
                compiler.compile_cell(cell)        # Compiler updates local buffers
                
    # 3. Commit
    compiler.commit_to_state(state)                # Final transfer to SolverState
    return state