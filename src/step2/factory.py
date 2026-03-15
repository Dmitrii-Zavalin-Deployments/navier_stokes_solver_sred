# src/step2/factory.py

from src.common.cell import Cell
from src.common.solver_state import SolverState

# Rule 7: Granular Traceability
DEBUG = False

# Centralized cache for Flyweight pattern
_CELL_CACHE = {}

# Explicit constants for ghost cell initialization (Rule 5 compliance)
GHOST_VELOCITY = (0.0, 0.0, 0.0)
GHOST_PRESSURE = 0.0
GHOST_MASK = 0

def get_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Unified entry point for Cell retrieval. 
    Implements Flyweight caching to ensure topological identity.
    """
    coord = (i, j, k)
    if coord in _CELL_CACHE:
        return _CELL_CACHE[coord]
    
    grid = state.grid
    # Determine if we are building a Core or Ghost cell based on grid bounds
    is_core = (0 <= i < grid.nx) and (0 <= j < grid.ny) and (0 <= k < grid.nz)
    
    if is_core:
        cell = _build_core_cell(i, j, k, state)
    else:
        cell = _build_ghost_cell(i, j, k, state)
        
    _CELL_CACHE[coord] = cell
    return cell

def _build_core_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Creates a View-based Cell (Logic Wiring).
    Uses local caching of SSoT pointers for high-performance access.
    """
    # 1. Local caching of SSoT containers (Rule 4)
    grid = state.grid
    fields = state.fields
    init = state.initial_conditions
    mask_grid = state.mask.mask

    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    # 1-based indexing to account for ghost halo
    index = (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))
    
    cell = Cell(index=index, fields_buffer=fields.data, is_ghost=False)
    
    if DEBUG:
        print(f"DEBUG [Step 2.1]: Core Cell Created -> Index: {index} at ({i}, {j}, {k})")
    
    # 4. Initialize physical fields and topological mask (Rule 9)
    # The Cell writes directly to the NumPy-managed Foundation buffer.
    cell.vx, cell.vy, cell.vz = init.velocity
    cell.p = init.pressure

    # Temporary Diagnostic
    print(f"DEBUG: Mapping loop indices ({i}, {j}, {k}) to mask shape {mask_grid.shape}")

    cell.mask = int(mask_grid[i, j, k])
    
    return cell

def _build_ghost_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Creates a View-based virtual cell on the perimeter with valid buffer indexing.
    Provides unique memory slots for boundary values.
    """
    grid = state.grid
    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    
    # Calculate index with ghost padding
    index = (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))
    
    cell = Cell(index=index, fields_buffer=state.fields.data, is_ghost=True)
    
    if DEBUG:
        print(f"DEBUG [Step 2.1]: Ghost Cell Created -> ({i}, {j}, {k}) at Index: {index}")
    
    # RULE 5: Explicitly zero all fields to ensure a clean state for the Poisson solver.
    # Using explicit constants for boundary initialization
    cell.vx, cell.vy, cell.vz = GHOST_VELOCITY
    cell.p = GHOST_PRESSURE
    cell.mask = GHOST_MASK
    
    return cell

def clear_cell_cache():
    """Utility to reset cache between simulation steps."""
    _CELL_CACHE.clear()