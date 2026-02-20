# tests/step2/test_laplacian_100_percent.py

import pytest
import numpy as np
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.solver_state import SolverState

def test_laplacian_explicit_none_bc_coverage():
    """
    Specifically targets lines 36-40 in build_laplacian_operators.py.
    Forces the 'if state.boundary_conditions is None' branch to execute.
    """
    # 1. Manual setup to bypass 'make_state' helpers
    grid = {'nx': 2, 'ny': 2, 'nz': 2, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0}
    
    # 2. Force the None condition
    state = SolverState(grid=grid, boundary_conditions=None)
    
    # 3. Minimum requirement for the loop
    state.is_fluid = np.ones((2, 2, 2), dtype=bool)
    
    # 4. This triggers the fallback: bc_table = {}
    build_laplacian_operators(state)
    
    # 5. Assertions to prove success
    assert "laplacian" in state.operators
    # A 2x2x2 grid has 8 cells
    assert state.operators["laplacian"].shape == (8, 8)

def test_laplacian_dirichlet_branch_coverage():
    """
    Ensures the Dirichlet logic (lines 56-65) is fully exercised 
    with a specific pressure BC.
    """
    grid = {'nx': 2, 'ny': 2, 'nz': 2, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0}
    bcs = {"x_max": {"type": "pressure", "p": 1.0}}
    state = SolverState(grid=grid, boundary_conditions=bcs)
    state.is_fluid = np.ones((2, 2, 2), dtype=bool)
    
    build_laplacian_operators(state)
    
    lap = state.operators["laplacian"].toarray()
    # Index for x_max on 2x2x2: i=1, j=0, k=0 -> Index 1
    # Dirichlet row should have 1.0 on diagonal
    assert lap[1, 1] == 1.0