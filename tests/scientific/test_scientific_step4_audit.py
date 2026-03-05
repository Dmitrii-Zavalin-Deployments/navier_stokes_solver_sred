# tests/scientific/test_scientific_step4_audit.py

import pytest
import numpy as np
from src.solver_state import SolverState

class AttributeDict(dict):
    """A dict that allows dot notation access for mocking frozen configs."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

@pytest.fixture
def state_audit():
    """Fixture to mock a state with a fully initialized grid for auditing."""
    state = SolverState()
    
    # 1. Grid Mocking: Set bounds and dimensions
    # L = 1.0, N = 10 -> dx = 0.1
    state.grid._nx, state.grid._ny, state.grid._nz = 10, 10, 10
    state.grid._x_min, state.grid._x_max = 0.0, 1.0
    state.grid._y_min, state.grid._y_max = 0.0, 1.0
    state.grid._z_min, state.grid._z_max = 0.0, 1.0
    
    # Ensure properties are reachable (Crucial for Rule 5)
    setattr(state.grid, 'dx', 0.1)
    setattr(state.grid, 'nx', 10)
    setattr(state.grid, 'ny', 10)
    setattr(state.grid, 'nz', 10)
    
    # 2. Config & Physics 
    state.config._fluid_properties = AttributeDict({
        "density": 1000.0,
        "viscosity": 1.0
    })
    state.config._simulation_parameters = AttributeDict({
        "time_step": 0.001
    })
    
    # Mock top-level properties used by the audit logic
    setattr(state, 'density', 1000.0)
    setattr(state, 'viscosity', 1.0)
    setattr(state, 'dt', 0.001)
    
    # 3. Health/Dynamics (Max velocity)
    state.health.max_u = 2.0
    
    # 4. Diagnostics container
    state.diagnostics = AttributeDict()
    
    return state

def test_audit_memory_calculation(state_audit):
    """Verifies Formula: (nx+2)*(ny+2)*(nz+2) * 8 bytes * 8 fields / 1e9."""
    from src.step4.audit_diagnostics import run_preflight_audit
    
    run_preflight_audit(state_audit)
    
    # Expected: (12^3) * 64 / 1e9 = 1728 * 64 / 1e9 = 0.000110592 GB
    expected_gb = (12 * 12 * 12 * 64) / 1e9
    assert pytest.approx(state_audit.diagnostics.memory_footprint_gb) == expected_gb

def test_audit_cfl_formula(state_audit):
    """Verifies CFL Stability: dt_limit = 0.5 * dx / max_u."""
    from src.step4.audit_diagnostics import run_preflight_audit
    
    # dx=0.1, max_u=2.0 -> limit = 0.5 * 0.1 / 2.0 = 0.025
    run_preflight_audit(state_audit)
    
    assert state_audit.diagnostics.initial_cfl_dt == pytest.approx(0.025)

def test_audit_diffusion_limit_logic(state_audit, capsys):
    """Verifies Diffusion Stability: dt < 0.5 * dx^2 / nu."""
    from src.step4.audit_diagnostics import run_preflight_audit
    
    # dx=0.1, nu = 1/1000 = 0.001
    # limit = 0.5 * (0.01) / 0.001 = 5.0
    run_preflight_audit(state_audit)
    
    captured = capsys.readouterr().out
    assert "Diffusion dt Limit: 5.000000e+00" in captured

def test_audit_fluid_at_rest(state_audit):
    """Verifies that if max_u is 0, CFL defaults to the state's current dt."""
    from src.step4.audit_diagnostics import run_preflight_audit
    
    state_audit.health.max_u = 0.0
    run_preflight_audit(state_audit)
    
    assert state_audit.diagnostics.initial_cfl_dt == 0.001

def test_audit_warning_trigger(state_audit, capsys):
    """Ensures the warning prints if configured dt exceeds stability limits."""
    from src.step4.audit_diagnostics import run_preflight_audit
    
    # Set dt to 1.0, which is > 0.025 (CFL limit)
    setattr(state_audit, 'dt', 1.0)
    
    run_preflight_audit(state_audit)
    
    captured = capsys.readouterr().out
    assert "WARNING" in captured
    assert "exceeds CFL limit" in captured

def test_audit_debug_formatting(state_audit, capsys):
    """Checks for Rule 5 compliance in debug string output."""
    from src.step4.audit_diagnostics import run_preflight_audit
    
    run_preflight_audit(state_audit)
    captured = capsys.readouterr().out
    
    assert "Auditing 10x10x10 grid" in captured
    assert "Memory Footprint" in captured