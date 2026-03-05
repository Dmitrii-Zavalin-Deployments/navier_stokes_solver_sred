# tests/scientific/test_scientific_step5_archivist.py

import os
import pytest
import shutil
import tempfile
from unittest.mock import patch
from src.solver_state import SolverState
from src.step5.archivist import record_snapshot

@pytest.fixture
def state_for_archiving():
    """
    Sets up a state with isolated paths for file IO validation.
    Injects a temporary directory to avoid 'output/default_case' persistence.
    """
    state = SolverState()
    
    # Create a unique temporary directory for this specific test
    tmp_base = tempfile.mkdtemp()
    
    # Setup Config via internal slots to satisfy the ValidatedContainer
    state.config._case_name = "scientific_test_run"
    
    # Scientific Metadata for VTK formulas
    state.grid._nx, state.grid._ny, state.grid._nz = 10, 20, 30
    state.iteration = 42
    state.time = 0.123456789
    state.health._divergence_norm = 0.0005678
    
    # Ensure manifest is clean for Step 5 logic
    state.manifest._saved_snapshots = []
    
    # We yield the state and the tmp_base so the test knows where to look
    yield state, tmp_base
    
    # Cleanup: Remove the temporary directory after the test
    if os.path.exists(tmp_base):
        shutil.rmtree(tmp_base)

def test_archivist_directory_creation(state_for_archiving):
    """Scientific check: Verifies directory recursion and path SSoT."""
    state, tmp_base = state_for_archiving
    
    # Use patch to override the hardcoded 'output' property in SolverConfig
    with patch("src.solver_state.SolverConfig.output_directory", tmp_base):
        record_snapshot(state)
        
        expected_path = os.path.join(tmp_base, "scientific_test_run")
        
        assert os.path.exists(expected_path), f"Archivist failed to create {expected_path}"
        assert state.manifest.output_directory == expected_path

def test_archivist_vtk_header_formula(state_for_archiving):
    """Scientific check: Verifies VTK header string formatting and precision."""
    state, tmp_base = state_for_archiving
    
    with patch("src.solver_state.SolverConfig.output_directory", tmp_base):
        record_snapshot(state)
        
        snap_path = state.manifest.saved_snapshots[0]
        
        with open(snap_path, "r") as f:
            content = f.read()
            
        # 1. Check Grid Topology Formula: DIMENSIONS nx ny nz
        assert "DIMENSIONS 10 20 30" in content
        
        # 2. Check Temporal Precision Formula: TIME={state.time:.6f}
        # 0.123456789 rounded to 6 decimal places is 0.123457
        assert "TIME=0.123457" in content
        
        # 3. Check Stability Metric Formula: DIV_NORM={state.health.divergence_norm:.2e}
        # 0.0005678 in scientific notation is 5.68e-04
        assert "DIV_NORM=5.68e-04" in content

def test_archivist_debug_handshake(state_for_archiving, capsys):
    """Scientific check: Verifies all debug signals including directory creation."""
    state, tmp_base = state_for_archiving
    
    with patch("src.solver_state.SolverConfig.output_directory", tmp_base):
        record_snapshot(state)
        
        captured = capsys.readouterr().out
        
        # These signals must appear in sequence for a valid Step 5 handshake
        assert "DEBUG [Step 5 Archivist]: Preparing output" in captured
        assert "DEBUG [Step 5 Archivist]: Created directory" in captured
        assert "DEBUG [Step 5 Archivist]: Snapshot saved" in captured
        assert "DEBUG [Step 5 Archivist]: Manifest updated" in captured