# tests/scientific/test_scientific_step5_archivist.py

import os
import pytest
import shutil
import tempfile
import numpy as np
from src.solver_state import SolverState
from src.step5.archivist import record_snapshot

@pytest.fixture
def state_for_archiving():
    """Sets up a state with specific metadata for file validation."""
    state = SolverState()
    
    # Create a unique temporary directory for THIS specific test instance
    tmp_dir = tempfile.mkdtemp()
    
    # Use leading underscores to bypass potential validation logic in setters
    state.config._output_directory = tmp_dir
    state.config._case_name = "scientific_test_run"
    
    # Scientific Metadata
    state.grid._nx, state.grid._ny, state.grid._nz = 10, 20, 30
    state.iteration = 42
    state.time = 0.123456789
    state.health._divergence_norm = 0.0005678
    
    # Initialize Manifest
    state.manifest._saved_snapshots = []
    
    yield state
    
    # Cleanup after test
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

def test_archivist_directory_creation(state_for_archiving):
    """Scientific check: Verifies OS directory recursion for the specific case."""
    record_snapshot(state_for_archiving)
    
    # Construct expected path using the dynamic temp dir
    expected_path = os.path.join(state_for_archiving.config.output_directory, "scientific_test_run")
    
    assert os.path.exists(expected_path), f"Directory {expected_path} was not created"
    assert state_for_archiving.manifest.output_directory == expected_path

def test_archivist_vtk_header_formula(state_for_archiving):
    """Verifies VTK header formula and scientific formatting."""
    record_snapshot(state_for_archiving)
    
    snap_path = state_for_archiving.manifest.saved_snapshots[0]
    
    with open(snap_path, "r") as f:
        content = f.read()
        
    assert "DIMENSIONS 10 20 30" in content
    assert "TIME=0.123457" in content  # .6f rounding
    assert "DIV_NORM=5.68e-04" in content # .2e scientific

def test_archivist_debug_handshake(state_for_archiving, capsys):
    """Scientific check: Verifies all debug signals, ensuring directory creation is logged."""
    # The fixture provides a fresh, empty temp directory, ensuring 'Created directory' triggers
    record_snapshot(state_for_archiving)
    
    captured = capsys.readouterr().out
    
    assert "DEBUG [Step 5 Archivist]: Preparing output" in captured
    assert "DEBUG [Step 5 Archivist]: Created directory" in captured
    assert "DEBUG [Step 5 Archivist]: Snapshot saved" in captured