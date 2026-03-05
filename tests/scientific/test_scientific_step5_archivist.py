# tests/scientific/test_scientific_step5_archivist.py

import os
import pytest
import shutil
import tempfile
from src.solver_state import SolverState
from src.step5.archivist import record_snapshot

@pytest.fixture
def temp_output_dir():
    """Provides a clean, temporary directory for file IO tests."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)

@pytest.fixture
def state_for_archiving(temp_output_dir):
    """Sets up a state with specific metadata for file validation."""
    state = SolverState()
    
    # 1. Config for Pathing
    state.config._output_directory = temp_output_dir
    state.config._case_name = "scientific_test_run"
    
    # 2. Scientific Metadata for Formulas
    # We use specific values to verify formatting: state.time:.6f and divergence_norm:.2e
    state.grid._nx, state.grid._ny, state.grid._nz = 10, 20, 30
    state.iteration = 42
    state.time = 0.123456789
    state.health._divergence_norm = 0.0005678
    
    # 3. Manifest initialization
    # Ensure lists are ready for Step 5 compliance
    state.manifest._saved_snapshots = []
    
    return state

def test_archivist_directory_creation(state_for_archiving):
    """Scientific check: Verifies OS directory recursion for the specific case."""
    record_snapshot(state_for_archiving)
    
    expected_path = os.path.join(state_for_archiving.config.output_directory, "scientific_test_run")
    assert os.path.exists(expected_path)
    assert state_for_archiving.manifest.output_directory == expected_path

def test_archivist_vtk_header_formula(state_for_archiving):
    """
    Scientific check: Verifies the VTK header formula:
    DIMENSIONS {nx} {ny} {nz}
    METADATA: TIME={time:.6f}, DIV_NORM={div:.2e}
    """
    record_snapshot(state_for_archiving)
    
    snap_path = state_for_archiving.manifest.saved_snapshots[0]
    
    with open(snap_path, "r") as f:
        content = f.read()
        
    # Verify VTK Structure
    assert "DATASET STRUCTURED_POINTS" in content
    # Verify Grid Formula
    assert "DIMENSIONS 10 20 30" in content
    # Verify Precision Formulas
    assert "TIME=0.123457" in content  # Rounded by .6f
    assert "DIV_NORM=5.68e-04" in content # Formatted by .2e

def test_archivist_manifest_sync(state_for_archiving):
    """Scientific check: Verifies SSoT synchronization between disk and state."""
    record_snapshot(state_for_archiving)
    
    # Rule 5: Check that the snap_path is actually stored in the manifest
    assert len(state_for_archiving.manifest.saved_snapshots) == 1
    assert "snapshot_0042.vtk" in state_for_archiving.manifest.saved_snapshots[0]
    assert state_for_archiving.manifest.log_file.endswith("solver_convergence.log")

def test_archivist_debug_handshake(state_for_archiving, capsys):
    """Scientific check: Verifies all debug signals for the Step 5 handshake."""
    record_snapshot(state_for_archiving)
    
    captured = capsys.readouterr().out
    
    assert "DEBUG [Step 5 Archivist]: Preparing output" in captured
    assert "DEBUG [Step 5 Archivist]: Created directory" in captured
    assert "DEBUG [Step 5 Archivist]: Snapshot saved" in captured
    assert "DEBUG [Step 5 Archivist]: Manifest updated" in captured