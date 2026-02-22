# tests/helpers/solver_output_schema_dummy.py

# Import the base Step 4 dummy to satisfy the composition
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

def make_output_schema_dummy(nx=4, ny=4, nz=4):
    """
    The 'Gold Standard' State. 
    This represents a SolverState that has passed through all 5 steps.
    
    Updates:
    - Included step5_diagnostics to satisfy 'The Archivist' trigger logic tests.
    - Synchronized iteration with output_interval to ensure a snapshot trigger.
    """
    # 1. Start from Step 4 (The most complete physical state)
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)

    # 2. Simulate a "Trigger" iteration (e.g., iteration 10 with interval 10)
    # This ensures (iteration % interval == 0) is True for the test logic.
    state.iteration = 10 
    interval = state.simulation_parameters.get("output_interval", 10)

    # 3. Add Step 5 Archivist Receipt (Required for Property Integrity tests)
    state.step5_diagnostics = {
        "snapshot_generated": (state.iteration % interval == 0),
        "write_success": True,
        "archive_path": "outputs/snapshot_010.json"
    }

    # 4. Add 'Step 5' Voice (Export Metadata)
    # This aligns with the 'step5_outputs' key in our schema
    state.step5_outputs = {
        "last_file_saved": "output_0010.vtk",
        "export_format": "VTK",
        "simulation_status": "COMPLETE",
        "total_runtime_seconds": 1.25
    }

    return state