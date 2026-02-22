# tests/helpers/solver_output_schema_dummy.py

# Import the base Step 4 dummy to satisfy the composition
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

def make_output_schema_dummy(nx=4, ny=4, nz=4):
    """
    The 'Gold Standard' State. 
    This represents a SolverState that has passed through all 5 steps.
    
    Updates:
    - Chronos Guard: time synced to total_time (1.0) to satisfy termination tests.
    - Status: ready_for_time_loop set to False to signal completion.
    - Archivist Logic: Step 5 diagnostics included for receipt verification.
    """
    # 1. Start from Step 4 (The most complete physical state)
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)

    # 2. Terminal Temporal State (Fix for The Judge)
    # We ensure time matches the simulation goal to represent a finished run.
    state.simulation_parameters["total_time"] = 1.0
    state.time = 1.0 
    state.iteration = 1000 
    state.ready_for_time_loop = False  # Critical: signals solver exit

    # 3. Simulate a "Trigger" iteration logic
    interval = state.simulation_parameters.get("output_interval", 10)

    # 4. Add Step 5 Archivist Receipt (Required for Property Integrity tests)
    state.step5_diagnostics = {
        "snapshot_generated": (state.iteration % interval == 0),
        "write_success": True,
        "archive_path": f"outputs/snapshot_{state.iteration:03d}.json"
    }

    # 5. Add 'Step 5' Voice (Export Metadata)
    state.step5_outputs = {
        "last_file_saved": f"output_{state.iteration:04d}.vtk",
        "export_format": "VTK",
        "simulation_status": "COMPLETE",
        "total_runtime_seconds": 1.25
    }

    return state