# tests/helpers/solver_output_schema_dummy.py

# Import the base Step 4 dummy to satisfy the composition
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

def make_output_schema_dummy(nx=4, ny=4, nz=4):
    """
    The 'Gold Standard' State. 
    This represents a SolverState that has passed through all 5 steps.
    """
    # 1. Start from Step 4 (The most complete physical state)
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)

    # 2. Add 'Step 5' Voice (Export Metadata)
    # This aligns with the 'step5_outputs' key in our schema
    state.step5_outputs = {
        "last_file_saved": "output_0001.vtk",
        "export_format": "VTK",
        "simulation_status": "COMPLETE",
        "total_runtime_seconds": 1.25
    }

    return state