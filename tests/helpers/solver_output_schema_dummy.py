# tests/helpers/solver_output_schema_dummy.py

import numpy as np
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

def make_output_schema_dummy(nx=4, ny=4, nz=4):
    """
    The 'Gold Standard' State: Post-Step 5 Completion.
    
    Constitutional Role:
    - Chronos Guard: Synchronizes state.time to total_time.
    - Archivist: Populates the OutputManifest with file paths and logs.
    - Termination: Flips ready_for_time_loop to False (Finished).
    """
    # 1. Start from Step 4 (The Physical Foundation)
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # 2. Terminal Temporal State (From orchestrate_step5_state)
    # ------------------------------------------------------------------
    # We simulate a completed run reaching total_time
    target_total_time = 1.0
    state.time = target_total_time
    state.iteration = 1000 
    
    # In Step 5, we call it 'step_index' based on your source code
    state.step_index = state.iteration 

    # ------------------------------------------------------------------
    # 3. Populate Manifest Safe (The Archivist)
    # ------------------------------------------------------------------
    # This reflects the results of write_output_snapshot and finalize_health
    state.manifest.output_directory = "output/simulation_results"
    state.manifest.saved_snapshots = [
        "output/snapshot_0000.vtk",
        "output/snapshot_0500.vtk",
        f"output/snapshot_{state.iteration:04d}.vtk"
    ]
    state.manifest.final_checkpoint = f"output/checkpoint_final_{state.iteration}.npy"
    state.manifest.log_file = "output/solver_convergence.log"

    # ------------------------------------------------------------------
    # 4. Final Health Summary (from finalize_simulation_health)
    # ------------------------------------------------------------------
    state.health.is_stable = True
    state.health.post_correction_divergence_norm = 1e-15
    # Verification that the final speed is within physical bounds
    state.health.max_u = 1.2 

    # ------------------------------------------------------------------
    # 5. Progression Gate
    # ------------------------------------------------------------------
    # The loop is finished; we are no longer "ready" to enter it again.
    state.ready_for_time_loop = False

    return state