# tests/helpers/solver_output_schema_dummy.py

import numpy as np
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

def make_output_schema_dummy(nx=4, ny=4, nz=4):
    """
    The 'Gold Standard' State: Post-Step 5 Completion.
    
    Constitutional Role:
    - Chronos Guard: Synchronizes state.time to total_time.
    - Archivist: Populates the OutputManifest via the manifest safe.
    - Termination: Flips ready_for_time_loop to False to signal completion.
    """
    # 1. Start from Step 4 (The Physical & Boundary Foundation)
    # Inherits: grid, operators, fields_ext, and diagnostics.
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # 2. Terminal Temporal State (Reflecting orchestrate_step5_state)
    # ------------------------------------------------------------------
    # We simulate reaching the finish line (total_time = 1.0)
    target_total_time = 1.0
    state.time = target_total_time
    state.iteration = 1000 
    
    # step_index is the loop counter used in the Step 5 while-loop logic.
    # We set it here to ensure any post-processing tests see a finished index.
    if hasattr(state, "step_index"):
        state.step_index = 1000

    # ------------------------------------------------------------------
    # 3. Populate Manifest Safe (The Archivist)
    # ------------------------------------------------------------------
    # These calls now succeed thanks to the @property.setter added to OutputManifest.
    state.manifest.output_directory = "output/simulation_results"
    state.manifest.saved_snapshots = [
        "output/snapshot_0000.vtk",
        "output/snapshot_0500.vtk",
        f"output/snapshot_{state.iteration:04d}.vtk"
    ]
    state.manifest.final_checkpoint = f"output/checkpoint_final_{state.iteration}.npy"
    state.manifest.log_file = "output/solver_convergence.log"

    # ------------------------------------------------------------------
    # 4. Final Health Summary (Simulating finalize_simulation_health)
    # ------------------------------------------------------------------
    state.health.is_stable = True
    state.health.post_correction_divergence_norm = 1e-15
    state.health.max_u = 1.2 

    # ------------------------------------------------------------------
    # 5. Progression Gate
    # ------------------------------------------------------------------
    # Critical: This tells the main_solver that no more iterations are required.
    state.ready_for_time_loop = False

    return state