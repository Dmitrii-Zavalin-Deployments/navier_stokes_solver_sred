# src/step5/archivist.py

import os
from src.solver_state import SolverState

def record_snapshot(state: SolverState) -> None:
    """
    Point 2: Create artifacts and update the Manifest.
    Rule 5: Explicit or Error. Directory is derived from config.
    """
    out_dir = state.config.output_directory
    
    # Ensure directory exists (Point 2: Create folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Generate standard file names
    snap_name = f"snapshot_{state.iteration:04d}.vtk"
    snap_path = os.path.join(out_dir, snap_name)
    
    # Point 2: Write "Artifacts" (Placeholder for real VTK export)
    with open(snap_path, "w") as f:
        f.write(f"Step: {state.iteration}\nTime: {state.time}")

    # Point 3: Update Manifest Safe
    state.manifest.output_directory = out_dir
    if snap_path not in state.manifest.saved_snapshots:
        state.manifest.saved_snapshots.append(snap_path)
    
    state.manifest.log_file = os.path.join(out_dir, "solver_convergence.log")