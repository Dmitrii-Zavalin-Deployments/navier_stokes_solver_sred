# src/main_solver.py

import json
import shutil
import os
from pathlib import Path

from src.solver_state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5_state import orchestrate_step5

from src.common.final_schema_utils import validate_final_state

def run_solver_from_file(input_path: str) -> str:
    """
    Entry point for the full pipeline.
    Standardized to the new Constitutional naming convention.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Point 1: Load Raw JSON
    with open(input_path, 'r') as f:
        config_data = json.load(f)

    # Point 3: Execute 5-Step Pipeline
    # Standardized names: orchestrate_step1 through orchestrate_step5
    state = orchestrate_step1(config_data)
    state = orchestrate_step2(state)
    
    # We call Step 3 once for t=0 initialization health check before the loop
    state = orchestrate_step3(state, current_time=0.0, step_index=0)
    
    state = orchestrate_step4(state)
    state = orchestrate_step5(state)

    # Validation
    validate_final_state(state)

    # Point 5: Archive results
    return archive_simulation_artifacts(state)

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Handles directory creation, file movement, 
    JSON state serialization, and high-ratio compression.
    """
    base_dir = Path("data/testing-input-output")
    output_dir = base_dir / "navier-stokes-output"
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Move snapshots mentioned in manifest
    for snapshot_path in state.manifest.saved_snapshots:
        src = Path(snapshot_path)
        if src.exists():
            shutil.copy2(src, output_dir / src.name)

    # Serialize State to JSON
    state_json_path = output_dir / "final_state_snapshot.json"
    with open(state_json_path, "w") as f:
        json.dump(state.to_json_safe(), f, indent=2)

    # Log sizes and Zip
    total_size = 0
    for f in output_dir.glob("*"):
        total_size += f.stat().st_size
    
    zip_base = str(base_dir / "navier-stokes-output")
    shutil.make_archive(zip_base, 'zip', output_dir)
    
    # Cleanup
    shutil.rmtree(output_dir)
    
    return f"{zip_base}.zip"