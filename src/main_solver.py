# src/main_solver.py
import json
import os
import shutil
from typing import Dict, Any
from pathlib import Path

from src.solver_state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1_state
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3_state
from src.step4.orchestrate_step4 import orchestrate_step4_state
from src.step5.orchestrate_step5_state import orchestrate_step5_state

from src.common.final_schema_utils import validate_final_state

def run_solver_from_file(input_path: str) -> str:
    """
    Entry point for the full pipeline.
    Reads JSON, runs solver, archives results.
    Returns the path to the final ZIP artifact.
    """
    # Point 1: Load Raw JSON
    with open(input_path, 'r') as f:
        config_data = json.load(f)

    # Point 3: Execute 5-Step Pipeline
    # Note: Step 1 converts the raw dict into the Constitutional SolverState
    state = orchestrate_step1_state(config_data)
    state = orchestrate_step2(state)
    state = orchestrate_step3_state(state, current_time=0.0, step_index=0)
    state = orchestrate_step4_state(state)
    state = orchestrate_step5_state(state)

    # Validation
    validate_final_state(state)

    # Point 5: Archive results
    return archive_simulation_artifacts(state)

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Handles Point 5.2 through 5.5: Directory creation, file movement, 
    JSON state serialization, and high-ratio compression.
    """
    base_dir = Path("data/testing-input-output")
    output_dir = base_dir / "navier-stokes-output"
    
    # Point 5.2: Create folder
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Point 5.3: Move snapshots mentioned in manifest
    # We assume the solver steps generated these files in 'output/'
    for snapshot_path in state.manifest.saved_snapshots:
        src = Path(snapshot_path)
        if src.exists():
            shutil.copy2(src, output_dir / src.name)

    # Point 5.4: Serialize State to JSON (Smallest memory footprint via compression)
    # We save the 'Truth' of the final state
    state_json_path = output_dir / "final_state_snapshot.json"
    with open(state_json_path, "w") as f:
        json.dump(state.to_json_safe(), f, indent=2)

    # Point 5.5: Log sizes, Zip, and Cleanup
    print(f"\n--- Artifact Audit for Iteration {state.iteration} ---")
    total_size = 0
    for f in output_dir.glob("*"):
        size_kb = f.stat().st_size / 1024
        print(f"File: {f.name} | Size: {size_kb:.2f} KB")
        total_size += f.stat().st_size
    print(f"Total Folder Size: {total_size / 1024:.2f} KB")

    # High-ratio ZIP (Standard lib 'zipfile' with ZIP_DEFLATED is safe/opensource)
    zip_path = base_dir / "navier-stokes-output" # .zip added by make_archive
    shutil.make_archive(str(zip_path), 'zip', output_dir)
    
    # Cleanup: Remove the unzipped folder
    shutil.rmtree(output_dir)
    
    final_zip = base_dir / "navier-stokes-output.zip"
    print(f"Artifact locked: {final_zip}")
    return str(final_zip)