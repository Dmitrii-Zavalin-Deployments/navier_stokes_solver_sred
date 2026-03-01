# src/main_solver.py

import json
import shutil
import os
import logging
from pathlib import Path

from src.solver_state import SolverState
from src.solver_input import SolverInput  # The Grader Class
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5

# Setup basic logging to help debug orchestration failures
logger = logging.getLogger(__name__)

def run_solver_from_file(input_path: str) -> str:
    """
    Entry point for the full pipeline. 
    Uses SolverInput as a Triage Gate before entering the 5-Step Pipeline.
    """
    # Edge Case: File Existence
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CRITICAL: Input file not found at {input_path}")

    # Point 1: Load Raw JSON
    try:
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"CRITICAL: Failed to parse input JSON: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading input file: {str(e)}")

    # Point 2: THE TRIAGE GATE
    # We populate the class to validate physics/schema constraints.
    try:
        logger.info("Triaging input data against SolverInput schema...")
        input_container = SolverInput.from_dict(raw_data)
    except Exception as e:
        logger.error(f"INPUT VALIDATION FAILED: {str(e)}")
        raise ValueError(f"CRITICAL: Input data failed triage: {str(e)}")

    # Point 3: Execute 5-Step Pipeline
    try:
        logger.info("Starting Navier-Stokes Orchestration Pipeline...")
        
        # Step 1: Receives the Typed Object instead of a raw dict
        state = orchestrate_step1(input_container)
        
        # Step 2-5: Pipeline progression
        state = orchestrate_step2(state)
        state = orchestrate_step3(state, current_time=0.0, step_index=0)
        state = orchestrate_step4(state)
        state = orchestrate_step5(state)
        
        logger.info("Orchestration completed successfully.")

    except Exception as e:
        logger.error(f"PIPELINE FAILURE: {str(e)}")
        raise RuntimeError(f"Solver Pipeline crashed: {str(e)}")

    # Point 5: Archive results
    try:
        return archive_simulation_artifacts(state)
    except Exception as e:
        raise RuntimeError(f"ARCHIVE FAILURE: Simulation finished but export failed: {str(e)}")

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Handles directory management and final state serialization.
    """
    base_dir = Path("data/testing-input-output")
    output_dir = base_dir / "navier-stokes-output"
    zip_path_no_ext = base_dir / "navier-stokes-output"
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Move snapshots
    if hasattr(state, 'manifest') and state.manifest.saved_snapshots:
        for snapshot_path in state.manifest.saved_snapshots:
            src = Path(snapshot_path)
            if src.exists():
                shutil.copy2(src, output_dir / src.name)

    # Final Brain Snapshot
    state_json_path = output_dir / "final_state_snapshot.json"
    try:
        with open(state_json_path, "w") as f:
            json.dump(state.to_json_safe(), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to serialize final state: {str(e)}")

    # High-ratio compression
    try:
        archive_full_path = shutil.make_archive(
            base_name=str(zip_path_no_ext), 
            format='zip', 
            root_dir=output_dir
        )
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
    
    return archive_full_path