# src/main_solver.py

import json
import shutil
import os
import logging
from pathlib import Path

from src.solver_state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5

# Setup basic logging to help debug orchestration failures
logger = logging.getLogger(__name__)

def run_solver_from_file(input_path: str) -> str:
    """
    Entry point for the full pipeline with robust error handling.
    Standardized to the Constitutional naming convention.
    """
    # Edge Case: File Existence
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CRITICAL: Input file not found at {input_path}")

    # Point 1: Load Raw JSON with Parse Error Handling
    try:
        with open(input_path, 'r') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"CRITICAL: Failed to parse input JSON: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading input file: {str(e)}")

    # Point 3: Execute 5-Step Pipeline with Global Try-Catch
    try:
        logger.info("Starting Navier-Stokes Orchestration Pipeline...")
        
        # Step 1: Configuration & Allocation
        state = orchestrate_step1(config_data)
        
        # Step 2: Operator Construction
        state = orchestrate_step2(state)
        
        # Step 3: Initial Projection/Health (t=0)
        # Note: We pass current_time and step_index as per contract
        state = orchestrate_step3(state, current_time=0.0, step_index=0)
        
        # Step 4: Boundary Enforcement & Padding
        state = orchestrate_step4(state)
        
        # Step 5: Iterative Loop & Convergence
        state = orchestrate_step5(state)
        
        logger.info("Orchestration completed successfully.")

    except Exception as e:
        logger.error(f"PIPELINE FAILURE during orchestration: {str(e)}")
        # We re-raise to ensure the caller (or test) knows the simulation failed
        raise RuntimeError(f"Solver Pipeline crashed: {str(e)}")

    # Point 5: Archive results with high-ratio compression
    try:
        return archive_simulation_artifacts(state)
    except Exception as e:
        raise RuntimeError(f"ARCHIVE FAILURE: Simulation finished but export failed: {str(e)}")

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Handles directory creation, file movement, JSON state serialization, 
    and high-ratio compression. Handles missing snapshot edge cases.
    """
    base_dir = Path("data/testing-input-output")
    output_dir = base_dir / "navier-stokes-output"
    zip_path_no_ext = base_dir / "navier-stokes-output"
    
    # 1. Clean environment
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Move snapshots mentioned in manifest (Edge Case: Snapshot missing on disk)
    if hasattr(state, 'manifest') and state.manifest.saved_snapshots:
        for snapshot_path in state.manifest.saved_snapshots:
            src = Path(snapshot_path)
            if src.exists():
                shutil.copy2(src, output_dir / src.name)
            else:
                logger.warning(f"Snapshot defined in manifest not found on disk: {snapshot_path}")

    # 3. Serialize State to JSON (The "Brain" of the simulation)
    state_json_path = output_dir / "final_state_snapshot.json"
    try:
        # Use to_json_safe to handle NumPy arrays and non-serializable objects
        with open(state_json_path, "w") as f:
            json.dump(state.to_json_safe(), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to serialize final state: {str(e)}")
        # We still want the zip even if the JSON fails, but we should log it.

    # 4. Zip the artifacts
    # format='zip' creates .zip extension automatically
    try:
        archive_full_path = shutil.make_archive(
            base_name=str(zip_path_no_ext), 
            format='zip', 
            root_dir=output_dir
        )
    finally:
        # Cleanup: Always remove the temp directory, even if zipping fails
        if output_dir.exists():
            shutil.rmtree(output_dir)
    
    return archive_full_path