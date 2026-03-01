import json
import shutil
import os
import logging
from pathlib import Path

from src.solver_state import SolverState
from src.solver_input import SolverInput
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5

logger = logging.getLogger(__name__)

def run_solver_from_file(input_path: str) -> str:
    """
    The Master Controller.
    Enforces the 5-Step Pipeline alignment from Phase A contracts to Phase C artifacts.
    """
    # 1. Triage Gate: Validate Schema
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Contract Violation: Input file missing at {input_path}")

    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    
    # Rule 5: Explicit Initialization via the Grader Class
    input_container = SolverInput.from_dict(raw_data)

    # 2. Pipeline Phase 1: Setup (Steps 1 & 2)
    # These are executed once to build the mathematical machine.
    state = orchestrate_step1(input_container)
    state = orchestrate_step2(state)

    # 3. Pipeline Phase 2: Execution (The Iterative Loop)
    # Rule: The loop persists until Step 5 flips ready_for_time_loop to False.
    logger.info(f"Starting Simulation: Target Time = {state.config.total_time}s")
    
    while state.ready_for_time_loop:
        # Step 3: Physical Prediction, PPE Solve, and Correction
        state = orchestrate_step3(state)
        
        # Step 4: Ghost Padding & Boundary Enforcement
        state = orchestrate_step4(state)
        
        # Step 5: Temporal Sync, I/O, and Termination Check
        state = orchestrate_step5(state)
        
        if state.iteration % 10 == 0:
            logger.info(f"Iteration {state.iteration}: Time = {state.time:.4f}s")

    # 4. Pipeline Phase 3: Archiving
    return archive_simulation_artifacts(state)

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Rule 4: SSoT Archiving. Pulls exclusively from state.manifest.
    """
    base_dir = Path("data/testing-input-output")
    zip_path_no_ext = base_dir / "navier-stokes-output"
    
    # Use the directory defined in the state during Step 5
    source_dir = Path(state.manifest.output_directory)
    
    # Save the 'Final Brain Snapshot' before zipping
    state_json_path = source_dir / "final_state_snapshot.json"
    with open(state_json_path, "w") as f:
        json.dump(state.to_json_safe(), f, indent=2)

    # Create the high-ratio archive
    archive_full_path = shutil.make_archive(
        base_name=str(zip_path_no_ext), 
        format='zip', 
        root_dir=source_dir
    )
    
    # Clean up the raw directory, leaving only the ZIP
    if source_dir.exists():
        shutil.rmtree(source_dir)
        
    return archive_full_path