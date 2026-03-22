# src/step3/orchestrate_step3.py

from src.common.simulation_context import SimulationContext
from src.common.stencil_block import StencilBlock
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ops.ghost_handler import sync_ghost_trial_buffers
from src.step3.ppe_solver import solve_pressure_poisson_step
from src.step3.predictor import compute_local_predictor_step

# Rule 7: Granular Traceability for GitHub Actions
DEBUG = False

def orchestrate_step3(
    block: StencilBlock, 
    context: SimulationContext, 
    is_first_pass: bool = False
) -> tuple[StencilBlock, float]:
    """
    Step 3 Orchestrator: Projection Method Kernel.
    
    Compliance:
    - Rule 0 (Performance): Stripped of state-management overhead. Operates 
      directly on the block's current state.
    - Rule 4 (SSoT): Assumes block.dt has been synchronized by the Main Solver 
      prior to execution.
    - Rule 7 (Traceability): Bubbles exceptions to the Main Solver for 
      centralized recovery and forensic logging.
    """
    
    # --- [A] GHOST SYNC PATH ---
    if block.center.is_ghost:
        if DEBUG:
            # Rule 7: High-resolution trace for Ghost recovery
            print(f"[DEBUG] Ghost Sync at Index {block.center.index}")
            
        sync_ghost_trial_buffers(block)
        return block, 0.0
    
    # --- [B] PHYSICS KERNEL PATH ---
    
    # 1. PREDICT: Intermediate star-velocity calculation (u*)
    # Uses block.dt implicitly within compute_local_predictor_step
    if is_first_pass:
        compute_local_predictor_step(block)
        return block, 0.0

    # 2. SOLVE: Iterative Pressure Poisson (SOR)
    # Returns the L2-norm delta for convergence checking
    delta = solve_pressure_poisson_step(block, context.config.ppe_omega)

    # 3. CORRECT: Final velocity projection (u_next)
    # Projects star-velocity onto a divergence-free field
    apply_local_velocity_correction(block)

    return block, delta