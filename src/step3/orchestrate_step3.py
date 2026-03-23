from src.common.simulation_context import SimulationContext
from src.common.stencil_block import StencilBlock
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ops.ghost_handler import sync_ghost_trial_buffers
from src.step3.ppe_solver import solve_pressure_poisson_step
from src.step3.predictor import compute_local_predictor_step

# Rule 8: Granular Sub-module Access
from src.step3.boundaries.applier import apply_boundary_values
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs

# Rule 7: Granular Traceability for GitHub Actions
DEBUG = False

def orchestrate_step3(
    block: StencilBlock, 
    context: SimulationContext, 
    state_grid: object,
    state_bc_manager: object,
    is_first_pass: bool = False
) -> tuple[StencilBlock, float]:
    """
    Step 3 Orchestrator: Projection Method Kernel with Integrated Boundaries.
    
    Compliance:
    - Rule 0 (Performance): Atomic execution of Predictor + Boundary Enforcement.
    - Rule 4 (SSoT): Uses state-level grid and bc_manager for rule dispatching.
    - Rule 9 (Hybrid Memory): Direct mutation of Foundation buffers via sub-modules.
    """
    
    # --- [A] GHOST SYNC PATH ---
    if block.center.is_ghost:
        if DEBUG:
            print(f"[DEBUG] Ghost Sync at Index {block.center.index}")
            
        sync_ghost_trial_buffers(block)
        return block, 0.0
    
    # --- [B] PHYSICS KERNEL PATH ---
    
    # 1. PREDICT & ENFORCE (Atomic Unit)
    if is_first_pass:
        # A. Intermediate star-velocity calculation (u*)
        compute_local_predictor_step(block)
        
        # B. Identify and Apply Boundary Rules immediately
        # This ensures 1e15 is in the field BEFORE the Poisson solver runs.
        rules = get_applicable_boundary_configs(
            block, 
            state_bc_manager.to_dict(),
            state_grid, 
            context.input_data.domain_configuration.to_dict()
        )
        
        for rule in rules:
            apply_boundary_values(block, rule)
            
        return block, 0.0

    # 2. SOLVE: Iterative Pressure Poisson (SOR)
    # The solver now encounters the massive gradient from the BC, triggering instability signal.
    delta = solve_pressure_poisson_step(block, context.config.ppe_omega)

    # 3. CORRECT: Final velocity projection (u_next)
    apply_local_velocity_correction(block)

    return block, delta