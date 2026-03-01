# src/step1/orchestrate_step1.py

from __future__ import annotations
import logging

from src.solver_state import SolverState
from src.solver_input import SolverInput  # The Typed Input Contract
from .parse_config import parse_config
from .initialize_grid import initialize_grid
from .allocate_fields import allocate_fields
from .map_geometry_mask import map_geometry_mask
from .parse_boundary_conditions import parse_boundary_conditions
from .compute_derived_constants import compute_derived_constants
from .validate_physical_constraints import validate_physical_constraints
from .assemble_simulation_state import assemble_simulation_state
from .apply_initial_conditions import apply_initial_conditions

logger = logging.getLogger(__name__)

# Constitutional Toggle: Set to False in production to avoid I/O overhead
DEBUG_STEP1 = True

def debug_state_step1(state_obj: SolverState) -> None:
    """
    Prints a summary using the SolverState object attributes.
    Ensures the SSoT containers are correctly populated.
    """
    print("\n" + "="*20 + " DEBUG: STEP-1 STATE SUMMARY " + "="*20)
    containers = ["grid", "fields", "masks", "fluid", "config"]
    for attr in containers:
        value = getattr(state_obj, attr, None)
        print(f"\n• {attr.upper()}: {type(value)}")
        
        if attr == "fields" and value:
            print(f"    U shape: {value.U.shape}, V shape: {value.V.shape}, W shape: {value.W.shape}")
        elif attr == "grid" and value:
            print(f"    Topology: {value.nx}x{value.ny}x{value.nz}")
        elif attr == "fluid" and value:
            print(f"    Physics: rho={value.rho}, mu={value.mu}")
            
    print("\n" + "="*69 + "\n")

def orchestrate_step1(
    input_data: SolverInput,
    **kwargs,
) -> SolverState:
    """
    Main entry point for Step 1. 
    Transforms the validated SolverInput object into a high-fidelity SolverState.
    
    Constitutional Role: The Memory Architect.
    Mandate: Hydrate the SSoT (Single Source of Truth) from the Triaged Input.
    """

    # 1. Spatial Governor (Grid Context)
    # Passed as the specific GridInput sub-container
    grid = initialize_grid(input_data.grid)
    
    # 2. Config Context (Solver Tuning)
    # We pass the full object; internal logic extracts what it needs
    parse_config(input_data)

    # 3. Memory Architect (Staggered Field Allocation)
    fields = allocate_fields(grid)
    
    # 4. Field Primer (Initial Conditions)
    # Uses InitialConditionsInput sub-container
    apply_initial_conditions(fields, input_data.initial_conditions)

    # 5. Topology Interpreter (Masks & Boundaries)
    # Mask input is a validated list; BCs are a list of BoundaryConditionItems
    mask, is_fluid, is_boundary_cell = map_geometry_mask(input_data.mask.data, input_data.grid)
    bc_table = parse_boundary_conditions(input_data.boundary_conditions.items, grid)

    # 6. Mathematical Translator (Physical Constants)
    # Pass Typed objects for density, viscosity, dt, etc.
    constants = compute_derived_constants(
        grid, 
        input_data.fluid_properties, 
        input_data.simulation_parameters
    )

    # 7. Synthesis Hub (Assembly into SSoT Hierarchy)
    # Note: assemble_simulation_state may still want a dict for the 'config_raw' field,
    # so we use input_data.to_dict() to satisfy that requirement.
    state = assemble_simulation_state(
        config_raw=input_data.to_dict(),
        grid_raw=grid,
        fields=fields,
        mask=mask,
        constants=constants,
        boundary_conditions=bc_table if bc_table else {},
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell,
        iteration=kwargs.get("iteration", 0),
        time=kwargs.get("time", 0.0),
        ready_for_time_loop=kwargs.get("ready_for_time_loop", False)
    )

    # 8. Logical Firewall (Final Physical Sanity Check)
    validate_physical_constraints(state)

    # 9. Debug Hook
    if DEBUG_STEP1:
        debug_state_step1(state)

    return state