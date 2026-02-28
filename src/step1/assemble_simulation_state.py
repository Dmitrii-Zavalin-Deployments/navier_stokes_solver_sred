# src/step1/assemble_simulation_state.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np

from src.solver_state import (
    SolverState, 
    SolverConfig, 
    GridContext, 
    FieldData, 
    MaskData,
    FluidProperties
)

def assemble_simulation_state(
    config: Dict[str, Any],
    grid: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    mask: np.ndarray,
    is_fluid: np.ndarray,
    is_boundary_cell: np.ndarray,
    **kwargs
) -> SolverState:
    """
    Final Hardened Step 1 Assembly.
    
    ZERO-DEFAULT POLICY (Phase C.5):
    - No Logical Defaults: iteration and time are pulled from kwargs.
    - No Structural Fallbacks: All config/grid keys are accessed via [key].
    - Constructor Integrity: FluidProperties is built before state assembly.
    """

    # 1. Map Config Safe (Explicit Access - Raises KeyError if missing)
    config_obj = SolverConfig(
        _simulation_parameters=config["simulation_parameters"],
        _boundary_conditions=config["boundary_conditions"],
        _external_forces=config["external_forces"],
        _initial_conditions=config["initial_conditions"],
        _fluid_properties=config["fluid_properties"],
        _ppe_tolerance=config["ppe_tolerance"],
        _ppe_atol=config["ppe_atol"],
        _ppe_max_iter=config["ppe_max_iter"]
    )

    # 2. Map Grid Context (No fallback spacing)
    grid_obj = GridContext(
        _nx=grid["nx"], _ny=grid["ny"], _nz=grid["nz"],
        _x_min=grid["x_min"], _x_max=grid["x_max"],
        _y_min=grid["y_min"], _y_max=grid["y_max"],
        _z_min=grid["z_min"], _z_max=grid["z_max"]
    )

    # 3. Initialize Fields (Strict mapping to private slots)
    field_internal = {f"_{k}": v for k, v in fields.items()}
    fields_obj = FieldData(**field_internal)

    # 4. Initialize Masks (Zero-detour data transfer)
    masks_obj = MaskData(
        _mask=mask,
        _is_fluid=is_fluid,
        _is_boundary=is_boundary_cell
    )

    # 5. Initialize Fluid Safe (Constructor Integrity Point 5.3)
    # We pull directly from the config object to ensure SSoT alignment
    fluid_obj = FluidProperties(
        _rho=config_obj.density,
        _mu=config_obj.viscosity
    )

    # 6. Assemble the Master Container
    # Every department is now a pre-validated, fully-populated object.
    state = SolverState(
        config=config_obj,
        grid=grid_obj,
        fields=fields_obj,
        masks=masks_obj,
        fluid=fluid_obj
    )

    # 7. Strict Odometer & Execution Gate (Point 5.1)
    # Must be explicitly provided by the orchestrator (e.g. from JSON or Checkpoint)
    state.iteration = int(kwargs["iteration"])
    state.time = float(kwargs["time"])
    state.ready_for_time_loop = bool(kwargs["ready_for_time_loop"])

    return state