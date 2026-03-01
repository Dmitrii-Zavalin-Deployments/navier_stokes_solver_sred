from __future__ import annotations
from typing import Any, Dict
import numpy as np
from src.solver_input import SolverInput

from src.solver_state import (
    SolverState, 
    SolverConfig, 
    GridContext, 
    FieldData, 
    MaskData,
    FluidProperties
)

def assemble_simulation_state(
    config_raw: SolverInput,
    grid_raw: GridInput,
    fields: Dict[str, np.ndarray],
    mask: list,
    constants: Dict[str, Any],
    is_fluid: np.ndarray,
    is_boundary_cell: np.ndarray,
    boundary_conditions: Dict[str, Any],
    **kwargs
) -> SolverState:
    """
    Final Hardened Step 1 Assembly.
    Strictly adheres to Phase C Zero-Default and SSoT Mandates.
    """

    # 1. Map Config - EXPLICIT OR ERROR
    # Note: solver_settings must be present in the config_raw
    # NOTE: Run float64 conversion for the numbers before running the calculator
    solver_settings = config_raw.solver_settings
    
    config_obj = SolverConfig(
        _simulation_parameters=config_raw.simulation_parameters,
        _time_step=float(config_raw.simulation_parameters.time_step),
        _boundary_conditions=config_raw.boundary_conditions,
        _external_forces=[float(x) for x in config_raw.external_forces.force_vector],
        _initial_conditions=config_raw.initial_conditions,
        _fluid_properties=config_raw.fluid_properties,
        _ppe_tolerance=solver_settings.ppe_tolerance,
        _ppe_atol=solver_settings.ppe_atol,
        _ppe_max_iter=solver_settings.ppe_max_iter
    )

    # 2. Map Grid Context - Geometric SSoT
    grid_obj = GridContext(
        _nx=grid_raw.nx, _ny=grid_raw.ny, _nz=grid_raw.nz,
        _x_min=grid_raw.x_min, _x_max=grid_raw.x_max,
        _y_min=grid_raw.y_min, _y_max=grid_raw.y_max,
        _z_min=grid_raw.z_min, _z_max=grid_raw.z_max,
        _dx=grid_raw.dx, _dy=grid_raw.dy, _dz=grid_raw.dz
    )

    # 3. Field Data - Physical SSoT
    fields_obj = FieldData(
        _P=fields["P"], _U=fields["U"], _V=fields["V"], _W=fields["W"]
    )

    # 4. Mask Data
    masks_obj = MaskData(
        _mask=mask,
        _is_fluid=is_fluid,
        _is_boundary=is_boundary_cell
    )

    # 5. Fluid Properties - Physical Context
    # Constants derived in compute_derived_constants are injected here
    fluid_obj = FluidProperties(
        _rho=constants["rho"],
        _mu=constants["mu"]
    )

    # 6. Master Assembly
    state = SolverState(
        config=config_obj,
        grid=grid_obj,
        fields=fields_obj,
        masks=masks_obj,
        fluid=fluid_obj
    )

    # 7. Metadata Injection - No defaults (kwargs MUST contain these)
    # If these keys are missing, the KeyError is intended (Deterministic Policy)
    state.iteration = int(kwargs["iteration"])
    state.time = float(kwargs["time"])
    state.ready_for_time_loop = bool(kwargs["ready_for_time_loop"])
    
    # Non-containerized metadata (Constants used for math scaling)
    state.constants = constants
    state.boundary_conditions = boundary_conditions

    return state
