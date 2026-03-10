# src/step1/orchestrate_step1.py

from __future__ import annotations

import numpy as np

from src.common.solver_state import (
    BoundaryCondition,
    BoundaryConditionManager,
    DomainManager,
    ExternalForceManager,
    FluidPropertiesManager,
    GridManager,
    InitialConditionManager,
    MaskManager,
    SimulationParameterManager,
    SolverState,
)
from src.solver_input import SolverInput

from .helpers import generate_3d_masks


def orchestrate_step1(input_data: SolverInput) -> SolverState:
    """
    Direct Ingestion Orchestrator (Phase C Compliant).
    Zero-Debt Policy: State is initialized to the defaults defined in SolverState.
    """
    state = SolverState()

    # --- 1. Geometric Context (Grid) ---
    g = input_data.grid
    state.grid = GridManager(
        _x_min=float(g.x_min), _x_max=float(g.x_max),
        _y_min=float(g.y_min), _y_max=float(g.y_max),
        _z_min=float(g.z_min), _z_max=float(g.z_max),
        _nx=int(g.nx), _ny=int(g.ny), _nz=int(g.nz)
    )
    
    # --- 2. Domain & Physical Context ---
    state.domain = DomainManager(
        type=str(input_data.domain.type), 
        reference_velocity=np.array(input_data.domain.reference_velocity, dtype=np.float64)
    )

    state.fluid = FluidPropertiesManager(
        density=float(input_data.fluid_properties.density),
        viscosity=float(input_data.fluid_properties.viscosity)
    )
    
    state.external_forces = ExternalForceManager(
        force_vector=np.array(input_data.external_forces.force_vector, dtype=np.float64)
    )

    # --- 3. Initial Conditions ---
    state.initial_conditions = InitialConditionManager(
        velocity=np.array(input_data.initial_conditions.velocity, dtype=np.float64),
        pressure=float(input_data.initial_conditions.pressure)
    )

    # --- 4. Simulation Parameters ---
    state.sim_params = SimulationParameterManager(
        time_step=float(input_data.simulation_parameters.time_step),
        total_time=float(input_data.simulation_parameters.total_time),
        output_interval=int(input_data.simulation_parameters.output_interval)
    )

    # --- 5. Topology & Geometry ---
    mask_3d, _, _ = generate_3d_masks(input_data.mask.data, g)
    state.masks = MaskManager(_mask=mask_3d)

    # --- 6. Boundary Conditions ---
    bc_objs = [
        BoundaryCondition(_location=str(item.location), _type=str(item.type), _values=dict(item.values))
        for item in input_data.boundary_conditions.items
    ]
    state.boundary_conditions = BoundaryConditionManager(conditions=bc_objs)

    return state