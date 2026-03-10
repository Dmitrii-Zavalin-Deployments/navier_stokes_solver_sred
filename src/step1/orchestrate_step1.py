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
    Uses official setters to ensure all data is validated at the gate.
    """
    state = SolverState()

    # --- 1. Grid & Domain ---
    state.grid = GridManager()
    g = input_data.grid
    state.grid.x_min, state.grid.x_max = float(g.x_min), float(g.x_max)
    state.grid.y_min, state.grid.y_max = float(g.y_min), float(g.y_max)
    state.grid.z_min, state.grid.z_max = float(g.z_min), float(g.z_max)
    state.grid.nx, state.grid.ny, state.grid.nz = int(g.nx), int(g.ny), int(g.nz)
    
    state.domain = DomainManager()
    state.domain.type = str(input_data.domain.type)
    state.domain.reference_velocity = np.array(input_data.domain.reference_velocity, dtype=np.float64)

    # --- 2. Physical Context (Fluid & Forces) ---
    state.fluid = FluidPropertiesManager()
    state.fluid.density = float(input_data.fluid_properties.density)
    state.fluid.viscosity = float(input_data.fluid_properties.viscosity)
    
    state.external_forces = ExternalForceManager()
    state.external_forces.force_vector = np.array(input_data.external_forces.force_vector, dtype=np.float64)

    # --- 3. Initial Conditions ---
    state.initial_conditions = InitialConditionManager()
    state.initial_conditions.velocity = np.array(input_data.initial_conditions.velocity, dtype=np.float64)
    state.initial_conditions.pressure = float(input_data.initial_conditions.pressure)

    # --- 4. Simulation Parameters ---
    state.sim_params = SimulationParameterManager()
    state.sim_params.time_step = float(input_data.simulation_parameters.time_step)
    state.sim_params.total_time = float(input_data.simulation_parameters.total_time)
    state.sim_params.output_interval = int(input_data.simulation_parameters.output_interval)

    # --- 5. Topology & Geometry ---
    state.masks = MaskManager()
    mask_3d, _, _ = generate_3d_masks(input_data.mask.data, input_data.grid)
    state.masks.mask = mask_3d

    # --- 6. Boundary Conditions ---
    state.boundary_conditions = BoundaryConditionManager()
    for item in input_data.boundary_conditions.items:
        bc = BoundaryCondition()
        bc.location = str(item.location)
        bc.type = str(item.type)
        bc.values = dict(item.values)
        state.boundary_conditions.add_condition(bc)

    return state