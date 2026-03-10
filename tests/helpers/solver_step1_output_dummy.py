# tests/helpers/solver_step1_output_dummy.py

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


def make_step1_output_dummy(nx=4, ny=4, nz=4):
    state = SolverState()
    
    # 1. Hydrate individual managers
    state.domain = DomainManager(type="INTERNAL", reference_velocity=np.array([0.0, 0.0, 0.0]))
    
    state.grid = GridManager(
        _x_min=0.0, _x_max=1.0, _y_min=0.0, _y_max=1.0, _z_min=0.0, _z_max=1.0, 
        _nx=nx, _ny=ny, _nz=nz
    )
    
    state.fluid = FluidPropertiesManager(density=1000.0, viscosity=0.001)
    
    state.initial_conditions = InitialConditionManager(
        velocity=np.array([0.0, 0.0, 0.0]), pressure=0.0
    )
    
    state.sim_params = SimulationParameterManager(
        time_step=0.001, total_time=1.0, output_interval=1
    )
    
    state.external_forces = ExternalForceManager(
        force_vector=np.array([0.0, 0.0, -9.81])
    )
    
    # 2. Boundary Condition Setup
    bcs = [
        BoundaryCondition(location='x_min', type='inflow', values={'u': 1.0, 'v': 0.0, 'w': 0.0, 'p': 1.0}),
        BoundaryCondition(location='x_max', type='outflow', values={'p': 0.0}),
        BoundaryCondition(location='y_min', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='y_max', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='z_min', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='z_max', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='wall', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0})
    ]
    state.boundary_conditions = BoundaryConditionManager(conditions=bcs)
    
    # 3. Geometry Mask
    state.masks = MaskManager(_mask=np.ones((nx, ny, nz), dtype=int))
    
    # 4. Engine state
    state.iteration = 0
    state.time = 0.0
    state.ready_for_time_loop = False

    return state