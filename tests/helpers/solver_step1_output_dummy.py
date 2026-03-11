"""
Archivist Testing: Explicit SolverState Hydration.

Compliance:
- Rule 4: SSoT (Hierarchy over convenience).
- Rule 5: Deterministic Initialization (Constructor-based injection).
"""

import numpy as np

from src.common.solver_state import (
    BoundaryCondition,
    BoundaryConditionManager,
    DomainManager,
    ExternalForceManager,
    FieldManager,
    FluidPropertiesManager,
    GridManager,
    InitialConditionManager,
    MaskManager,
    SimulationParameterManager,
    SolverState,
)


def make_step1_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4) -> SolverState:
    """
    Hydrates a SolverState prototype using atomic constructor injection.
    Validation is triggered immediately upon instantiation of each manager.
    """
    state = SolverState()
    
    # 1. Geometry & Domain: Atomic Constructor Injection
    state._grid = GridManager()
    state._grid._x_min, state._grid._x_max = 0.0, 1.0
    state._grid._y_min, state._grid._y_max = 0.0, 1.0
    state._grid._z_min, state._grid._z_max = 0.0, 1.0
    state._grid._nx, state._grid._ny, state._grid._nz = nx, ny, nz
    
    state._domain = DomainManager(
        type="INTERNAL", 
        reference_velocity=np.array([0.0, 0.0, 0.0])
    )
    
    # 2. Physics & Foundation: Atomic Constructor Injection
    state._fluid = FluidPropertiesManager(density=1000.0, viscosity=0.001)
    state._initial_conditions = InitialConditionManager(
        velocity=np.array([0.0, 0.0, 0.0]), 
        pressure=0.0
    )
    state._sim_params = SimulationParameterManager(
        time_step=0.001, 
        total_time=1.0, 
        output_interval=1
    )
    state._external_forces = ExternalForceManager(
        force_vector=np.array([0.0, 0.0, -9.81])
    )
    
    # 3. Foundation Allocation: Explicit call per Rule 9
    state._fields = FieldManager()
    state._fields.allocate(nx * ny * nz)
    
    # 4. Topology: Explicit injection
    state._masks = MaskManager()
    state._masks.mask = np.ones((nx, ny, nz), dtype=int)
    
    # 5. Boundary Condition Setup: Atomic instantiation
    state._boundary_conditions = BoundaryConditionManager(conditions=[
        BoundaryCondition(location='x_min', type='inflow', values={'u': 1.0, 'v': 0.0, 'w': 0.0, 'p': 1.0}),
        BoundaryCondition(location='x_max', type='outflow', values={'p': 0.0}),
        BoundaryCondition(location='y_min', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='y_max', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='z_min', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='z_max', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='wall', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0})
    ])
    
    # 6. Engine State: Explicit declaration
    state._iteration = 0
    state._time = 0.0
    state._ready_for_time_loop = False

    return state