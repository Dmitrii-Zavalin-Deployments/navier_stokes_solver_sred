# src/common/solver_state.py

from dataclasses import dataclass, field
import numpy as np
from src.common.base_container import ValidatedContainer

# =========================================================
# THE DEPARTMENT SAFES (Memory-Hardened Managers)
# =========================================================

@dataclass
class DomainManager(ValidatedContainer):
    __slots__ = ['type', 'reference_velocity']
    type: str  # Mandatory: No defaults per Rule 5
    reference_velocity: np.ndarray

    def __post_init__(self):
        if self.type not in ["INTERNAL", "EXTERNAL"]:
            raise ValueError("Domain type must be INTERNAL or EXTERNAL.")

@dataclass
class GridManager(ValidatedContainer):
    __slots__ = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 'nx', 'ny', 'nz']
    x_min: float; x_max: float
    y_min: float; y_max: float
    z_min: float; z_max: float
    nx: int; ny: int; nz: int

    @property
    def dx(self) -> float: return (self.x_max - self.x_min) / self.nx
    @property
    def dy(self) -> float: return (self.y_max - self.y_min) / self.ny
    @property
    def dz(self) -> float: return (self.z_max - self.z_min) / self.nz

@dataclass
class FluidPropertiesManager(ValidatedContainer):
    __slots__ = ['density', 'viscosity']
    density: float
    viscosity: float

@dataclass
class InitialConditionManager(ValidatedContainer):
    __slots__ = ['velocity', 'pressure']
    velocity: np.ndarray
    pressure: float

@dataclass
class SimulationParameterManager(ValidatedContainer):
    __slots__ = ['time_step', 'total_time', 'output_interval']
    time_step: float
    total_time: float
    output_interval: int

@dataclass
class BoundaryCondition(ValidatedContainer):
    __slots__ = ['location', 'type', 'values']
    location: str
    type: str
    values: dict

@dataclass
class BoundaryConditionManager(ValidatedContainer):
    __slots__ = ['conditions']
    conditions: list[BoundaryCondition]

@dataclass
class MaskManager(ValidatedContainer):
    __slots__ = ['_mask']
    _mask: np.ndarray

    @property
    def mask(self) -> np.ndarray: return self._mask

@dataclass
class ExternalForceManager(ValidatedContainer):
    __slots__ = ['force_vector']
    force_vector: np.ndarray

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

@dataclass
class SolverState(ValidatedContainer):
    __slots__ = [
        'domain', 'grid', 'fluid', 'initial_conditions', 
        'boundary_conditions', 'external_forces', 'sim_params', 
        'masks', 'iteration', 'time', 'ready_for_time_loop'
    ]

    domain: DomainManager
    grid: GridManager
    fluid: FluidPropertiesManager
    initial_conditions: InitialConditionManager
    boundary_conditions: BoundaryConditionManager
    external_forces: ExternalForceManager
    sim_params: SimulationParameterManager
    masks: MaskManager
    
    iteration: int = 0
    time: float = 0.0
    ready_for_time_loop: bool = False