# src/solver_state.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

@dataclass
class SolverState:
    """
    The Project Constitution: Article 3 (The Universal State Container).
    Updated Feb 2026: Implements Canonical Facades for Zero-Shadowing.
    
    This object acts as the 'Single Source of Truth' for the simulation.
    Properties provide a mathematical facade over departmentalized data.
    """

    # ---------------------------------------------------------
    # Step 0: Input / configuration
    # ---------------------------------------------------------
    config: Dict[str, Any] = field(default_factory=dict)
    # The source of truth for temporal settings
    simulation_parameters: Dict[str, Any] = field(default_factory=dict) 

    # ---------------------------------------------------------
    # Step 1: Grid, fields, mask, constants, BCs, Fluid Physics
    # ---------------------------------------------------------
    grid: Dict[str, Any] = field(default_factory=dict)            
    fields: Dict[str, np.ndarray] = field(default_factory=dict)   
    
    # Spatial Masks (Initialized as None to avoid type-mismatch in initial JSON)
    mask: Optional[np.ndarray] = None                             
    is_fluid: Optional[np.ndarray] = None
    is_boundary_cell: Optional[np.ndarray] = None
    is_solid: Optional[np.ndarray] = None
    
    # Physics & Environment (Primary homes for data)
    constants: Dict[str, Any] = field(default_factory=dict)       
    fluid_properties: Dict[str, Any] = field(default_factory=dict) 
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Global health tracking
    health: Dict[str, Any] = field(default_factory=dict)          

    # ---------------------------------------------------------
    # Step 2-5: Departments (Operators, PPE, Intermediate, Ext, Outputs)
    # ---------------------------------------------------------
    operators: Dict[str, Any] = field(default_factory=dict)       
    ppe: Dict[str, Any] = field(default_factory=dict)             

    intermediate_fields: Dict[str, np.ndarray] = field(default_factory=dict) 
    step3_diagnostics: Dict[str, Any] = field(default_factory=dict)

    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": [],
    })

    P_ext: Optional[np.ndarray] = None
    U_ext: Optional[np.ndarray] = None
    V_ext: Optional[np.ndarray] = None
    W_ext: Optional[np.ndarray] = None
    step4_diagnostics: Dict[str, Any] = field(default_factory=dict)
    step5_outputs: Dict[str, Any] = field(default_factory=dict)

    # Flags & Time tracking
    ready_for_time_loop: bool = False
    iteration: int = 0
    time: float = 0.0

    # ---------------------------------------------------------
    # Zero-Debt Property Facades (The Single Source of Truth)
    # ---------------------------------------------------------
    @property
    def dt(self) -> Optional[float]:
        """Fetch dt from simulation_parameters (Article 5)."""
        return self.simulation_parameters.get("time_step")

    @property
    def rho(self) -> Optional[float]:
        """Fetch density from fluid_properties."""
        return self.fluid_properties.get("density")

    @property
    def mu(self) -> Optional[float]:
        """Fetch viscosity from fluid_properties."""
        return self.fluid_properties.get("viscosity")

    @property
    def nu(self) -> float:
        """
        Calculated kinematic viscosity (mu / rho).
        Enforces physical consistency without data duplication.
        """
        rho = self.rho
        mu = self.mu
        if rho and mu and rho != 0:
            return mu / rho
        return 0.0

    @property
    def g(self) -> float:
        """Fetch gravity from constants, default to 9.81."""
        return self.constants.get("g", 9.81)

    # ---------------------------------------------------------
    # Field Accessors (Readability shortcuts for math)
    # ---------------------------------------------------------
    @property
    def pressure(self) -> Optional[np.ndarray]:
        return self.fields.get("P")

    @property
    def velocity_u(self) -> Optional[np.ndarray]:
        return self.fields.get("U")

    @property
    def velocity_v(self) -> Optional[np.ndarray]:
        return self.fields.get("V")

    @property
    def velocity_w(self) -> Optional[np.ndarray]:
        return self.fields.get("W")

    # ---------------------------------------------------------
    # Serialization (Scale Guard Compliant)
    # ---------------------------------------------------------
    def to_json_safe(self) -> Dict[str, Any]:
        """
        Convert state to JSON-safe dict. 
        Enforces Phase C, Article 7 (Anti-Density Rule).
        """
        try:
            from scipy.sparse import issparse
        except ImportError:
            def issparse(obj): return False

        def convert(value):
            if issparse(value):
                return {
                    "type": str(value.format),
                    "shape": list(value.shape),
                    "nnz": int(value.nnz)
                }
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            if callable(value):
                return None
            if hasattr(value, "item") and not isinstance(value, (list, dict, np.ndarray)):
                return value.item()
            if value is None:
                return None
            return value

        return {key: convert(value) for key, value in self.__dict__.items()}