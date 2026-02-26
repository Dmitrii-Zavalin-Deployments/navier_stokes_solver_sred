# src/solver_state.py

from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np

# =========================================================
# THE DEPARTMENT SAFES (Support Classes)
# =========================================================

@dataclass
class SolverConfig:
    """Step 0: Global instructions that change between runs."""
    case_name: str = "default_case"
    method: str = "jacobi"  # jacobi, cg, or direct
    precision: str = "float64" # float64 for science, float32 for speed

@dataclass
class GridContext:
    """
    Step 1: The Spatial World. 
    Mandatory parameters mapped directly from the 3D model.
    Defaults set to None to force a 'Loud Error' if not populated.
    """
    nx: int = None
    ny: int = None
    nz: int = None
    
    x_min: float = None
    x_max: float = None
    y_min: float = None
    y_max: float = None
    z_min: float = None
    z_max: float = None

    @property
    def dx(self) -> float: return (self.x_max - self.x_min) / self.nx
    @property
    def dy(self) -> float: return (self.y_max - self.y_min) / self.ny
    @property
    def dz(self) -> float: return (self.z_max - self.z_min) / self.nz
    
    @property
    def total_cells(self) -> int: return self.nx * self.ny * self.nz

@dataclass
class FieldData:
    """
    Step 1: The Memory Map.
    Initialized as None to force explicit allocation.
    Removed Optional to stay consistent with the 'Empty Safe' design.
    """
    P: np.ndarray = None
    U: np.ndarray = None
    V: np.ndarray = None
    W: np.ndarray = None

    def is_allocated(self) -> bool:
        """Check if all primary fields have been created."""
        return all(f is not None for f in [self.P, self.U, self.V, self.W])

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

@dataclass
class SolverState:
    """
    The Project Constitution: Article 3 (The Universal State Container).
    """

    # ---------------------------------------------------------
    # Step 0: Input / configuration
    # ---------------------------------------------------------
    config: SolverConfig = field(default_factory=SolverConfig)

    # ---------------------------------------------------------
    # Step 1: Grid & Fields (Solidified Safes)
    # ---------------------------------------------------------
    grid: GridContext = field(default_factory=GridContext)
    fields: FieldData = field(default_factory=FieldData)
    
    # Spatial Masks (Initialized as None to avoid Boolean type-mismatch in JSON)
    mask: Optional[np.ndarray] = None                             
    is_fluid: Optional[np.ndarray] = None
    is_boundary_cell: Optional[np.ndarray] = None
    is_solid: Optional[np.ndarray] = None
    
    # Physics & Environment
    constants: Dict[str, Any] = field(default_factory=dict)       
    fluid_properties: Dict[str, Any] = field(default_factory=dict) 
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Global health tracking
    health: Dict[str, Any] = field(default_factory=dict)          

    # ---------------------------------------------------------
    # Step 2: Operators & PPE structure
    # ---------------------------------------------------------
    operators: Dict[str, Any] = field(default_factory=dict)       
    ppe: Dict[str, Any] = field(default_factory=dict)             

    # ---------------------------------------------------------
    # Step 3: Projection & Intermediate Logic
    # ---------------------------------------------------------
    intermediate_fields: Dict[str, np.ndarray] = field(default_factory=dict) 
    step3_diagnostics: Dict[str, Any] = field(default_factory=dict)

    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": [],
    })

    # ---------------------------------------------------------
    # Step 4: Extended fields + Boundary Logic
    # ---------------------------------------------------------
    P_ext: Optional[np.ndarray] = None
    U_ext: Optional[np.ndarray] = None
    V_ext: Optional[np.ndarray] = None
    W_ext: Optional[np.ndarray] = None
    step4_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Step 5: Finalization & Outputs
    # ---------------------------------------------------------
    step5_outputs: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Flags & Time tracking
    # ---------------------------------------------------------
    ready_for_time_loop: bool = False
    iteration: int = 0
    time: float = 0.0

    # ---------------------------------------------------------
    # Attribute Interface (Interface Facade Rule)
    # ---------------------------------------------------------
    @property
    def pressure(self) -> Optional[np.ndarray]:
        """Accessor for cell-centered pressure."""
        return self.fields.get("P")

    @property
    def velocity_u(self) -> Optional[np.ndarray]:
        """Accessor for staggered x-velocity."""
        return self.fields.get("U")

    @property
    def velocity_v(self) -> Optional[np.ndarray]:
        """Accessor for staggered y-velocity."""
        return self.fields.get("V")

    @property
    def velocity_w(self) -> Optional[np.ndarray]:
        """Accessor for staggered z-velocity."""
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

        result = {}
        for key, value in self.__dict__.items():
            result[key] = convert(value)
        return result