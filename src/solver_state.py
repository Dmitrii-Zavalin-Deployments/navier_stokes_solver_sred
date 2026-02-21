# src/solver_state.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

@dataclass
class SolverState:
    """
    The Project Constitution: Article 3 (The Universal State Container).
    
    This object is the 'Living Container' for the Navier-Stokes simulation.
    It enforces the 'Pure Path' by maintaining data in one specific location:
    - No aliases (e.g., no self.grid = config['grid'])
    - Data flows in a single direction
    - Step-specific dictionaries isolate progress logic
    """

    # ---------------------------------------------------------
    # Input / configuration (Step 0)
    # ---------------------------------------------------------
    # Raw input data and high-level solver settings
    config: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Step 1: Grid, fields, mask, constants, BCs
    # ---------------------------------------------------------
    # The Grid is the ONLY place where x_min, nx, dx, etc., live.
    grid: Dict[str, Any] = field(default_factory=dict)            
    
    # Active simulation fields: P (Scalar), U, V, W (Staggered vectors)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)   
    
    # Physical/Logical Domain Masks
    mask: Optional[np.ndarray] = None                             
    is_fluid: Optional[np.ndarray] = None
    is_boundary_cell: Optional[np.ndarray] = None
    is_solid: Optional[np.ndarray] = None
    
    # Simulation Parameters
    constants: Dict[str, Any] = field(default_factory=dict)       
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Global health tracking (metrics like divergence, residual norms)
    health: Dict[str, Any] = field(default_factory=dict)          

    # ---------------------------------------------------------
    # Step 2: Operators & PPE structure
    # ---------------------------------------------------------
    # Discrete differential operators (Sparse Matrices)
    operators: Dict[str, Any] = field(default_factory=dict)       
    
    # Pressure Poisson Equation (PPE) setup
    ppe: Dict[str, Any] = field(default_factory=dict)             

    # ---------------------------------------------------------
    # Step 3: Projection & Intermediate Logic
    # ---------------------------------------------------------
    # Intermediate 'Star' fields (U*, V*, W*) for the fractional step method
    intermediate_fields: Dict[str, np.ndarray] = field(default_factory=dict) 
    
    # Performance and convergence diagnostics for Step 3
    step3_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Simulation History
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
    # Extended domains for boundary condition application (Ghost cells)
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
    # Serialization (Scale Guard Compliant)
    # ---------------------------------------------------------
    def to_json_safe(self) -> Dict[str, Any]:
        """
        Convert state to JSON-safe dict. 
        Enforces Phase C, Article 7 (Anti-Density Rule): 
        Sparse matrices are serialized as metadata, never densified.
        """
        try:
            from scipy.sparse import issparse
        except ImportError:
            def issparse(obj): return False

        def convert(value):
            # 1. SciPy Sparse Matrices (Scale Guard: Do not use .todense())
            if issparse(value):
                return {
                    "type": str(value.format),
                    "shape": list(value.shape),
                    "nnz": int(value.nnz)
                }

            # 2. Dense NumPy arrays
            if isinstance(value, np.ndarray):
                return value.tolist()

            # 3. Dictionaries (Recursive)
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            
            # 4. Lists (Recursive)
            if isinstance(value, list):
                return [convert(v) for v in value]

            # 5. Callables (Discarded for JSON)
            if callable(value):
                return None

            # 6. NumPy scalars (Fixing serialization types)
            if hasattr(value, "item") and not isinstance(value, (list, dict, np.ndarray)):
                return value.item()

            return value

        result = {}
        for key, value in self.__getstate__().items() if hasattr(self, "__getstate__") else self.__dict__.items():
            result[key] = convert(value)

        return result