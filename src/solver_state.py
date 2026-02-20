# src/solver_state.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class SolverState:
    """
    Central state object for the Navier–Stokes solver.
    Aligned with Step 3 contracts and fractional-step requirements.

    This object is progressively filled by Steps 1–5:
      - Step 1: grid, fields, mask, constants, BCs
      - Step 2: operators, PPE structure, health updates
      - Step 3: corrected fields, health, diagnostics
      - Step 4: extended fields (P_ext, U_ext, V_ext, W_ext), BC application diagnostics
      - Step 5: final diagnostics / outputs

    Step 6 serializes this object via `to_json_safe()` and validates it
    against the final output schema.
    """

    # ---------------------------------------------------------
    # Input / configuration (Step 0)
    # ---------------------------------------------------------
    config: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Step 1: Grid, fields, mask, constants, BCs
    # ---------------------------------------------------------
    grid: Dict[str, Any] = field(default_factory=dict)            # x_min, nx, dx, etc.
    fields: Dict[str, np.ndarray] = field(default_factory=dict)   # P, U, V, W
    mask: Optional[np.ndarray] = None                             # geometry mask
    constants: Dict[str, Any] = field(default_factory=dict)       # rho, dt, nu, etc.
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Global health tracking (post-Step 2 and post-Step 3 metrics)
    health: Dict[str, Any] = field(default_factory=dict)          

    # ---------------------------------------------------------
    # Step 2: Operators, PPE structure, mask semantics
    # ---------------------------------------------------------
    operators: Dict[str, Any] = field(default_factory=dict)       # divergence, gradient, laplacian
    ppe: Dict[str, Any] = field(default_factory=dict)             # PPE RHS, matrix structure
    is_fluid: Optional[np.ndarray] = None
    is_boundary_cell: Optional[np.ndarray] = None
    is_solid: Optional[np.ndarray] = None

    # ---------------------------------------------------------
    # Step 3: Projection, velocity correction, diagnostics
    # ---------------------------------------------------------
    # Intermediate "Star" fields for Fractional Step method
    intermediate_fields: Dict[str, np.ndarray] = field(default_factory=dict) 
    
    # Specific diagnostics for the Step 3 execution phase
    step3_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Simulation History (Internal update, not necessarily in output schemas)
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": [],
    })

    # ---------------------------------------------------------
    # Step 4: Extended fields + BC application
    # ---------------------------------------------------------
    P_ext: Optional[np.ndarray] = None
    U_ext: Optional[np.ndarray] = None
    V_ext: Optional[np.ndarray] = None
    W_ext: Optional[np.ndarray] = None
    step4_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Step 5: Final diagnostics / outputs
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
        Convert state to JSON-safe dict. Replaces sparse matrices with metadata
        and callables with None to prevent memory/serialization crashes.
        """
        try:
            from scipy.sparse import issparse
        except ImportError:
            def issparse(obj): return False

        def convert(value):
            # 1. SciPy Sparse Matrices
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
            
            # 4. Lists (Recursive - for History)
            if isinstance(value, list):
                return [convert(v) for v in value]

            # 5. Callables
            if callable(value):
                return None

            # 6. NumPy scalars
            if hasattr(value, "item") and not isinstance(value, (list, dict)):
                return value.item()

            return value

        result = {}
        for key, value in self.__dict__.items():
            result[key] = convert(value)

        return result