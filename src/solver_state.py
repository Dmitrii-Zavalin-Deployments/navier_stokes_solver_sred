# src/solver_state.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class SolverState:
    """
    Central state object for the Navier–Stokes solver.

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
    constants: Dict[str, Any] = field(default_factory=dict)       # rho, dt, dx, etc.
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, Any] = field(default_factory=dict)          # divergence norms, etc.

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
    step3_diagnostics: Dict[str, Any] = field(default_factory=dict)

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
    # Flags
    # ---------------------------------------------------------
    ready_for_time_loop: bool = False

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------
    def to_json_safe(self) -> Dict[str, Any]:
        """
        Convert the entire solver state into a JSON-safe dictionary.
        
        Updated to satisfy the Phase C, Rule 7 (Scale Guard):
        - SciPy sparse matrices are summarized as metadata to prevent OOM errors.
        - NumPy arrays become lists.
        - Callables are nullified to maintain schema key-presence.
        """
        from scipy.sparse import issparse

        def convert(value):
            # 1. Handle SciPy Sparse Matrices (Scale Guard)
            # Prevents .tolist() on 10^6 x 10^6 matrices
            if issparse(value):
                return {
                    "type": str(value.format),  # e.g., 'csr', 'csc'
                    "shape": list(value.shape),
                    "nnz": int(value.nnz)
                }

            # 2. Handle Dense NumPy arrays
            if isinstance(value, np.ndarray):
                return value.tolist()

            # 3. Handle Dictionaries recursively
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}

            # 4. Handle Callables (replaced with None for JSON)
            if callable(value):
                return None

            # 5. Handle NumPy scalars (np.float64, etc.)
            if hasattr(value, "item") and not isinstance(value, (list, dict)):
                return value.item()

            return value

        result = {}
        for key, value in self.__dict__.items():
            result[key] = convert(value)

        return result