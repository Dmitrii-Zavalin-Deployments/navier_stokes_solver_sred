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

    This class replaces the fragmented per-step JSON-safe mirrors and
    becomes the single source of truth for the entire solver pipeline.
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
        NumPy arrays become lists; nested dicts are converted recursively.
        Callables (e.g., operators) are skipped.
        """
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        result = {}
        for key, value in self.__dict__.items():
            # Skip runtime-only fields if needed later (e.g., callables)
            if key == "operators":
                # Operators often contain callables; skip or sanitize later
                continue
            result[key] = convert(value)

        return result
