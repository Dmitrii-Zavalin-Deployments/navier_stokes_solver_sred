# src/solver_state.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class SolverState:
    """
    Central state object for the Navier–Stokes solver.
    Progressively filled by Steps 1–5.
    """

    # Step 0
    config: Dict[str, Any] = field(default_factory=dict)

    # Step 1
    grid: Dict[str, Any] = field(default_factory=dict)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    mask: Optional[np.ndarray] = None
    constants: Dict[str, Any] = field(default_factory=dict)
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, Any] = field(default_factory=dict)

    # Step 2
    operators: Dict[str, Any] = field(default_factory=dict)
    ppe: Dict[str, Any] = field(default_factory=dict)
    is_fluid: Optional[np.ndarray] = None
    is_boundary_cell: Optional[np.ndarray] = None
    is_solid: Optional[np.ndarray] = None  # Added to match Step 1 orchestrator use

    # Step 3
    step3_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Step 4
    P_ext: Optional[np.ndarray] = None
    U_ext: Optional[np.ndarray] = None
    V_ext: Optional[np.ndarray] = None
    W_ext: Optional[np.ndarray] = None
    step4_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Step 5
    step5_outputs: Dict[str, Any] = field(default_factory=dict)

    # Flags
    ready_for_time_loop: bool = False

    def to_json_safe(self) -> Dict[str, Any]:
        """
        Convert the state to a JSON-safe dictionary.
        NumPy arrays become lists.
        Handles the 'operators' key safely to satisfy schema requirements.
        """
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                # Recursively convert, but skip callables/matrices inside dicts
                return {k: convert(v) for k, v in value.items() if not callable(v)}
            if callable(value):
                return None
            return value

        result = {}
        # Use __dict__ to iterate over all dataclass fields
        for key, value in self.__dict__.items():
            # If we are at Step 1, operators is an empty dict {}.
            # We convert it instead of skipping it so the 'key' exists in JSON.
            result[key] = convert(value)

        return result