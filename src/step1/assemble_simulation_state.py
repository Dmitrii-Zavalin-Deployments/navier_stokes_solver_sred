# file: step1/assemble_simulation_state.py
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .types import (
    Config,
    DerivedConstants,
    Fields,
    GridConfig,
    SimulationState,
)


def assemble_simulation_state(
    config: Config,
    grid: GridConfig,
    fields: Fields,
    mask_3d: np.ndarray,
    bc_table: Dict[str, Any],
    constants: DerivedConstants,
) -> SimulationState:
    """
    Combine all validated and constructed components into the final SimulationState.
    Step 1: mask_3d is structurally valid only; semantics are deferred.
    """
    return SimulationState(
        config=config,
        grid=grid,
        fields=fields,
        mask_3d=mask_3d,
        boundary_table=bc_table,
        constants=constants,
    )
