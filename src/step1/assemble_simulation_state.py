from .simulation_state import SimulationState, Grid
import numpy as np
from typing import Dict, Any


def assemble_simulation_state(config: Dict[str, Any],
                              grid: Grid,
                              P: np.ndarray,
                              U: np.ndarray,
                              V: np.ndarray,
                              W: np.ndarray,
                              mask: np.ndarray,
                              constants: Dict[str, float]) -> SimulationState:
    return SimulationState(
        grid=grid,
        P=P,
        U=U,
        V=V,
        W=W,
        mask=mask,
        constants=constants,
        config=config,
    )
