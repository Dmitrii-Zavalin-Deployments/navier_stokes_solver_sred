# tests/step_2/test_orchestrate_step2_health_structure.py
import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2


class DummyState:
    def __init__(self, nx=4, ny=4, nz=4):
        self.Grid = {"nx": nx, "ny": ny, "nz": nz, "dx": 1.0, "dy": 1.0, "dz": 1.0}
        self.Config = {
            "fluid_properties": {"density": 1.0, "viscosity": 1.0},
            "simulation_parameters": {"dt": 0.1, "advection_scheme": "central"},
        }
        self.Mask = np.ones((nx, ny, nz), dtype=int)
        self.U = np.zeros((nx + 1, ny, nz))
        self.V = np.zeros((nx, ny + 1, nz))
        self.W = np.zeros((nx, ny, nz + 1))


def test_orchestrate_step2_health_structure():
    state = DummyState()
    result = orchestrate_step2(state)

    health = result.Health

    assert "initial_divergence_norm" in health
    assert "max_velocity_magnitude" in health
    assert "cfl_advection_estimate" in health
