# tests/step_2/test_orchestrate_step2_no_validator.py
import numpy as np
import src.step2.orchestrate_step2 as o


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


def test_orchestrate_step2_no_validator():
    o.validate_json_schema = None  # simulate missing validator

    state = DummyState()
    result = o.orchestrate_step2(state)

    assert hasattr(result, "Constants")
