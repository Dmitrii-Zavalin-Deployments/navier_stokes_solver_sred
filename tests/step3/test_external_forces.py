# tests/step3/test_external_forces.py

import numpy as np
from src.step3.predict_velocity import predict_velocity
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


# ----------------------------------------------------------------------
# Helper: convert Step‑2 dummy → Step‑3 input shape
# ----------------------------------------------------------------------
def adapt_step2_to_step3(state):
    return {
        "Config": state["config"],
        "Mask": state["fields"]["Mask"],
        "is_fluid": state["fields"]["Mask"] == 1,
        "is_boundary_cell": np.zeros_like(state["fields"]["Mask"], bool),

        "P": state["fields"]["P"],
        "U": state["fields"]["U"],
        "V": state["fields"]["V"],
        "W": state["fields"]["W"],

        "BCs": state["boundary_table_list"],

        "Constants": {
            "rho": state["config"]["fluid"]["density"],
            "mu": state["config"]["fluid"]["viscosity"],
            "dt": state["config"]["simulation"]["dt"],
            "dx": state["grid"]["dx"],
            "dy": state["grid"]["dy"],
            "dz": state["grid"]["dz"],
        },

        "Operators": state["operators"],

        "PPE": {
            "solver": None,
            "tolerance": 1e-6,
            "max_iterations": 100,
            "ppe_is_singular": False,
        },

        "Health": {},
        "History": {},
    }


# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------

def test_external_forces_modify_velocity():
    # Step‑2‑schema‑valid dummy
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    # Apply a simple force in x-direction
    state["Config"]["external_forces"] = {"fx": 1.0, "fy": 0.0, "fz": 0.0}

    U0 = state["U"].copy()

    U_star, V_star, W_star = predict_velocity(state)

    # U* must increase due to fx
    assert np.any(U_star > U0)
