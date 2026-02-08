import numpy as np
from src.step3.orchestrate_step3 import step3
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


class DummyHandler:
    def __init__(self):
        self.pre_called = False
        self.post_called = False

    def apply_pre(self, state):
        self.pre_called = True

    def apply_post(self, state):
        self.post_called = True


def test_bc_handler_hooks_called():
    # Create a Step‑2‑schema‑valid dummy state
    s2 = Step2SchemaDummyState(nx=4, ny=4, nz=4)

    # Convert to Step‑3 shape
    state = adapt_step2_to_step3(s2)

    # Attach the handler
    handler = DummyHandler()
    state["BC_handler"] = handler

    # Run Step 3
    step3(state, current_time=0.0, step_index=0)

    # Verify hooks were called
    assert handler.pre_called
    assert handler.post_called
