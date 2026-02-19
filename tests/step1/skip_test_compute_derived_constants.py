# tests/step1/test_compute_derived_constants.py

from src.step1.orchestrate_step1 import orchestrate_step1


def test_step1_constants_match_dummy():
    """
    Step 1 no longer computes mathematical derived constants.
    It simply forwards density, viscosity, dt, and uses unit spacing (dx=dy=dz=1.0),
    exactly as defined in the frozen Step 1 dummy.
    """

    json_input = {
        "domain": {"nx": 2, "ny": 2, "nz": 2},

        "fluid_properties": {
            "density": 5.0,
            "viscosity": 0.2,
        },

        # Step 1 parser still expects simulation_parameters.time_step
        "simulation_parameters": {"time_step": 0.05},

        "external_forces": {"force_vector": [0.0, 0.0, 0.0]},

        # Simple 2×2×2 mask
        "mask": [
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
        ],
    }

    state = orchestrate_step1(json_input)
    constants = state["constants"]

    # Frozen Step 1 dummy rules:
    assert constants["rho"] == 5.0
    assert constants["mu"] == 0.2
    assert constants["dt"] == 0.05

    # Step 1 dummy uses unit spacing
    assert constants["dx"] == 1.0
    assert constants["dy"] == 1.0
    assert constants["dz"] == 1.0
