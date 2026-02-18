# tests/step1/test_compute_derived_constants_math.py

from src.step1.orchestrate_step1 import orchestrate_step1


def test_compute_derived_constants_matches_dummy():
    """
    Step 1 no longer computes mathematical derived constants.
    It simply forwards density, viscosity, dt, and uses unit spacing (dx=dy=dz=1.0),
    exactly as defined in the frozen Step 1 dummy.
    """

    json_input = {
        "domain": {"nx": 4, "ny": 4, "nz": 4},
        "fluid_properties": {
            "density": 10.0,
            "viscosity": 2.0,
        },
        "time_integration": {"dt": 0.1},
        "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
        "mask": [
            [[1, 1, 1, 1]] * 4,
            [[1, 1, 1, 1]] * 4,
            [[1, 1, 1, 1]] * 4,
            [[1, 1, 1, 1]] * 4,
        ],
    }

    state = orchestrate_step1(json_input)
    constants = state["constants"]

    # Frozen Step 1 dummy rules:
    assert constants["rho"] == 10.0
    assert constants["mu"] == 2.0
    assert constants["dt"] == 0.1

    # Step 1 dummy uses unit spacing
    assert constants["dx"] == 1.0
    assert constants["dy"] == 1.0
    assert constants["dz"] == 1.0
