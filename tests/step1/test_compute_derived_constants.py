# tests/step1/test_compute_derived_constants.py

from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


def test_step1_constants_match_dummy():
    """
    Step 1 no longer computes mathematical derived constants.
    It simply forwards density, viscosity, dt, and uses unit spacing (dx=dy=dz=1.0),
    exactly as defined in the frozen Step 1 dummy.
    """

    # 1. Start with the canonical, schema-valid dummy
    json_input = solver_input_schema_dummy()

    # 2. Override specific values for this test case
    # We set nx=2 and x_max=2.0 (with x_min=0.0) to ensure dx = 1.0
    json_input["domain"].update({
        "nx": 2, "ny": 2, "nz": 2,
        "x_min": 0.0, "x_max": 2.0,
        "y_min": 0.0, "y_max": 2.0,
        "z_min": 0.0, "z_max": 2.0
    })

    json_input["fluid_properties"] = {
        "density": 5.0,
        "viscosity": 0.2,
    }

    # Ensure the simulation parameters block is fully satisfied
    json_input["simulation_parameters"].update({
        "time_step": 0.05,
        "total_time": 1.0,
        "output_interval": 1
    })

    # The dummy already provides a flat mask of length 8 (2*2*2)
    # which satisfies the schema requirement that 'mask' be a 1D array.

    # 3. Execute Step 1
    state = orchestrate_step1(json_input)
    constants = state["constants"]

    # 4. Assertions based on overrides and frozen Step 1 dummy rules:
    assert constants["rho"] == 5.0
    assert constants["mu"] == 0.2
    assert constants["dt"] == 0.05

    # Step 1 dummy uses unit spacing: (x_max - x_min) / nx = 2.0 / 2 = 1.0
    assert constants["dx"] == 1.0
    assert constants["dy"] == 1.0
    assert constants["dz"] == 1.0