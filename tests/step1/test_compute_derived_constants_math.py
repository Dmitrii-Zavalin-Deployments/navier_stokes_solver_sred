# tests/step1/test_compute_derived_constants_math.py

from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_compute_derived_constants_matches_dummy():
    """
    Step 1 no longer computes mathematical derived constants.
    It simply forwards density, viscosity, dt, and uses unit spacing (dx=dy=dz=1.0),
    exactly as defined in the frozen Step 1 dummy.
    """

    # 1. Start with the canonical, schema-valid dummy
    json_input = solver_input_schema_dummy()

    # 2. Override only the values specific to this math test
    # We set nx=1, x_max=1.0 to ensure dx=1.0 for the unit spacing test
    json_input["domain"].update({"nx": 2, "x_min": 0.0, "x_max": 2.0})
    json_input["domain"].update({"ny": 2, "y_min": 0.0, "y_max": 2.0})
    json_input["domain"].update({"nz": 2, "z_min": 0.0, "z_max": 2.0})
    
    json_input["fluid_properties"] = {
        "density": 10.0,
        "viscosity": 2.0,
    }
    json_input["simulation_parameters"]["time_step"] = 0.1

    # 3. Run Orchestrator
    state = orchestrate_step1(json_input)
    constants = state["constants"]

    # 4. Assertions based on overrides
    assert constants["rho"] == 10.0
    assert constants["mu"] == 2.0
    assert constants["dt"] == 0.1

    # Step 1 dummy uses unit spacing: (x_max - x_min) / nx = (2.0 - 0.0) / 2 = 1.0
    assert constants["dx"] == 1.0
    assert constants["dy"] == 1.0
    assert constants["dz"] == 1.0