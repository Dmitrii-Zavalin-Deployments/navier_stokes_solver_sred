import pytest
from src.step1.construct_simulation_state import construct_simulation_state


@pytest.fixture
def valid_json():
    """Baseline valid JSON input for Step 1 (schema-compliant)."""
    return {
        "domain_definition": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 4, "ny": 4, "nz": 4
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.1
        },
        "initial_conditions": {
            "initial_velocity": [1.0, 0.0, 0.0],
            "initial_pressure": 0.0
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 1.0,
            "output_interval": 10
        },
        "boundary_conditions": [
            {
                "role": "wall",
                "type": "dirichlet",
                "faces": ["x_min"],
                "apply_to": ["velocity"],
                "velocity": [0.0, 0.0, 0.0],
                "pressure": 0.0,
                "pressure_gradient": 0.0,
                "no_slip": True,
                "comment": "test"
            }
        ],
        "geometry_definition": {
            "geometry_mask_flat": [1] * (4 * 4 * 4),
            "geometry_mask_shape": [4, 4, 4],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "i + nx*(j + ny*k)"
        },
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "units": "N",
            "comment": "none"
        }
    }


# ---------------------------------------------------------------------------
# 1. valid_full_input
# ---------------------------------------------------------------------------

def test_valid_full_input(valid_json):
    """End‑to‑end: ensure SimulationState is fully constructed."""
    state = construct_simulation_state(valid_json)

    assert state is not None
    assert state.P.shape == (4, 4, 4)
    assert state.U.shape == (5, 4, 4)
    assert state.V.shape == (4, 5, 4)
    assert state.W.shape == (4, 4, 5)
    assert state.constants["dx"] > 0
    assert state.mask.shape == (4, 4, 4)


# ---------------------------------------------------------------------------
# 2. valid_minimal_grid
# ---------------------------------------------------------------------------

def test_valid_minimal_grid(valid_json):
    """nx=ny=nz=1 should still produce valid staggered shapes."""
    valid_json["domain_definition"]["nx"] = 1
    valid_json["domain_definition"]["ny"] = 1
    valid_json["domain_definition"]["nz"] = 1

    valid_json["geometry_definition"]["geometry_mask_flat"] = [1]
    valid_json["geometry_definition"]["geometry_mask_shape"] = [1, 1, 1]

    state = construct_simulation_state(valid_json)

    assert state.P.shape == (1, 1, 1)
    assert state.U.shape == (2, 1, 1)
    assert state.V.shape == (1, 2, 1)
    assert state.W.shape == (1, 1, 2)
    assert state.mask.shape == (1, 1, 1)


# ---------------------------------------------------------------------------
# 3. valid_zero_viscosity
# ---------------------------------------------------------------------------

def test_valid_zero_viscosity(valid_json):
    """viscosity=0 must not raise."""
    valid_json["fluid_properties"]["viscosity"] = 0.0

    state = construct_simulation_state(valid_json)

    assert state.constants["mu"] == 0.0


# ---------------------------------------------------------------------------
# 4. valid_negative_bounds
# ---------------------------------------------------------------------------

def test_valid_negative_bounds(valid_json):
    """Negative coordinates must still produce positive dx."""
    valid_json["domain_definition"]["x_min"] = -1.0
    valid_json["domain_definition"]["x_max"] = 1.0

    state = construct_simulation_state(valid_json)

    assert state.grid.dx > 0


# ---------------------------------------------------------------------------
# 5. valid_custom_flattening
# ---------------------------------------------------------------------------

def test_valid_custom_flattening(valid_json):
    """Custom flattening_order must reshape mask correctly."""
    valid_json["geometry_definition"]["flattening_order"] = "j + ny*(i + nx*k)"

    state = construct_simulation_state(valid_json)

    assert state.mask.shape == (4, 4, 4)
