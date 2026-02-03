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
# 1. dx must be positive
# ---------------------------------------------------------------------------

def test_grid_spacing_positive(valid_json):
    state = construct_simulation_state(valid_json)
    assert state.grid.dx > 0


# ---------------------------------------------------------------------------
# 2. dx must match domain extents
# ---------------------------------------------------------------------------

def test_grid_spacing_correct(valid_json):
    dom = valid_json["domain_definition"]
    expected_dx = abs(dom["x_max"] - dom["x_min"]) / dom["nx"]

    state = construct_simulation_state(valid_json)
    assert abs(state.grid.dx - expected_dx) < 1e-12


# ---------------------------------------------------------------------------
# 3. Degenerate domain must raise
# ---------------------------------------------------------------------------

def test_grid_invalid_zero_extent(valid_json):
    valid_json["domain_definition"]["x_min"] = 1.0
    valid_json["domain_definition"]["x_max"] = 1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 4. Inverted domain must raise
# ---------------------------------------------------------------------------

def test_grid_invalid_inverted_extent(valid_json):
    valid_json["domain_definition"]["x_min"] = 2.0
    valid_json["domain_definition"]["x_max"] = 1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 5. Non-positive grid resolution must raise
# ---------------------------------------------------------------------------

def test_grid_invalid_resolution_nx(valid_json):
    valid_json["domain_definition"]["nx"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_grid_invalid_resolution_ny(valid_json):
    valid_json["domain_definition"]["ny"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_grid_invalid_resolution_nz(valid_json):
    valid_json["domain_definition"]["nz"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)
