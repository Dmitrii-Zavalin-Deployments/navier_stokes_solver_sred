import pytest
from src.step1.construct_simulation_state import construct_simulation_state


@pytest.fixture
def valid_json():
    """Baseline valid JSON input for Step 1."""
    return {
        "domain": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 4, "ny": 4, "nz": 4
        },
        "fluid": {
            "density": 1.0,
            "viscosity": 0.1
        },
        "simulation": {
            "dt": 0.01,
            "total_time": 1.0,
            "flattening_order": "i + nx*(j + ny*k)",
            "initial_pressure": 0.0,
            "initial_velocity": [1.0, 0.0, 0.0],
            "force_vector": [0.0, 0.0, 0.0]
        },
        "geometry_mask_flat": [1] * (4 * 4 * 4),
        "boundary_conditions": [
            {"face": "x_min", "role": "wall"},
            {"face": "x_max", "role": "outlet"},
            {"face": "y_min", "role": "wall"},
            {"face": "y_max", "role": "wall"},
            {"face": "z_min", "role": "wall"},
            {"face": "z_max", "role": "wall"}
        ]
    }


# ---------------------------------------------------------------------------
# 1. Missing required keys
# ---------------------------------------------------------------------------

def test_missing_required_key_domain(valid_json):
    del valid_json["domain"]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_missing_required_key_fluid(valid_json):
    del valid_json["fluid"]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_missing_required_key_simulation(valid_json):
    del valid_json["simulation"]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_missing_required_key_mask(valid_json):
    del valid_json["geometry_mask_flat"]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 2. Wrong types
# ---------------------------------------------------------------------------

def test_invalid_type_nx(valid_json):
    valid_json["domain"]["nx"] = "four"
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_type_initial_velocity(valid_json):
    valid_json["simulation"]["initial_velocity"] = "not a list"
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_type_force_vector(valid_json):
    valid_json["simulation"]["force_vector"] = "abc"
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 3. Invalid lengths
# ---------------------------------------------------------------------------

def test_invalid_velocity_length(valid_json):
    valid_json["simulation"]["initial_velocity"] = [1.0, 0.0]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_force_vector_length(valid_json):
    valid_json["simulation"]["force_vector"] = [0.0, 0.0]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_mask_length(valid_json):
    valid_json["geometry_mask_flat"] = valid_json["geometry_mask_flat"][:-1]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 4. Missing flattening_order
# ---------------------------------------------------------------------------

def test_missing_flattening_order(valid_json):
    del valid_json["simulation"]["flattening_order"]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)
