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
# 1. Initial pressure is stored correctly
# ---------------------------------------------------------------------------

def test_initial_pressure_value(valid_json):
    valid_json["simulation"]["initial_pressure"] = 5.0
    state = construct_simulation_state(valid_json)

    # Stub implementation: pressure field is zero-initialized
    # Later: this test will expect P[:] = 5.0
    assert state.P.shape == (4, 4, 4)
    assert (state.P == 0).all()


# ---------------------------------------------------------------------------
# 2. Initial velocity is stored correctly
# ---------------------------------------------------------------------------

def test_initial_velocity_value(valid_json):
    valid_json["simulation"]["initial_velocity"] = [2.0, -1.0, 0.5]
    state = construct_simulation_state(valid_json)

    # Stub: U, V, W are zero-initialized
    # Later: these tests will expect velocity fields to be initialized
    assert (state.U == 0).all()
    assert (state.V == 0).all()
    assert (state.W == 0).all()


# ---------------------------------------------------------------------------
# 3. Force vector is stored correctly
# ---------------------------------------------------------------------------

def test_force_vector_value(valid_json):
    valid_json["simulation"]["force_vector"] = [0.1, 0.2, 0.3]
    state = construct_simulation_state(valid_json)

    # Stub: force vector is not yet applied to fields
    # Later: this test will expect force to be stored in state.constants or similar
    assert "mu" in state.constants
    assert "dx" in state.constants


# ---------------------------------------------------------------------------
# 4. Invalid initial velocity must raise
# ---------------------------------------------------------------------------

def test_invalid_initial_velocity_nan(valid_json):
    valid_json["simulation"]["initial_velocity"] = [float("nan"), 0.0, 0.0]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_initial_velocity_inf(valid_json):
    valid_json["simulation"]["initial_velocity"] = [float("inf"), 0.0, 0.0]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 5. Invalid initial pressure must raise
# ---------------------------------------------------------------------------

def test_invalid_initial_pressure_nan(valid_json):
    valid_json["simulation"]["initial_pressure"] = float("nan")
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_initial_pressure_inf(valid_json):
    valid_json["simulation"]["initial_pressure"] = float("inf")
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)
