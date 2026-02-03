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
    valid_json["domain"]["nx"] = 1
    valid_json["domain"]["ny"] = 1
    valid_json["domain"]["nz"] = 1
    valid_json["geometry_mask_flat"] = [1]

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
    valid_json["fluid"]["viscosity"] = 0.0

    state = construct_simulation_state(valid_json)

    assert state.constants["mu"] == 0.0


# ---------------------------------------------------------------------------
# 4. valid_negative_bounds
# ---------------------------------------------------------------------------

def test_valid_negative_bounds(valid_json):
    """Negative coordinates must still produce positive dx."""
    valid_json["domain"]["x_min"] = -1.0
    valid_json["domain"]["x_max"] = 1.0

    state = construct_simulation_state(valid_json)

    assert state.grid.dx > 0


# ---------------------------------------------------------------------------
# 5. valid_custom_flattening
# ---------------------------------------------------------------------------

def test_valid_custom_flattening(valid_json):
    """Custom flattening_order must reshape mask correctly."""
    valid_json["simulation"]["flattening_order"] = "j + ny*(i + nx*k)"

    state = construct_simulation_state(valid_json)

    # Mask is still placeholder but must have correct shape
    assert state.mask.shape == (4, 4, 4)
