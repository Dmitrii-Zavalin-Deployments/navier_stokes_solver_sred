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
# 1. dx, dy, dz must be positive
# ---------------------------------------------------------------------------

def test_grid_spacing_positive(valid_json):
    state = construct_simulation_state(valid_json)
    assert state.grid.dx > 0


# ---------------------------------------------------------------------------
# 2. dx must match domain extents
# ---------------------------------------------------------------------------

def test_grid_spacing_correct(valid_json):
    dom = valid_json["domain"]
    expected_dx = abs(dom["x_max"] - dom["x_min"]) / dom["nx"]

    state = construct_simulation_state(valid_json)
    assert abs(state.grid.dx - expected_dx) < 1e-12


# ---------------------------------------------------------------------------
# 3. Degenerate domain must raise
# ---------------------------------------------------------------------------

def test_grid_invalid_zero_extent(valid_json):
    valid_json["domain"]["x_min"] = 1.0
    valid_json["domain"]["x_max"] = 1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 4. Inverted domain must raise
# ---------------------------------------------------------------------------

def test_grid_invalid_inverted_extent(valid_json):
    valid_json["domain"]["x_min"] = 2.0
    valid_json["domain"]["x_max"] = 1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 5. Non-positive grid resolution must raise
# ---------------------------------------------------------------------------

def test_grid_invalid_resolution_nx(valid_json):
    valid_json["domain"]["nx"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_grid_invalid_resolution_ny(valid_json):
    valid_json["domain"]["ny"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_grid_invalid_resolution_nz(valid_json):
    valid_json["domain"]["nz"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)
