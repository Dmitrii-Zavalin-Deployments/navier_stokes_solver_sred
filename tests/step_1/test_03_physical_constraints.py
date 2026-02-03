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
# 1. Density constraints
# ---------------------------------------------------------------------------

def test_invalid_density_zero(valid_json):
    valid_json["fluid"]["density"] = 0.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_density_negative(valid_json):
    valid_json["fluid"]["density"] = -1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 2. Viscosity constraints
# ---------------------------------------------------------------------------

def test_invalid_viscosity_negative(valid_json):
    valid_json["fluid"]["viscosity"] = -0.01
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 3. Grid size constraints
# ---------------------------------------------------------------------------

def test_invalid_nx_zero(valid_json):
    valid_json["domain"]["nx"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_ny_zero(valid_json):
    valid_json["domain"]["ny"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_nz_zero(valid_json):
    valid_json["domain"]["nz"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 4. Domain extent constraints
# ---------------------------------------------------------------------------

def test_invalid_domain_degenerate(valid_json):
    valid_json["domain"]["x_min"] = 1.0
    valid_json["domain"]["x_max"] = 1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_domain_inverted(valid_json):
    valid_json["domain"]["x_min"] = 2.0
    valid_json["domain"]["x_max"] = 1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 5. CFL pre-check (optional)
# ---------------------------------------------------------------------------

def test_invalid_cfl_precheck(valid_json):
    valid_json["simulation"]["dt"] = 10.0
    valid_json["simulation"]["initial_velocity"] = [100.0, 0.0, 0.0]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)
