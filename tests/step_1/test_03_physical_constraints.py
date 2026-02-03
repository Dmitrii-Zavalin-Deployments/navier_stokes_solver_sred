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
# 1. Density constraints
# ---------------------------------------------------------------------------

def test_invalid_density_zero(valid_json):
    valid_json["fluid_properties"]["density"] = 0.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_density_negative(valid_json):
    valid_json["fluid_properties"]["density"] = -1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 2. Viscosity constraints
# ---------------------------------------------------------------------------

def test_invalid_viscosity_negative(valid_json):
    valid_json["fluid_properties"]["viscosity"] = -0.01
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 3. Grid size constraints
# ---------------------------------------------------------------------------

def test_invalid_nx_zero(valid_json):
    valid_json["domain_definition"]["nx"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_ny_zero(valid_json):
    valid_json["domain_definition"]["ny"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_nz_zero(valid_json):
    valid_json["domain_definition"]["nz"] = 0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 4. Domain extent constraints
# ---------------------------------------------------------------------------

def test_invalid_domain_degenerate(valid_json):
    valid_json["domain_definition"]["x_min"] = 1.0
    valid_json["domain_definition"]["x_max"] = 1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_domain_inverted(valid_json):
    valid_json["domain_definition"]["x_min"] = 2.0
    valid_json["domain_definition"]["x_max"] = 1.0
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 5. CFL pre-check
# ---------------------------------------------------------------------------

def test_invalid_cfl_precheck(valid_json):
    valid_json["simulation_parameters"]["time_step"] = 10.0
    valid_json["initial_conditions"]["initial_velocity"] = [100.0, 0.0, 0.0]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)
