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
# 1. Pressure field shape
# ---------------------------------------------------------------------------

def test_pressure_field_shape(valid_json):
    state = construct_simulation_state(valid_json)
    assert state.P.shape == (4, 4, 4)


# ---------------------------------------------------------------------------
# 2. Velocity field shapes (MAC grid)
# ---------------------------------------------------------------------------

def test_u_field_shape(valid_json):
    state = construct_simulation_state(valid_json)
    assert state.U.shape == (5, 4, 4)  # nx+1, ny, nz


def test_v_field_shape(valid_json):
    state = construct_simulation_state(valid_json)
    assert state.V.shape == (4, 5, 4)  # nx, ny+1, nz


def test_w_field_shape(valid_json):
    state = construct_simulation_state(valid_json)
    assert state.W.shape == (4, 4, 5)  # nx, ny, nz+1


# ---------------------------------------------------------------------------
# 3. Fields must be zero-initialized
# ---------------------------------------------------------------------------

def test_fields_zero_initialized(valid_json):
    state = construct_simulation_state(valid_json)

    assert (state.P == 0).all()
    assert (state.U == 0).all()
    assert (state.V == 0).all()
    assert (state.W == 0).all()


# ---------------------------------------------------------------------------
# 4. Minimal grid still allocates correctly
# ---------------------------------------------------------------------------

def test_minimal_grid_allocation(valid_json):
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


# ---------------------------------------------------------------------------
# 5. Invalid grid sizes must raise
# ---------------------------------------------------------------------------

def test_invalid_grid_negative_nx(valid_json):
    valid_json["domain_definition"]["nx"] = -1
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_grid_negative_ny(valid_json):
    valid_json["domain_definition"]["ny"] = -1
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_grid_negative_nz(valid_json):
    valid_json["domain_definition"]["nz"] = -1
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)
