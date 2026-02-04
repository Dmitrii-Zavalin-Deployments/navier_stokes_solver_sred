import pytest

@pytest.fixture
def sample_json_input():
    nx = ny = nz = 4
    return {
        "domain_definition": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 1.0,
            "nx": nx,
            "ny": ny,
            "nz": nz,
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.1,
        },
        "initial_conditions": {
            "initial_velocity": [1.0, 0.0, 0.0],
            "initial_pressure": 0.0,
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 1.0,
            "output_interval": 10,
        },
        "boundary_conditions": [
            {
                "role": "wall",
                "type": "dirichlet",
                "faces": ["x_min"],
                "apply_to": ["velocity"],
                "no_slip": True,
                "velocity": [0.0, 0.0, 0.0],
                "comment": "test",
            }
        ],
        "geometry_definition": {
            "geometry_mask_flat": [1] * (nx * ny * nz),
            "geometry_mask_shape": [nx, ny, nz],
            "flattening_order": "i + nx*(j + ny*k)",
        },
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "units": "N",
            "comment": "none",
        },
    }
