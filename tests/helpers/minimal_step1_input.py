# tests/helpers/minimal_step1_input.py

"""
Minimal valid Step‑1 input used across tests.
"""

def minimal_step1_input():
    return {
        "domain_definition": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 2, "ny": 2, "nz": 2
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.1
        },
        "initial_conditions": {
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_pressure": 0.0
        },
        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 1.0,
            "output_interval": 1
        },
        "boundary_conditions": [
            {
                "role": "wall",
                "type": "dirichlet",
                "faces": ["x_min"],
                "apply_to": ["velocity", "pressure"],
                "velocity": [0.0, 0.0, 0.0],
                "pressure": 0.0,
                "pressure_gradient": 0.0,
                "no_slip": True,
                "comment": "minimal BC"
            }
        ],
        "geometry_definition": {
            "geometry_mask_flat": [1, 1, 1, 1, 1, 1, 1, 1],
            "geometry_mask_shape": [2, 2, 2],
            "mask_encoding": {"fluid": 1, "solid": -1},
            "flattening_order": "C"
        },
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "units": "N",
            "comment": "no external forces"
        }
    }

# Backwards compatibility for existing Step‑1 tests
MINIMAL_VALID_INPUT = minimal_step1_input()
