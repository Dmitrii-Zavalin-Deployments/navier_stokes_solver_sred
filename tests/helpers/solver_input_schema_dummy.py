# tests/helpers/solver_input_schema_dummy.py

"""
solver_input_schema_dummy.py
Updated for Constitutional Closure: Now provides all 6 mandatory faces.
"""

def solver_input_schema_dummy():
    nx, ny, nz = 2, 2, 2
    total_cells = nx * ny * nz
    mask_flat = [0, -1, 1, -1, 0, 1, 0, -1]

    return {
        "grid": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": nx, "ny": ny, "nz": nz,
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.1,
        },
        "initial_conditions": {
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
        },
        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 1.0,
            "output_interval": 1,
        },
        "boundary_conditions": [
            {"location": "x_min", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
            {"location": "x_max", "type": "outflow", "values": {}},
            {"location": "y_min", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
            {"location": "y_max", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
            {"location": "z_min", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
            {"location": "z_max", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
        ],
        "mask": mask_flat,
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
        },
    }
