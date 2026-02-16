# tests/helpers/solver_input_schema_dummy.py

"""
solver_input_schema_dummy.py

Canonical JSON‑safe dummy input that fully satisfies solver_input_schema.json.
Used for Step 1 tests, schema validation tests, and as a base for override‑based
unit tests (e.g., boundary conditions, mask validation, domain validation).
"""

def solver_input_schema_dummy():
    # Small, simple domain
    nx, ny, nz = 2, 2, 2

    # Helper to build nested lists of integers for mask
    def ints(shape, value=1):
        if len(shape) == 1:
            return [value for _ in range(shape[0])]
        return [ints(shape[1:], value) for _ in range(shape[0])]

    return {
        "domain": {
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
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
        },

        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 1.0,
            "output_interval": 1,
        },

        # Minimal valid BC list: empty list is allowed
        "boundary_conditions": [],

        # 3D mask of allowed values {-1, 0, 1}
        "mask": ints((nx, ny, nz), value=1),

        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
        },
    }
