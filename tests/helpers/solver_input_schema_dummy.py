# tests/helpers/solver_input_schema_dummy.py

"""
solver_input_schema_dummy.py

Canonical JSON‑safe dummy input that fully satisfies solver_input_schema.json.
Used for Step 1 tests, schema validation tests, and as a base for override‑based
unit tests (e.g., boundary conditions, mask validation, grid validation).

Updated for the new contract:
- mask is now a flat 1D array of length nx*ny*nz
- mask values ∈ {-1, 0, 1}
- canonical flattening rule is i + nx*(j + ny*k)
- boundary_conditions now includes numerical 'values' for Step 2/3 parity.
- Physical Logic Fix: 'outflow' type no longer defines 'p' to avoid validation errors.
"""

def solver_input_schema_dummy():
    # Small, simple grid
    nx, ny, nz = 2, 2, 2
    total_cells = nx * ny * nz

    # Intuitive, human-readable flat mask (length = 8)
    # Demonstrates all allowed values: -1, 0, 1
    mask_flat = [
        0,   # cell 0
        -1,  # cell 1
        1,   # cell 2
        -1,  # cell 3
        0,   # cell 4
        1,   # cell 5
        0,   # cell 6
        -1   # cell 7
    ]

    # Safety: ensure length matches nx*ny*nz
    assert len(mask_flat) == total_cells

    return {
        "grid": {
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

        # Updated: Non-empty list with mandatory 'values' sub-dictionary
        # Note: 'outflow' values dict is empty to satisfy parse_boundary_conditions.py logic
        "boundary_conditions": [
            {
                "location": "x_min", 
                "type": "no-slip", 
                "values": {"u": 0.0, "v": 0.0, "w": 0.0}
            },
            {
                "location": "x_max", 
                "type": "outflow", 
                "values": {} 
            }
        ],

        # Flat mask (canonical)
        "mask": mask_flat,

        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
        },
    }