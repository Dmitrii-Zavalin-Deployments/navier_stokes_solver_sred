# tests/helpers/solver_input_schema_dummy.py

from src.solver_input import SolverInput

def solver_input_schema_dummy() -> dict:
    """
    Returns a raw dictionary representing valid JSON input.
    Use this for integration tests or file-loading tests.
    """
    nx, ny, nz = 2, 2, 2
    mask_flat = [0] * (nx * ny * nz) # All fluid for the dummy

    return {
        "grid": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": nx, "ny": ny, "nz": nz,
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.001,
        },
        "initial_conditions": {
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 0.1,
            "output_interval": 1,
        },
        "boundary_conditions": [
            {"location": "x_min", "type": "no-slip", "values": {"u": 0.0}, "comment": "wall"},
            {"location": "x_max", "type": "outflow", "values": {}, "comment": "exit"},
            {"location": "y_min", "type": "no-slip", "values": {"v": 0.0}, "comment": "wall"},
            {"location": "y_max", "type": "no-slip", "values": {"v": 0.0}, "comment": "wall"},
            {"location": "z_min", "type": "no-slip", "values": {"w": 0.0}, "comment": "wall"},
            {"location": "z_max", "type": "no-slip", "values": {"w": 0.0}, "comment": "wall"},
        ],
        "mask": mask_flat,
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "comment": "No gravity"
        },
    }

def make_solver_input_dummy() -> SolverInput:
    """
    Returns a fully hydrated SolverInput OBJECT.
    Use this for unit testing orchestrate_step1 directly.
    """
    raw_dict = solver_input_schema_dummy()
    return SolverInput.from_dict(raw_dict)