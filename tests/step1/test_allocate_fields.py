# tests/step1/test_allocate_fields.py

import numpy as np
from src.step1.allocate_fields import allocate_fields


def test_cell_centered_and_staggered_field_shapes():
    # Step 1 now uses simple dict-based grid metadata
    grid = {"nx": 4, "ny": 5, "nz": 6}

    fields = allocate_fields(grid)

    # Required keys
    for key in ["P", "U", "V", "W"]:
        assert key in fields, f"Missing field: {key}"

    # Shapes (matching the frozen Step 1 dummy)
    assert fields["P"].shape == (4, 5, 6)
    assert fields["U"].shape == (4 + 1, 5, 6)
    assert fields["V"].shape == (4, 5 + 1, 6)
    assert fields["W"].shape == (4, 5, 6 + 1)

    # Types
    for name, arr in fields.items():
        assert isinstance(arr, np.ndarray), f"{name} must be a numpy array"
        assert arr.dtype == float or arr.dtype == np.float64
