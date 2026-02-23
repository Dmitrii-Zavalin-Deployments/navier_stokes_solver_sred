# tests/step1/test_verify_shapes.py

import pytest
import numpy as np
from src.step1.verify_cell_centered_shapes import verify_cell_centered_shapes

@pytest.fixture
def valid_state():
    """Provides a valid Step 1 state dictionary with lists that need conversion."""
    nx, ny, nz = 2, 2, 2
    # Create flat lists that resemble JSON-parsed data
    data_size = nx * ny * nz
    return {
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "fields": {
            "P": [0.0] * data_size,
            "U": [0.0] * data_size,
            "V": [0.0] * data_size,
            "W": [0.0] * data_size,
            "Mask": [1] * data_size
        }
    }

def test_successful_conversion_and_verification(valid_state):
    """Verifies that lists are converted to arrays and shapes are validated."""
    # Reshape lists to match expected 3D structure for a clean pass
    for key in ["P", "U", "V", "W"]:
        valid_state["fields"][key] = np.zeros((2, 2, 2)).tolist()
    valid_state["fields"]["Mask"] = np.ones((2, 2, 2), dtype=np.int8).tolist()

    verify_cell_centered_shapes(valid_state)

    fields = valid_state["fields"]
    # Check types (Requirement: np.float64 and np.int8)
    assert isinstance(fields["P"], np.ndarray)
    assert fields["P"].dtype == np.float64
    assert fields["Mask"].dtype == np.int8
    # Check shapes
    assert fields["P"].shape == (2, 2, 2)

def test_missing_grid_returns_early():
    """Ensures the guard skips processing if 'grid' key is missing (Line 18-19)."""
    state = {"fields": {}}
    # Should not raise any error
    assert verify_cell_centered_shapes(state) is None

def test_malformed_grid_returns_early():
    """Ensures guard skips if grid values are not castable to int (Line 24-27)."""
    state = {
        "grid": {"nx": "invalid", "ny": 2, "nz": 2},
        "fields": {}
    }
    assert verify_cell_centered_shapes(state) is None

def test_missing_essential_field_raises_keyerror(valid_state):
    """Checks that missing 'P', 'U', 'V', 'W', or 'Mask' raises KeyError."""
    del valid_state["fields"]["P"]
    with pytest.raises(KeyError, match="Missing essential field"):
        verify_cell_centered_shapes(valid_state)

def test_invalid_field_type_raises_typeerror(valid_state):
    """Ensures fields that are neither lists nor arrays raise TypeError."""
    valid_state["fields"]["P"] = "not a list"
    with pytest.raises(TypeError, match="must be a list or numpy array"):
        verify_cell_centered_shapes(verify_cell_centered_shapes(valid_state))

def test_dimension_mismatch_raises_value_error(valid_state):
    """Checks that arrays with wrong dimensions trigger the logic firewall."""
    # Grid expects (2, 2, 2), we give (3, 2, 2)
    valid_state["fields"]["P"] = np.zeros((3, 2, 2))
    
    # We must ensure other fields are valid arrays or lists to reach the shape check
    for key in ["U", "V", "W", "Mask"]:
        valid_state["fields"][key] = np.zeros((2, 2, 2))

    with pytest.raises(ValueError, match="Dimension Mismatch"):
        verify_cell_centered_shapes(valid_state)

def test_already_numpy_arrays_pass(valid_state):
    """Ensures the function handles states that already contain numpy arrays."""
    nx, ny, nz = 2, 2, 2
    valid_state["fields"]["P"] = np.zeros((nx, ny, nz))
    valid_state["fields"]["U"] = np.zeros((nx, ny, nz))
    valid_state["fields"]["V"] = np.zeros((nx, ny, nz))
    valid_state["fields"]["W"] = np.zeros((nx, ny, nz))
    valid_state["fields"]["Mask"] = np.zeros((nx, ny, nz), dtype=np.int8)

    # This should pass without raising exceptions
    verify_cell_centered_shapes(valid_state)
    assert isinstance(valid_state["fields"]["P"], np.ndarray)