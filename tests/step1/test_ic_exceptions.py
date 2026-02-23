# tests/step1/test_ic_exceptions.py

import pytest
import numpy as np
from src.step1.apply_initial_conditions import apply_initial_conditions

def test_ic_pressure_exception_handling():
    """Triggers lines 24-25: Invalid pressure types."""
    fields = {"P": np.zeros((2, 2, 2))}
    
    # Passing something that cannot be cast to float
    ic = {"pressure": "not_a_number"}
    
    with pytest.raises(ValueError, match="Invalid pressure initial condition"):
        apply_initial_conditions(fields, ic)

def test_ic_velocity_shape_validation():
    """Triggers lines 31-34: Velocity list length/type check."""
    fields = {"U": np.zeros((3, 2, 2)), "V": np.zeros((2, 3, 2)), "W": np.zeros((2, 2, 3))}
    
    # Passing 2 elements instead of 3
    ic = {"velocity": [1.0, 0.0]}
    
    with pytest.raises(ValueError, match="Initial velocity must be a 3-element list"):
        apply_initial_conditions(fields, ic)

def test_ic_velocity_cast_exception():
    """Triggers lines 43-44: Velocity component casting error."""
    fields = {"U": np.zeros((3, 2, 2)), "V": np.zeros((2, 3, 2)), "W": np.zeros((2, 2, 3))}
    
    # Correct length, but un-castable content
    ic = {"velocity": [1.0, "fast", 0.0]}
    
    with pytest.raises(ValueError, match="Could not cast initial velocity components to float"):
        apply_initial_conditions(fields, ic)