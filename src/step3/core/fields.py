# src/step3/core/fields.py

import numpy as np

def create_velocity_field(nx, ny, nz, initial_value):
    """
    Mandatory initialization of a velocity field.
    Returns a (3, nx, ny, nz) numpy array.
    """
    # Force full initialization to avoid junk data
    return np.full((3, nx, ny, nz), initial_value, dtype=np.float64)

def create_pressure_field(nx, ny, nz, initial_value):
    """
    Mandatory initialization of a pressure field.
    Returns an (nx, ny, nz) numpy array.
    """
    return np.full((nx, ny, nz), initial_value, dtype=np.float64)