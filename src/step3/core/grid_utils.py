# src/step3/core/grid_utils.py

def get_interior_slices():
    """
    Returns a tuple of slices for interior grid access.
    Usage: p[get_interior_slices()] = ...
    """
    return (slice(1, -1), slice(1, -1), slice(1, -1))