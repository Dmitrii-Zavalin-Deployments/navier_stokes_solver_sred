import numpy as np
from .simulation_state import Grid


def _validate_mask_values(mask_flat):
    values = set(mask_flat)
    if any(v < 0 for v in values):
        raise ValueError("geometry_mask_flat contains invalid values")

    # Binary mask {0,1}
    if values <= {0, 1}:
        return

    # Pattern mask: no 0 or 1 allowed
    if 0 not in values and 1 not in values:
        return

    # Mixed binary/pattern
    raise ValueError("geometry_mask_flat contains invalid values")


def map_geometry_mask(config: dict, grid: Grid) -> np.ndarray:
    geom = config["geometry_definition"]
    flat = geom["geometry_mask_flat"]
    shape = geom["geometry_mask_shape"]

    expected_len = grid.nx * grid.ny * grid.nz
    if len(flat) != expected_len:
        raise ValueError("geometry_mask_flat length mismatch")

    if len(shape) != 3:
        raise ValueError("geometry_mask_shape must have length 3")

    _validate_mask_values(flat)

    mask = np.array(flat, dtype=int).reshape(shape)
    return mask
