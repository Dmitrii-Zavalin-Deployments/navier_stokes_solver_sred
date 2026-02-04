import numpy as np
from jsonschema import ValidationError
from .simulation_state import Grid


def _validate_mask_values(mask_flat):
    """
    Validate that mask values are either:
    - binary {0,1}, or
    - pattern values (no 0 or 1), but not mixed.
    """
    values = set(mask_flat)

    # Negative values are always invalid
    if any(v < 0 for v in values):
        raise ValueError("geometry_mask_flat contains invalid values")

    # Pure binary mask {0,1}
    if values <= {0, 1}:
        return

    # Pure pattern mask (no 0 or 1)
    if 0 not in values and 1 not in values:
        return

    # Mixed binary + pattern → invalid
    raise ValueError("geometry_mask_flat contains invalid values")


def map_geometry_mask(config: dict, grid: Grid) -> np.ndarray:
    geom = config["geometry_definition"]
    flat = geom["geometry_mask_flat"]
    shape = geom["geometry_mask_shape"]

    # 1. Shape must be 3D
    if len(shape) != 3:
        raise ValueError("geometry_mask_shape must have length 3")

    # 2. Shape must match grid resolution
    if list(shape) != [grid.nx, grid.ny, grid.nz]:
        # Tests expect ValidationError here
        raise ValidationError("geometry_mask_shape does not match grid resolution")

    # 3. Length must match shape product
    expected_len = grid.nx * grid.ny * grid.nz
    if len(flat) != expected_len:
        raise ValueError("geometry_mask_flat length mismatch")

    # 4. Validate mask values (ValueError per tests)
    _validate_mask_values(flat)

    # 5. Reshape into 3D mask
    mask = np.array(flat, dtype=int).reshape(shape)
    return mask