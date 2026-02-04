import numpy as np
from jsonschema import ValidationError
from .simulation_state import Grid

def map_geometry_mask(config: dict, grid: Grid) -> np.ndarray:
    geom = config["geometry_definition"]
    flat = geom["geometry_mask_flat"]
    shape = geom["geometry_mask_shape"]

    # 1. Shape must be 3D
    if len(shape) != 3:
        raise ValueError("geometry_mask_shape must have length 3")

    # 2. Shape must match grid resolution
    if list(shape) != [grid.nx, grid.ny, grid.nz]:
        raise ValidationError("geometry_mask_shape does not match grid resolution")

    # 3. Length must match shape product
    expected_len = grid.nx * grid.ny * grid.nz
    if len(flat) != expected_len:
        raise ValueError("geometry_mask_flat length mismatch")

    # 4. Step 1 value validation rules
    values = set(flat)

    # Negative values always invalid in Step 1
    if any(v < 0 for v in values):
        raise ValueError("geometry_mask_flat contains invalid values")

    # Pure binary mask {0,1} → OK
    if values <= {0, 1}:
        pass

    # Pure pattern mask (no 0 or 1) → OK
    elif 0 not in values and 1 not in values:
        pass

    # Mixed binary + pattern → invalid
    else:
        raise ValueError("geometry_mask_flat contains invalid values")

    # 5. Reshape into 3D mask
    return np.array(flat, dtype=int).reshape(shape)
