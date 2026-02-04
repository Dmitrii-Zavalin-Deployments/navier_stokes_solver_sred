import numpy as np
from jsonschema import ValidationError
from .simulation_state import Grid


def map_geometry_mask(config: dict, grid: Grid) -> np.ndarray:
    """
    Map the flat geometry mask from the JSON config into a 3D numpy array
    with shape (nx, ny, nz). Step 1 does not validate mask values, only
    shape and length, because tests use arbitrary integers (range(...))
    to verify reshape behavior.
    """
    geom = config["geometry_definition"]
    flat = geom["geometry_mask_flat"]
    shape = geom["geometry_mask_shape"]

    # 1. Shape must be 3D
    if len(shape) != 3:
        raise ValueError("geometry_mask_shape must have length 3")

    # 2. Shape must match grid resolution
    if list(shape) != [grid.nx, grid.ny, grid.nz]:
        # Safety net; schema-level check already exists in validate_json_schema
        raise ValidationError("geometry_mask_shape does not match grid resolution")

    # 3. Length must match shape product
    expected_len = grid.nx * grid.ny * grid.nz
    if len(flat) != expected_len:
        raise ValueError("geometry_mask_flat length mismatch")

    # 4. Reshape into 3D mask (no value validation in Step 1)
    mask = np.array(flat, dtype=int).reshape(shape)
    return mask
