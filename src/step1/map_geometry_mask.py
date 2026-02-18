def map_geometry_mask(mask_3d, domain):
    """
    Step 1: accept a 3D mask in canonical (i, j, k) order.

    The input schema defines mask as a 3D array, so no flattening or
    unflattening logic is needed. We simply validate shape and values.
    """

    nx = int(domain["nx"])
    ny = int(domain["ny"])
    nz = int(domain["nz"])

    # Convert to array
    arr = np.asarray(mask_3d, dtype=int)

    # Validate shape
    if arr.shape != (nx, ny, nz):
        raise ValueError(
            f"Mask shape {arr.shape} does not match expected {(nx, ny, nz)}"
        )

    # Validate values
    if not np.isin(arr, [-1, 0, 1]).all():
        raise ValueError("Mask entries must be -1, 0, or 1")

    return arr
