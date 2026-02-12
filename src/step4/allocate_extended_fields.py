# src/step4/allocate_extended_fields.py

import numpy as np


def allocate_extended_fields(state):
    """
    Allocate extended (halo) fields for Step 4.

    For each core 3D field present in the state (e.g. U, V, W, P, mask, is_fluid),
    this function allocates an extended array with a one-cell halo in each
    direction and copies the interior values into the [1:-1, 1:-1, 1:-1] region.

    New fields are stored under "<name>_ext" keys, e.g. "U_ext", "P_ext".
    The original (non-extended) fields are left unchanged.

    Parameters
    ----------
    state : dict-like
        Step 3 state dictionary, expected to contain at least
        state["config"]["domain"]["nx"], ["ny"], ["nz"], and some core fields.

    Returns
    -------
    state : dict-like
        The same state object with additional *_ext arrays.
    """
    # Known core 3D fields we may want to extend.
    candidate_fields = ("U", "V", "W", "P", "mask", "is_fluid")

    for name in candidate_fields:
        if name not in state:
            continue

        arr = state[name]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim != 3:
            # Only extend 3D fields here; others are left untouched.
            continue

        # Allocate extended array with a one-cell halo in each direction.
        ext_shape = tuple(s + 2 for s in arr.shape)
        ext = np.zeros(ext_shape, dtype=arr.dtype)

        # Copy interior into the extended field.
        ext[1:-1, 1:-1, 1:-1] = arr

        state[f"{name}_ext"] = ext

    return state
