# tests/step1/test_step1_optional.py

import numpy as np
import pytest

from src.step1.verify_staggered_shapes import verify_staggered_shapes
from src.step1.map_geometry_mask import map_geometry_mask


def test_invalid_geometry_mask_shape():
    """
    Step‑1 must reject geometry masks with invalid shapes.
    """
    nx, ny, nz = 4, 4, 4
    bad_mask = np.ones((nx + 1, ny, nz))  # wrong shape

    with pytest.raises(Exception):
        map_geometry_mask({"grid": {"nx": nx, "ny": ny, "nz": nz}}, bad_mask)


def test_verify_staggered_shapes_rejects_wrong_U_shape():
    """
    verify_staggered_shapes must reject incorrect staggered field shapes.
    """
    nx, ny, nz = 3, 3, 3

    fields = {
        "U": np.zeros((nx, ny, nz)),        # WRONG: should be (nx+1, ny, nz)
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz)),
    }

    with pytest.raises(Exception):
        verify_staggered_shapes({"grid": {"nx": nx, "ny": ny, "nz": nz}}, fields)
