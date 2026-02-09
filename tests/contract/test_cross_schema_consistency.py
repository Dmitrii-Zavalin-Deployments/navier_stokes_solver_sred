# tests/contract/test_cross_schema_consistency.py

import numpy as np

from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState


def test_step2_output_has_keys_required_by_step3():
    """
    Step‑3 consumes Step‑2 output. This test ensures that the Step‑2 dummy
    contains all top‑level keys that Step‑3 expects in real operation.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    # These are the ACTUAL keys produced by Step‑2 and consumed by Step‑3.
    required = [
        "grid",
        "fields",
        "mask",
        "is_fluid",
        "is_solid",
        "constants",
        "config",
        "operators",
        "ppe",
        "health",
    ]

    for key in required:
        assert key in s2, f"Step‑2 dummy missing key required by Step‑3: {key}"


def test_step2_step3_pressure_shape_compatible():
    """
    Step‑3 expects pressure fields with the same shape as Step‑2.
    This test ensures the two dummies are structurally aligned.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    s3 = Step3SchemaDummyState(nx=3, ny=3, nz=3)

    P2 = np.asarray(s2["fields"]["P"])
    P3 = np.asarray(s3["fields"]["P"])

    assert P2.shape == P3.shape, "Pressure field shapes must match"
