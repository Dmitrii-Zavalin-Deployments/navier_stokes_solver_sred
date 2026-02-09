# tests/contract/test_cross_schema_consistency.py

import numpy as np

from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState


def test_step2_output_has_keys_required_by_step3():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    s3 = Step3SchemaDummyState(nx=3, ny=3, nz=3)

    required = [
        "fields",
        "mask_semantics",
        "constants",
        "advection",
        "laplacians",
        "divergence",
        "ppe_structure",
        "config",
    ]

    for key in required:
        assert key in s2, f"Step‑2 dummy missing key required by Step‑3: {key}"


def test_step2_step3_pressure_shape_compatible():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    s3 = Step3SchemaDummyState(nx=3, ny=3, nz=3)

    assert s2["fields"]["P"].shape == s3["fields"]["P"].shape
    assert s2["mask_semantics"]["mask"].shape == s3["mask_semantics"]["mask"].shape
