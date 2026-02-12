# tests/contract/test_cross_schema_consistency.py

import numpy as np

from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState


def test_step2_output_has_keys_required_by_step3():
    """
    Step‑3 consumes Step‑2 output. This test ensures that the Step‑2 dummy
    contains all top‑level keys that Step‑3 actually relies on.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    required = [
        "grid",
        "fields",
        "mask",
        "is_fluid",
        # 'is_solid' is NOT present in Step‑2 and not required by Step‑3.
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


def test_step3_output_has_keys_required_by_step4():
    """
    Step‑4 consumes Step‑3 output. This test ensures that the Step‑3 dummy
    contains all top‑level keys that Step‑4 actually relies on.
    """
    s3 = Step3SchemaDummyState(nx=3, ny=3, nz=3)

    required = [
        "fields",       # interior u, v, w, p
        "constants",    # rho, mu, dt, dx, dy, dz
        "config",       # domain, BC definitions, simulation parameters
        "operators",    # divergence, gradients, Laplacians, advection
        "mask",         # geometry mask
        "is_fluid",     # fluid mask
        "health",       # diagnostics
    ]

    for key in required:
        assert key in s3, f"Step‑3 dummy missing key required by Step‑4: {key}"


# ---------------------------------------------------------------------------
# FUTURE TEST — Step‑4 → Step‑5 contract
# ---------------------------------------------------------------------------
# This test is commented out because Step‑4 is still under development.
# Once Step‑4 dummy state exists, uncomment this test.
#
# def test_step4_output_has_keys_required_by_step5():
#     """
#     Step‑5 consumes Step‑4 output. This test ensures that the Step‑4 dummy
#     contains all top‑level keys that Step‑5 actually relies on.
#     """
#     from tests.helpers.step4_schema_dummy_state import Step4SchemaDummyState
#     s4 = Step4SchemaDummyState(nx=3, ny=3, nz=3)
#
#     required = [
#         "U_ext",
#         "V_ext",
#         "W_ext",
#         "P_ext",
#         "Domain",
#         "BCs",
#         "RHS_Source",
#         "constants",
#         "config",
#         "operators",
#         "mask",
#         "is_fluid",
#         "health",
#         "time",
#         "step_index",
#     ]
#
#     for key in required:
#         assert key in s4, f"Step‑4 dummy missing key required by Step‑5: {key}"
