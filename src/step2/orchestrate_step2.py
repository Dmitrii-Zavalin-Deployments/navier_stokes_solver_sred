# src/step2/orchestrate_step2.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Import all Step 2 functions
from .enforce_mask_semantics import enforce_mask_semantics
from .precompute_constants import precompute_constants
from .create_fluid_mask import create_fluid_mask
from .build_divergence_operator import build_divergence_operator
from .build_gradient_operators import build_gradient_operators
from .build_laplacian_operators import build_laplacian_operators
from .build_advection_structure import build_advection_structure
from .prepare_ppe_structure import prepare_ppe_structure
from .compute_initial_health import compute_initial_health

# Optional JSON-schema validation
try:  # pragma: no cover
    from ..step1.validate_json_schema import validate_json_schema
    from ..step1.schema_utils import load_schema
except Exception:  # pragma: no cover
    validate_json_schema = None
    load_schema = None


def orchestrate_step2(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure Step‑2 orchestrator.
    Input: Step‑1 output (dict with lists)
    Output: Step‑2 output (dict with lists)
    """

    # ------------------------------------------------------------
    # 1. Validate Step‑1 output BEFORE doing any computation
    # ------------------------------------------------------------
    if validate_json_schema and load_schema:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schema" / "step1_output_schema.json"
        )
        schema = load_schema(str(schema_path))
        validate_json_schema(state, schema)

    # ------------------------------------------------------------
    # 2. Precompute constants (pure)
    # ------------------------------------------------------------
    constants = precompute_constants(state)

    # ------------------------------------------------------------
    # 3. Enforce CFD mask semantics
    # ------------------------------------------------------------
    mask_semantics = enforce_mask_semantics(state)

    # ------------------------------------------------------------
    # 4. Create boolean fluid masks
    # ------------------------------------------------------------
    fluid_mask = create_fluid_mask(state)

    # ------------------------------------------------------------
    # 5. Build discrete operators (pure)
    # ------------------------------------------------------------
    divergence = build_divergence_operator(state)
    gradients = build_gradient_operators(state)
    laplacians = build_laplacian_operators(state)
    advection = build_advection_structure(state)

    # ------------------------------------------------------------
    # 6. Prepare PPE structure
    # ------------------------------------------------------------
    ppe = prepare_ppe_structure(state)

    # ------------------------------------------------------------
    # 7. Compute initial solver health diagnostics
    # ------------------------------------------------------------
    health = compute_initial_health(state)

    # ------------------------------------------------------------
    # 8. Assemble pure Step‑2 output
    # ------------------------------------------------------------
    output = {
        "constants": constants,
        "mask_semantics": mask_semantics,
        "fluid_mask": fluid_mask,
        "divergence": divergence,
        "pressure_gradients": gradients["pressure_gradients"],
        "laplacians": laplacians["laplacians"],
        "advection": advection["advection"],
        "ppe_structure": ppe,
        "health": health,
        "meta": {
            "step": 2,
            "description": "Step‑2 numerical preprocessing",
        },
    }

    # ------------------------------------------------------------
    # 9. Validate Step‑2 output (optional)
    # ------------------------------------------------------------
    if validate_json_schema and load_schema:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schema" / "step2_output_schema.json"
        )
        schema = load_schema(str(schema_path))
        validate_json_schema(output, schema)

    return output
