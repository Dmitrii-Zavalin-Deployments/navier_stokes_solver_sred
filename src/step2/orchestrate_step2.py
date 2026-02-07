# src/step2/orchestrate_step2.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

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
    validate_json_schema = None  # type: ignore
    load_schema = None  # type: ignore


def _convert_fields_to_numpy(state: Any) -> None:
    """Convert Step‑1 Python lists into NumPy arrays for Step‑2 numerical code."""
    fields = state.get("fields", {})
    for key, value in fields.items():
        if not isinstance(value, np.ndarray):
            fields[key] = np.asarray(value)


def orchestrate_step2(state: Any) -> Any:
    """
    High-level orchestrator for Step 2.

    Validates:
      - Input against step1_output_schema.json (lists)
      - Output against step2_output_schema.json (arrays allowed)
    """

    # ------------------------------------------------------------
    # 0. Validate Step‑1 output (lists)
    # ------------------------------------------------------------
    if isinstance(state, dict) and validate_json_schema is not None and load_schema is not None:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schema" / "step1_output_schema.json"
        )
        try:
            schema = load_schema(str(schema_path))
            validate_json_schema(state, schema)
        except Exception as exc:
            raise RuntimeError(
                f"\n[Step 2] Input schema validation FAILED.\n"
                f"Expected schema: {schema_path}\n"
                f"Validation error: {exc}\n"
                f"Aborting Step 2 — upstream Step 1 output is malformed.\n"
            ) from exc

    # ------------------------------------------------------------
    # 1. Convert lists → NumPy arrays (Step‑2 numerical world)
    # ------------------------------------------------------------
    _convert_fields_to_numpy(state)

    # ------------------------------------------------------------
    # 2. Precompute constants
    # ------------------------------------------------------------
    precompute_constants(state)

    # ------------------------------------------------------------
    # 3. Enforce CFD mask semantics
    # ------------------------------------------------------------
    enforce_mask_semantics(state)

    # ------------------------------------------------------------
    # 4. Create boolean fluid masks
    # ------------------------------------------------------------
    create_fluid_mask(state)

    # ------------------------------------------------------------
    # 5. Provide top-level mask required by Step‑2 schema
    # ------------------------------------------------------------
    state["mask"] = state["fields"]["Mask"]

    # ------------------------------------------------------------
    # 6. Build discrete operators
    # ------------------------------------------------------------
    build_divergence_operator(state)
    build_gradient_operators(state)
    build_laplacian_operators(state)
    build_advection_structure(state)

    # ------------------------------------------------------------
    # 7. Prepare PPE structure
    # ------------------------------------------------------------
    prepare_ppe_structure(state)

    # ------------------------------------------------------------
    # 8. Compute initial solver health diagnostics
    # ------------------------------------------------------------
    compute_initial_health(state)

    # ------------------------------------------------------------
    # 9. Validate Step‑2 output (arrays allowed)
    # ------------------------------------------------------------
    if isinstance(state, dict) and validate_json_schema is not None and load_schema is not None:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schema" / "step2_output_schema.json"
        )
        try:
            schema = load_schema(str(schema_path))
            validate_json_schema(state, schema)
        except Exception as exc:
            raise RuntimeError(
                f"\n[Step 2] Output schema validation FAILED.\n"
                f"Expected schema: {schema_path}\n"
                f"Validation error: {exc}\n"
                f"Aborting — Step 2 produced an invalid SimulationState.\n"
            ) from exc

    return state
