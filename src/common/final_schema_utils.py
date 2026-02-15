# src/common/final_schema_utils.py

from typing import Dict, Any
from src.solver_state import SolverState
from src.step1.schema_utils import load_schema, validate_json_schema


def validate_final_state(state: SolverState) -> None:
    """
    Validate the JSON-safe representation of SolverState against
    final_output_schema.json.

    Raises:
        ValidationError: if the final state does not match the schema.

    This function is optional during migration and can be enabled
    in debug mode or after all steps are fully migrated.
    """
    schema = load_schema("final_output_schema.json")
    payload: Dict[str, Any] = state.to_json_safe()

    # Optional context label improves error messages
    validate_json_schema(
        payload,
        schema,
        context_label="Final SolverState validation"
    )
