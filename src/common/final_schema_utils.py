# src/common/final_schema_utils.py

from typing import Dict, Any

from src.step1.schema_utils import load_schema
from src.step1.validate_json_schema import validate_json_schema


def validate_final_state(state) -> None:
    """
    Validate the final SolverState against final_output_schema.json.
    """

    schema = load_schema("final_output_schema.json")

    payload: Dict[str, Any] = state.to_json_safe()

    validate_json_schema(
        data=payload,
        schema=schema,
    )
