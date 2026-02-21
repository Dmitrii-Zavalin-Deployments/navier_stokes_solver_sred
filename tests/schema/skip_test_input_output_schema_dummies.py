# tests/schema/test_input_output_schema_dummies.py

"""
test_schema_dummies.py

Validates that the canonical solver input/output dummies fully satisfy
their corresponding JSON schemas.

This ensures:
  • the schemas remain valid,
  • the dummies remain aligned with the schemas,
  • refactors cannot silently break the solver’s external contract.

These tests act as contract guards for the solver pipeline.
"""

import json
import jsonschema
from pathlib import Path

from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_output_schema_dummy import solver_output_schema_dummy


# Path to the schema directory
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"


def load_schema(filename: str):
    """Load a JSON schema from the schema directory."""
    path = SCHEMA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_solver_input_dummy_matches_schema():
    schema = load_schema("solver_input_schema.json")
    dummy = solver_input_schema_dummy()

    # Raises jsonschema.ValidationError on failure
    jsonschema.validate(instance=dummy, schema=schema)


def test_solver_output_dummy_matches_schema():
    schema = load_schema("solver_output_schema.json")
    dummy = solver_output_schema_dummy()

    # Raises jsonschema.ValidationError on failure
    jsonschema.validate(instance=dummy, schema=schema)
