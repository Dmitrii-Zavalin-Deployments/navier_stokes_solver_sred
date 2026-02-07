# tests/step1/test_schema_utils.py
import json
import pytest
from src.step1.schema_utils import load_schema, validate_with_schema
from jsonschema import ValidationError


def test_load_schema():
    schema = load_schema("schema/input_schema.json")
    assert isinstance(schema, dict)
    assert "properties" in schema


def test_validate_with_schema():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "number"}},
        "required": ["x"]
    }

    validate_with_schema({"x": 5}, schema)

    with pytest.raises(ValidationError):
        validate_with_schema({}, schema)
