# file: src/step1/schema_utils.py
import json
from pathlib import Path
from jsonschema import validate


def load_schema(path: str) -> dict:
    schema_path = Path(path)
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_with_schema(instance: dict, schema: dict) -> None:
    validate(instance=instance, schema=schema)
