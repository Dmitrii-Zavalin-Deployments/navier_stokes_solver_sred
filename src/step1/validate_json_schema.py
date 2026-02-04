import json
from pathlib import Path
from jsonschema import validate, ValidationError


SCHEMA_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "schema"
    / "input_schema.json"
)

with open(SCHEMA_PATH, "r") as f:
    INPUT_SCHEMA = json.load(f)


def validate_json_schema(json_input: dict) -> None:
    """
    Validate the raw JSON input against schema/input_schema.json,
    then apply a small extra consistency check that the tests
    expect: geometry_mask_shape must match (nx, ny, nz).
    """
    # 1. Schema validation
    validate(instance=json_input, schema=INPUT_SCHEMA)

    # 2. Extra check for geometry_mask_shape vs grid resolution
    dom = json_input["domain_definition"]
    geom = json_input["geometry_definition"]

    nx, ny, nz = dom["nx"], dom["ny"], dom["nz"]
    shape = geom["geometry_mask_shape"]

    if list(shape) != [nx, ny, nz]:
        # tests/step_1/test_02_schema_validation.py::test_geometry_mask_shape_mismatch
        raise ValidationError("geometry_mask_shape does not match grid resolution")
