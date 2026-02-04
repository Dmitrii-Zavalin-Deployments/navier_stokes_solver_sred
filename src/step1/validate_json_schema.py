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
    # 1. Schema validation
    validate(instance=json_input, schema=INPUT_SCHEMA)

    # 2. Extra check required by test_geometry_mask_shape_mismatch
    dom = json_input["domain_definition"]
    geom = json_input["geometry_definition"]

    nx, ny, nz = dom["nx"], dom["ny"], dom["nz"]
    shape = geom["geometry_mask_shape"]

    if list(shape) != [nx, ny, nz]:
        raise ValidationError("geometry_mask_shape does not match grid resolution")
