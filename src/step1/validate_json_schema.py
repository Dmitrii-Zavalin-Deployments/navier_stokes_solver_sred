from jsonschema import ValidationError


def validate_json_schema(data: dict) -> None:
    """
    Lightweight schema-like validation that raises jsonschema.ValidationError
    for the structural issues our tests expect.
    This is a stand-in for a full JSON Schema file.
    """
    required_top = [
        "domain_definition",
        "fluid_properties",
        "initial_conditions",
        "simulation_parameters",
        "boundary_conditions",
        "geometry_definition",
        "external_forces",
    ]
    for key in required_top:
        if key not in data:
            raise ValidationError(f"Missing top-level key: {key}")

    dom = data["domain_definition"]
    for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]:
        if k not in dom:
            raise ValidationError(f"Missing domain_definition.{k}")

    if not isinstance(dom["nx"], int):
        raise ValidationError("nx must be integer")
    if not isinstance(dom["ny"], int):
        raise ValidationError("ny must be integer")
    if not isinstance(dom["nz"], int):
        raise ValidationError("nz must be integer")

    ic = data["initial_conditions"]
    if "initial_velocity" not in ic or len(ic["initial_velocity"]) != 3:
        raise ValidationError("initial_velocity must be length 3")

    ef = data["external_forces"]
    if "force_vector" not in ef or len(ef["force_vector"]) != 3:
        raise ValidationError("force_vector must be length 3")

    geom = data["geometry_definition"]
    if "flattening_order" not in geom:
        raise ValidationError("flattening_order is required")

    if "geometry_mask_shape" not in geom:
        raise ValidationError("geometry_mask_shape is required")
    if len(geom["geometry_mask_shape"]) != 3:
        raise ValidationError("geometry_mask_shape must have length 3")
