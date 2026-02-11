# tests/step1/test_parse_config_math.py

import pytest
from src.step1.parse_config import parse_config
from src.step1.types import Config


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def make_minimal_valid_input():
    return {
        "domain_definition": {},
        "fluid_properties": {},
        "simulation_parameters": {},
        "geometry_definition": {
            "geometry_mask_flat": [0],
            "geometry_mask_shape": [1, 1, 1],
            "mask_encoding": "none",
            "flattening_order": "C",
        },
        "boundary_conditions": [],
        "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
    }


# ---------------------------------------------------------
# Required top-level keys
# ---------------------------------------------------------

def test_missing_required_top_level_keys_raises():
    base = make_minimal_valid_input()
    for key in ["domain_definition", "fluid_properties",
                "simulation_parameters", "geometry_definition"]:
        bad = dict(base)
        del bad[key]
        with pytest.raises(KeyError):
            parse_config(bad)


# ---------------------------------------------------------
# Sections must be dictionaries
# ---------------------------------------------------------

def test_sections_must_be_dicts():
    base = make_minimal_valid_input()

    for key in ["domain_definition", "fluid_properties",
                "simulation_parameters", "geometry_definition"]:
        bad = dict(base)
        bad[key] = "not_a_dict"
        with pytest.raises(TypeError):
            parse_config(bad)


# ---------------------------------------------------------
# geometry_definition structure
# ---------------------------------------------------------

def test_geometry_definition_missing_required_keys():
    base = make_minimal_valid_input()
    for key in ["geometry_mask_flat", "geometry_mask_shape",
                "mask_encoding", "flattening_order"]:
        bad_geom = dict(base["geometry_definition"])
        del bad_geom[key]

        bad = dict(base)
        bad["geometry_definition"] = bad_geom

        with pytest.raises(KeyError):
            parse_config(bad)


# ---------------------------------------------------------
# boundary_conditions must be a list
# ---------------------------------------------------------

def test_boundary_conditions_must_be_list():
    base = make_minimal_valid_input()
    base["boundary_conditions"] = "not_a_list"
    with pytest.raises(TypeError):
        parse_config(base)


# ---------------------------------------------------------
# external_forces validation
# ---------------------------------------------------------

def test_external_forces_must_have_force_vector_or_gravity():
    base = make_minimal_valid_input()
    base["external_forces"] = {}  # neither force_vector nor gravity
    with pytest.raises(KeyError):
        parse_config(base)


def test_force_vector_must_be_length_3():
    base = make_minimal_valid_input()
    base["external_forces"]["force_vector"] = [1, 2]  # wrong length
    with pytest.raises(ValueError):
        parse_config(base)


def test_force_vector_entries_must_be_finite():
    bad_values = [float("inf"), float("nan"), "x"]

    for bad in bad_values:
        base = make_minimal_valid_input()
        base["external_forces"]["force_vector"] = [0.0, 0.0, bad]
        with pytest.raises(ValueError):
            parse_config(base)


def test_gravity_fallback_produces_force_vector():
    base = make_minimal_valid_input()
    del base["external_forces"]["force_vector"]
    base["external_forces"]["gravity"] = [1.0, 2.0, 3.0]

    cfg = parse_config(base)
    assert cfg.forces["force_vector"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------
# Valid config
# ---------------------------------------------------------

def test_valid_config_parses_successfully():
    base = make_minimal_valid_input()
    cfg = parse_config(base)

    assert isinstance(cfg, Config)
    assert cfg.geometry_definition["flattening_order"] == "C"
    assert cfg.forces["force_vector"] == [0.0, 0.0, 0.0]
