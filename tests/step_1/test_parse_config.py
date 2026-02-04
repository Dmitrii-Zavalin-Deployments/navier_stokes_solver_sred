# tests/step_1/test_parse_config.py
from src.step1.parse_config import parse_config


def test_parse_config_basic():
    data = {
        "domain_definition": {"nx": 2},
        "fluid_properties": {"density": 1.0},
        "simulation_parameters": {"time_step": 0.1},
        "external_forces": {"gravity": [0, -9.8, 0]},
        "boundary_conditions": [],
        "geometry_definition": {"geometry_mask_flat": [0], "geometry_mask_shape": [1,1,1], "flattening_order": "i"},
    }

    cfg = parse_config(data)

    assert cfg.domain["nx"] == 2
    assert cfg.fluid["density"] == 1.0
    assert cfg.simulation["time_step"] == 0.1
    assert cfg.forces["gravity"] == [0, -9.8, 0]
