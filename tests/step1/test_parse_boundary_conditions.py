# tests/step1/test_parse_boundary_conditions.py

import pytest

from src.step1.parse_boundary_conditions import parse_boundary_conditions
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

# Use the dummy helper to get the domain dictionary
# This satisfies the requirement to use the dummy as a base
DUMMY_DATA = solver_input_schema_dummy()
GRID = DUMMY_DATA["domain"]


def test_invalid_location_rejected():
    bc = [{"location": "diagonal", "type": "no-slip"}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


def test_invalid_type_rejected():
    bc = [{"location": "x_min", "type": "warp-drive"}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


def test_duplicate_location_rejected():
    bc = [
        {"location": "x_min", "type": "no-slip"},
        {"location": "x_min", "type": "pressure", "values": {"p": 1.0}},
    ]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


def test_inflow_requires_velocity():
    bc = [{"location": "x_min", "type": "inflow", "values": {}}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


def test_pressure_bc_requires_p():
    bc = [{"location": "x_max", "type": "pressure", "values": {}}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


def test_no_slip_disallows_pressure():
    bc = [{"location": "y_min", "type": "no-slip", "values": {"p": 5.0}}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)