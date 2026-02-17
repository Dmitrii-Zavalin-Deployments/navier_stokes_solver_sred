# tests/step2/test_step2_output_dummy_schema.py

import numpy as np
from tests.helpers.solver_step2_output_dummy import solver_step2_output_dummy
from tests.helpers.solver_step2_output_schema import solver_step2_output_schema


def test_step2_dummy_matches_schema():
    state = solver_step2_output_dummy()

    schema = solver_step2_output_schema

    # ------------------------------------------------------------
    # Helper for ndarray check
    # ------------------------------------------------------------
    def is_ndarray(x):
        return isinstance(x, np.ndarray)

    # ------------------------------------------------------------
    # Recursive validator
    # ------------------------------------------------------------
    def validate(obj, sch):
        if isinstance(sch, dict):
            assert isinstance(obj, dict) or hasattr(obj, "__dict__")
            container = obj if isinstance(obj, dict) else obj.__dict__
            for key, subschema in sch.items():
                assert key in container
                validate(container[key], subschema)
        elif sch == "ndarray":
            assert is_ndarray(obj)
        elif sch == "callable":
            assert callable(obj)
        elif isinstance(sch, type):
            assert isinstance(obj, sch)
        else:
            raise ValueError(f"Unknown schema entry: {sch}")

    validate(state, schema)
