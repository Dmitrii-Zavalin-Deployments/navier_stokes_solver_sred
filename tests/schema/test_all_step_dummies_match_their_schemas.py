# tests/schema/test_all_step_dummies_match_their_schemas.py

import jsonschema

# Step 1
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step1_output_schema import step1_output_schema

# Step 2
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step2_output_schema import step2_output_schema

# Step 3
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step3_output_schema import step3_output_schema

# Step 4
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step4_output_schema import step4_output_schema

# Step 5
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy
from tests.helpers.solver_step5_output_schema import step5_output_schema


def test_step1_dummy_matches_schema():
    dummy = make_step1_output_dummy()
    jsonschema.validate(instance=dummy, schema=step1_output_schema)


def test_step2_dummy_matches_schema():
    dummy = make_step2_output_dummy()
    jsonschema.validate(instance=dummy, schema=step2_output_schema)


def test_step3_dummy_matches_schema():
    dummy = make_step3_output_dummy()
    jsonschema.validate(instance=dummy, schema=step3_output_schema)


def test_step4_dummy_matches_schema():
    dummy = make_step4_output_dummy()
    jsonschema.validate(instance=dummy, schema=step4_output_schema)


def test_step5_dummy_matches_schema():
    dummy = make_step5_output_dummy()
    jsonschema.validate(instance=dummy, schema=step5_output_schema)
