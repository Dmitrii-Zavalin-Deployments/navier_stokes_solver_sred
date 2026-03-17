# tests/property_integrity/test_step1_vertical_integrity.py

import pytest
import numpy as np

# Core Orchestrators
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5
from src.main_solver import MainSolver

# Frozen Dummies from tests/helpers/
from tests.helpers.solver_input_schema_dummy import INPUT_DUMMY
from tests.helpers.solver_step1_output_dummy import STEP1_DUMMY
from tests.helpers.solver_step2_output_dummy import STEP2_DUMMY
from tests.helpers.solver_step3_output_dummy import STEP3_DUMMY
from tests.helpers.solver_step4_output_dummy import STEP4_DUMMY
from tests.helpers.solver_step5_output_dummy import STEP5_DUMMY
from tests.helpers.solver_output_schema_dummy import OUTPUT_SCHEMA_DUMMY

class TestVerticalInheritance:
    """
    Vertical Integrity Mandate (Rule 5):
    Verifies the linear sequence of the 5-Step Pipeline Architecture.
    """

    def test_input_to_step1_pipeline(self):
        """Verify Step 1 processing produces valid Step 2 input."""
        # 1. Take solver_input_schema_dummy
        # 2. Run orchestrate_step1
        # 3. Verify it matches solver_step1_output_dummy
        actual_step1 = orchestrate_step1(INPUT_DUMMY)
        
        foreach key in INPUT_DUMMY:
            assert INPUT_DUMMY(key) == STEP1_DUMMY(key)
    
    def test_step1_to_step2_pipeline(self):
        """Verify Step 1 processing produces valid Step 2 input."""
        # 1. Take solver_input_schema_dummy
        # 2. Run orchestrate_step1
        # 3. Verify it matches solver_step1_output_dummy
        actual_step1 = orchestrate_step1(INPUT_DUMMY)
        
        assert actual_step1.grid.nx == STEP1_DUMMY.grid.nx
        assert actual_step1.config.viscosity == STEP1_DUMMY.config.viscosity
        # Verify transition to Step 2
        actual_step2 = orchestrate_step2(actual_step1)
        assert len(actual_step2.stencil_matrix) == len(STEP2_DUMMY.stencil_matrix)

    def test_step2_to_step3_pipeline(self):
        """Verify Step 2 output feeds correctly into Step 3 physics."""
        actual_step3 = orchestrate_step3(STEP2_DUMMY)
        
        # Verify the physics fields were initialized in Step 3
        assert actual_step3.fields.data.shape == STEP3_DUMMY.fields.data.shape
        assert actual_step3.iteration == STEP3_DUMMY.iteration

    def test_step3_to_step4_pipeline(self):
        """Verify Step 3 physics state reaches Step 4 boundary logic."""
        actual_step4 = orchestrate_step4(STEP3_DUMMY)
        
        # Verify boundary flags are preserved
        assert actual_step4.grid.boundary_types == STEP4_DUMMY.grid.boundary_types

    def test_step4_to_step5_pipeline(self):
        """Verify Step 4 state reaches Step 5 archival logic."""
        actual_step5 = orchestrate_step5(STEP4_DUMMY)
        
        # Step 5 result is the finalized archive state
        assert actual_step5.is_archived is True

    def test_complete_solver_lifecycle(self):
        """
        The Full Runcycle Test:
        take solver_input_schema_dummy, run main_solver.py 
        and verify if we get solver_output_schema_dummy.
        """
        solver = MainSolver(INPUT_DUMMY)
        final_output = solver.run()
        
        # Rule 5: 100% match of inputs and outputs between steps
        assert final_output.status == "SUCCESS"
        assert np.isclose(final_output.final_energy, OUTPUT_SCHEMA_DUMMY.final_energy)
        assert final_output.total_steps == OUTPUT_SCHEMA_DUMMY.total_steps

    def test_property_survival_matrix(self):
        """Ensure critical fluid constants survive through Step 5."""
        s1 = orchestrate_step1(INPUT_DUMMY)
        s2 = orchestrate_step2(s1)
        s3 = orchestrate_step3(s2)
        s4 = orchestrate_step4(s3)
        s5 = orchestrate_step5(s4)
        
        # Check density at the very end of the pipe
        # Access via Rule 4: Hierarchy over Convenience
        assert s5.config.physics.density == INPUT_DUMMY.density