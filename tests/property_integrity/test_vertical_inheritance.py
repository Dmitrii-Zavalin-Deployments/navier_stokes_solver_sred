# tests/property_integrity/test_vertical_inheritance.py


# Core Logic
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2

# Factory Functions (The "Recipes")
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

# Mock Config Data (Rule 5: Explicit numerical settings)
MOCK_CONFIG = {
    "ppe_tolerance": 1e-6,
    "ppe_atol": 1e-10,
    "ppe_max_iter": 1000,
    "ppe_omega": 1.5
}

def assert_structural_parity(actual, expected, path=""):
    """
    Knowledge Gate Helper: Recursively verifies that keys and data types match.
    Ignores specific values to focus on Pipeline Integrity (Rule 5).
    """
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())
    
    assert actual_keys == expected_keys, (
        f"Structure Break at '{path}': Missing or extra keys.\n"
        f"Diff: {actual_keys.symmetric_difference(expected_keys)}"
    )
    
    for key in actual:
        actual_val = actual[key]
        expected_val = expected[key]
        current_path = f"{path}.{key}" if path else key
        
        assert type(actual_val) is type(expected_val), (
            f"Type Mismatch at '{current_path}': "
            f"Expected {type(expected_val).__name__}, got {type(actual_val).__name__}"
        )
        
        if isinstance(actual_val, dict):
            assert_structural_parity(actual_val, expected_val, current_path)
        
        elif isinstance(actual_val, list):
            assert len(actual_val) == len(expected_val), (
                f"List Length Drift at '{current_path}': "
                f"Expected {len(expected_val)}, got {len(actual_val)}"
            )
            if len(actual_val) > 0:
                assert type(actual_val[0]) is type(expected_val[0]), (
                    f"List Element Type Mismatch at '{current_path}'"
                )

class TestVerticalIntegrity:
    """
    Vertical Integrity Mandate (Rule 5):
    Verifies data survival and structural alignment across the pipeline.
    """

    def test_input_to_step1_pipeline(self):
        """Phase 1: Validates Input -> Step 1 Orchestration"""
        NX, NY, NZ = 4, 4, 4
        
        input_dummy = create_validated_input(nx=NX, ny=NY, nz=NZ)
        expected_dummy = make_step1_output_dummy(nx=NX, ny=NY, nz=NZ)
        
        config_obj = SolverConfig(**MOCK_CONFIG)
        context = SimulationContext(input_data=input_dummy, config=config_obj)
        
        actual_state = orchestrate_step1(context)
        
        print(f"\n" + "-"*30)
        print(f"AUDIT: Input -> Step 1 ({NX}x{NY}x{NZ})")
        assert_structural_parity(actual_state.to_dict(), expected_dummy.to_dict())
        print("✅ Step 1 Structural Parity Secured")

    def test_step1_to_step2_pipeline(self):
        """Phase 2: Validates Step 1 -> Step 2 Orchestration"""
        NX, NY, NZ = 4, 4, 4
        
        # We start with the established Dummy for Step 1
        step1_dummy = make_step1_output_dummy(nx=NX, ny=NY, nz=NZ)
        expected_dummy = make_step2_output_dummy(nx=NX, ny=NY, nz=NZ)
        
        # Step 2 consumes the state directly
        actual_state = orchestrate_step2(step1_dummy)
        
        print(f"\n" + "-"*30)
        print(f"AUDIT: Step 1 -> Step 2 ({NX}x{NY}x{NZ})")
        assert_structural_parity(actual_state.to_dict(), expected_dummy.to_dict())
        print("✅ Step 2 Structural Parity Secured")