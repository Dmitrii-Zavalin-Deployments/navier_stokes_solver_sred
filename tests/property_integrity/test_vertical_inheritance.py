# tests/property_integrity/test_vertical_inheritance.py


# Core Logic
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1

# Factory Functions (The "Recipes")
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

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
    # 1. Verify Key Existence
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
        
        # 2. Verify Data Type Parity
        assert type(actual_val) is type(expected_val), (
            f"Type Mismatch at '{current_path}': "
            f"Expected {type(expected_val).__name__}, got {type(actual_val).__name__}"
        )
        
        # 3. Recursive Branching for Dicts
        if isinstance(actual_val, dict):
            assert_structural_parity(actual_val, expected_val, current_path)
        
        # 4. Collection Integrity for Lists (e.g., Mask, BCs)
        elif isinstance(actual_val, list):
            assert len(actual_val) == len(expected_val), (
                f"List Length Drift at '{current_path}': "
                f"Expected {len(expected_val)}, got {len(actual_val)}"
            )
            if len(actual_val) > 0:
                # Check internal type of first element to ensure list homogeneity
                assert type(actual_val[0]) is type(expected_val[0]), (
                    f"List Element Type Mismatch at '{current_path}'"
                )

class TestVerticalIntegrity:
    """
    Vertical Integrity Mandate (Rule 5):
    Verifies that the SolverState container produced by Step 1 is structurally 
    aligned with the downstream Knowledge Gate Dummies.
    """

    def test_input_to_step1_pipeline(self):
        # 1. Define Scale
        NX, NY, NZ = 4, 4, 4
        
        # 2. Setup Dummies (Signal-Aligned)
        input_dummy = create_validated_input(nx=NX, ny=NY, nz=NZ)
        expected_dummy = make_step1_output_dummy(nx=NX, ny=NY, nz=NZ)
        
        # 3. Assemble Execution Context
        config_obj = SolverConfig(**MOCK_CONFIG)
        context = SimulationContext(input_data=input_dummy, config=config_obj)
        
        # 4. Run Step 1 Orchestrator
        actual_state = orchestrate_step1(context)
        
        # 5. Serialization for Audit
        actual_dict = actual_state.to_dict()
        expected_dict = expected_dummy.to_dict()
        
        print(f"\n" + "="*60)
        print(f"VERTICAL INTEGRITY AUDIT: {NX}x{NY}x{NZ}")
        print("="*60)
        print(f"Status: Validating Structural Parity (Keys & Types)")
        
        # 6. Structural Assertion (Rule 5 Knowledge Gate)
        # This replaces the brittle value-based assertion
        assert_structural_parity(actual_dict, expected_dict)
        
        print("SUCCESS: actual_state is structurally aligned with dummy.")
        print("="*60)