# tests/contracts/test_solver_state_integrity.py

import pytest
import numpy as np
from src.solver_state import SolverState, FieldData, Diagnostics, SimulationHistory

class TestSolverStateIntegrity:
    """
    AUDIT GATE: Verifies that SolverState and its sub-containers satisfy 
    the project's structural contract and achieve 100% logic coverage.
    """

    def test_solver_state_facades_and_shortcuts(self):
        """
        Hits lines: 72, 703, 707, 711, 715, 719, 725, 730, 745, 749, 753, 757.
        Validates the 'Shortcut' properties used by the physics engine.
        """
        state = SolverState()
        
        # Setup underlying data required for facades
        state.config.fluid_properties = {"density": 1000.0, "viscosity": 0.001}
        state.config.simulation_parameters = {
            "time_step": 0.1, 
            "total_time": 1.0, 
            "output_interval": 1
        }
        state.grid.x_min, state.grid.x_max, state.grid.nx = 0.0, 1.0, 10
        state.grid.y_min, state.grid.y_max, state.grid.ny = 0.0, 1.0, 10
        state.grid.z_min, state.grid.z_max, state.grid.nz = 0.0, 1.0, 10
        
        # 1. Physics Facades
        assert state.rho == 1000.0
        assert state.mu == 0.001
        
        # 2. Numerical Facades
        assert state.dt == 0.1
        assert state.inv_dx == 10.0
        assert state.inv_dy == 10.0
        assert state.inv_dz == 10.0
        
        # 3. Field Facades (Extended Fields)
        arr = np.zeros((3, 3, 3))
        state.fields.U_ext = arr
        assert np.array_equal(state.U_ext, arr)

    def test_field_data_dictionary_interface(self):
        """Hits lines 304, 311-315: __getitem__ and __setitem__ logic."""
        fields = FieldData()
        arr = np.array([1.0, 2.0, 3.0])
        
        # Test Case-Insensitive set/get
        fields["U_STAR"] = arr
        fields["p"] = arr
        
        assert np.array_equal(fields["u_star"], arr)
        assert np.array_equal(fields["P"], arr)
        
        # Test KeyError path
        with pytest.raises(KeyError, match="is not recognized"):
            _ = fields["GHOST_PARTICLE_FIELD"]

    def test_diagnostics_safe_get_and_mapping(self):
        """Hits lines 647, 656: get() and legacy __getitem__ mapping."""
        diag = Diagnostics()
        diag.memory_footprint_gb = 5.0
        diag.bc_verification_passed = True
        
        # Test Legacy Key Mapping
        assert diag["memory_footprint"] == 5.0
        assert diag["bc_verification"] is True
        
        # Test .get() with and without defaults
        assert diag.get("memory_footprint") == 5.0
        assert diag.get("missing_key", "default_val") == "default_val"

    def test_simulation_health_mapping(self):
        """Hits lines 542-551: SimulationHealth.get() mapping."""
        state = SolverState()
        state.health.max_u = 10.5
        state.health.is_stable = True
        
        assert state.health.get("max_velocity_magnitude") == 10.5
        assert state.health.get("is_stable") is True
        assert state.health.get("non_existent", 0.0) == 0.0

    def test_simulation_history_dictionary_access(self):
        """Hits lines 609-610: History __getitem__."""
        history = SimulationHistory()
        history.times.append(0.1)
        
        assert history["times"] == [0.1]
        
        with pytest.raises(KeyError, match="not found"):
            _ = history["future_predictions"]

    def test_legacy_support_and_roundtrip(self):
        """Hits lines 764, 767, 780-795: to_legacy_dict and to_json_safe."""
        state = SolverState()
        state.time = 0.5
        
        # Test Legacy Export
        legacy = state.to_legacy_dict()
        assert legacy["ready_for_time_loop"] is False
        assert "step4_diagnostics" in legacy
        
        # Test JSON Safe Export (The Contract Bridge)
        safe_dict = state.to_json_safe()
        assert safe_dict["time"] == 0.5
        assert isinstance(safe_dict["config"], dict)