# tests/scientific/test_scientific_step1_orchestration.py

import pytest
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1
from src.solver_input import (
    SolverInput, GridInput, FluidProperties, 
    ExternalForces, BoundaryConditions, BoundaryConditionItem,
    InitialConditions, MaskInput, SimulationParameters
)
from src.solver_state import SolverState

def create_mock_input():
    """Helper to assemble a valid, scientific-grade SolverInput."""
    inp = SolverInput()
    
    # 1. Grid
    inp.grid.nx, inp.grid.ny, inp.grid.nz = 4, 4, 4
    inp.grid.x_min, inp.grid.x_max = 0.0, 1.0
    inp.grid.y_min, inp.grid.y_max = 0.0, 1.0
    inp.grid.z_min, inp.grid.z_max = 0.0, 1.0
    
    # 2. Fluid
    inp.fluid_properties.density = 1000.0
    inp.fluid_properties.viscosity = 0.001
    
    # 3. Forces & BCs
    inp.external_forces.force_vector = [0.0, -9.81, 0.0]
    
    bc_item = BoundaryConditionItem()
    bc_item.location = "x_min"
    bc_item.type = "inflow"
    bc_item.values = {"u": 1.0}
    inp.boundary_conditions.items = [bc_item]
    
    # 4. Initial Conditions
    inp.initial_conditions.pressure = 101325.0
    inp.initial_conditions.velocity = [1.0, 0.0, 0.0]
    
    # 5. Mask (64 cells for 4x4x4)
    inp.mask.data = [1] * 64 
    
    # 6. Sim Params
    inp.simulation_parameters.dt = 0.01
    inp.simulation_parameters.t_end = 1.0
    
    return inp

def test_scientific_orchestration_geometry_mapping():
    """Verify state geometry is a bit-perfect copy of input grid."""
    inp = create_mock_input()
    state = orchestrate_step1(inp)
    
    assert state.grid.nx == 4
    assert state.grid.x_max == 1.0
    assert state.fluid.rho == 1000.0

def test_scientific_field_priming_and_precision():
    """Verify fields are allocated and primed with ICs at float64."""
    inp = create_mock_input()
    state = orchestrate_step1(inp)
    
    # Check Pressure Priming
    np.testing.assert_allclose(state.fields.P, 101325.0)
    # Check Velocity Priming
    np.testing.assert_allclose(state.fields.U, 1.0)
    
    assert state.fields.P.dtype == np.float64
    assert state.fields.U.shape == (5, 4, 4)  # N+1 staggering check

def test_scientific_topology_alignment():
    """Verify mask reconstruction and boundary lookup integration."""
    inp = create_mock_input()
    state = orchestrate_step1(inp)
    
    assert state.masks.mask.shape == (4, 4, 4)
    assert state.masks.is_fluid.all()  # We filled with 1s
    assert "x_min" in state.boundary_lookup
    assert state.boundary_lookup["x_min"]["u"] == 1.0

def test_scientific_firewall_density():
    """Verify _final_audit catches non-physical fluid properties."""
    inp = create_mock_input()
    inp.fluid_properties.density = -1.0  # Physical Impossibility
    
    with pytest.raises(ValueError, match="Non-physical density"):
        orchestrate_step1(inp)

def test_scientific_firewall_finiteness():
    """Verify _final_audit catches NaN/Inf contamination."""
    inp = create_mock_input()
    inp.initial_conditions.velocity = [np.nan, 0.0, 0.0]
    
    with pytest.raises(ValueError, match="Non-finite values"):
        orchestrate_step1(inp)