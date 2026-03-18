# tests/ quality_gates/physics_gate/test_advection.py

import pytest
import numpy as np
from src.common.field_schema import FI
from src.step3.ops.advection import compute_local_advection, compute_local_advection_vector

# --- Scenario 1: Null Field (The Zero-Gate) ---
def test_advection_zero_velocity(mock_stencil_block):
    """Verify (0 ⋅ ∇)f = 0 even if the field f has a steep gradient."""
    block = mock_stencil_block
    # Set velocity to 0
    for cell in block.all_cells():
        cell.set_field(FI.VX, 0.0)
        cell.set_field(FI.VY, 0.0)
        cell.set_field(FI.VZ, 0.0)
        # Set scalar field f to a steep gradient (f = x + y + z)
        cell.set_field(FI.P, cell.i + cell.j + cell.k)

    result = compute_local_advection(block, FI.P)
    assert result == 0.0, f"Advection should be zero with no velocity. Got: {result}"

# --- Scenario 2: Uniform Velocity & Linear Gradient ---
def test_advection_linear_fidelity(mock_stencil_block):
    """
    Verify (v ⋅ ∇)f for uniform velocity v=(1,1,1) and f=x+y+z.
    Analytical result: 1*(df/dx) + 1*(df/dy) + 1*(df/dz) = 1*1 + 1*1 + 1*1 = 3.0
    """
    block = mock_stencil_block
    block.dx = block.dy = block.dz = 1.0 # Simplify grid
    
    for cell in block.all_cells():
        # v = (1, 1, 1)
        cell.set_field(FI.VX, 1.0)
        cell.set_field(FI.VY, 1.0)
        cell.set_field(FI.VZ, 1.0)
        # f = x + y + z (using cell indices as coordinates)
        cell.set_field(FI.P, float(cell.i + cell.j + cell.k))

    result = compute_local_advection(block, FI.P)
    # df/dx = (f_ip - f_im)/2 = ((i+1) - (i-1))/2 = 1.0
    # u_c = (1 + 1)/2 = 1.0
    assert abs(result - 3.0) < 1e-12, f"Linear advection fidelity failed. Expected 3.0, got {result}"

# --- Scenario 3: Staggered Velocity Alignment (Vector Test) ---
def test_advection_vector_component_isolation(mock_stencil_block):
    """
    Ensure the vector version correctly isolates components.
    If v = (2, 0, 0) and f_vector = (x, 0, 0), then advection is (2, 0, 0).
    """
    block = mock_stencil_block
    block.dx = 1.0
    
    for cell in block.all_cells():
        cell.set_field(FI.VX, 2.0) # u = 2
        cell.set_field(FI.VY, 0.0)
        cell.set_field(FI.VZ, 0.0)
        
        cell.set_field(FI.VX_STAR, float(cell.i)) # f_x = x
        cell.set_field(FI.VY_STAR, 0.0)            # f_y = 0
        cell.set_field(FI.VZ_STAR, 0.0)            # f_z = 0

    # We use FI.VX_STAR as the target field to avoid overwriting primary VX
    adv_x, adv_y, adv_z = (
        compute_local_advection(block, FI.VX_STAR),
        compute_local_advection(block, FI.VY_STAR),
        compute_local_advection(block, FI.VZ_STAR)
    )
    
    assert adv_x == 2.0, f"X-Advection failed. Got {adv_x}"
    assert adv_y == 0.0, f"Y-Advection should be zero. Got {adv_y}"
    assert adv_z == 0.0, f"Z-Advection should be zero. Got {adv_z}"

# --- Scenario 4: Edge Case - High Gradient Reversal ---
def test_advection_opposite_directions(mock_stencil_block):
    """
    Test a case where velocity and gradient are opposing.
    v = (1, 0, 0), f = -x -> (1 * -1) = -1.0
    """
    block = mock_stencil_block
    block.dx = 1.0
    
    for cell in block.all_cells():
        cell.set_field(FI.VX, 1.0)
        cell.set_field(FI.P, -float(cell.i))

    result = compute_local_advection(block, FI.P)
    assert result == -1.0, f"Reversed advection failed. Expected -1.0, got {result}"