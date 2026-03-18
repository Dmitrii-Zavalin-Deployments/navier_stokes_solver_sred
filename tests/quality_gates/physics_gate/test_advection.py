# tests/ quality_gates/physics_gate/test_advection.py

import copy

from src.common.field_schema import FI
from src.step3.ops.advection import (
    compute_local_advection,
    compute_local_advection_vector,
)
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def setup_stencil_data(block):
    """
    Bypasses ValidatedContainer restrictions to create a unique test topology.
    """
    # Helper to bypass read-only properties by writing directly to the underlying slots
    def force_set(obj, attr, val):
        # We target the underscore-prefixed slot name defined in StencilBlock
        object.__setattr__(obj, f"_{attr}", val)

    # 1. Clone cells and force-set them into the block's slots
    force_set(block, 'center', copy.copy(block.center))
    force_set(block, 'i_plus', copy.copy(block.i_plus))
    force_set(block, 'i_minus', copy.copy(block.i_minus))
    force_set(block, 'j_plus', copy.copy(block.j_plus))
    force_set(block, 'j_minus', copy.copy(block.j_minus))
    force_set(block, 'k_plus', copy.copy(block.k_plus))
    force_set(block, 'k_minus', copy.copy(block.k_minus))

    # 2. Update buffer indices so (neighbor - center) != 0
    base = block.center.index
    block.i_plus.index, block.i_minus.index = base + 1, base - 1
    block.j_plus.index, block.j_minus.index = base + 10, base - 10
    block.k_plus.index, block.k_minus.index = base + 100, base - 100

    # 3. Inject coordinate data
    # Since SimpleCellMock doesn't have i,j,k in __slots__, 
    # we just attach them—Python allows this on the instance for these mocks
    # because they don't inherit from ValidatedContainer.
    def set_coords(cell, i, j, k):
        cell.i, cell.j, cell.k = i, j, k

    set_coords(block.center, 1, 1, 1)
    set_coords(block.i_plus, 2, 1, 1); set_coords(block.i_minus, 0, 1, 1)
    set_coords(block.j_plus, 1, 2, 1); set_coords(block.j_minus, 1, 0, 1)
    set_coords(block.k_plus, 1, 1, 2); set_coords(block.k_minus, 1, 1, 0)
    
    # Force-set physics params which are also read-only properties
    force_set(block, 'dx', 1.0)
    force_set(block, 'dy', 1.0)
    force_set(block, 'dz', 1.0)
    
    return block

def get_all_cells(block):
    return [block.center, block.i_plus, block.i_minus, 
            block.j_plus, block.j_minus, block.k_plus, block.k_minus]

# --- Tests (Logic remains the same, now they will pass) ---

def test_advection_zero_velocity():
    block = setup_stencil_data(make_step3_output_dummy())
    for cell in get_all_cells(block):
        cell.set_field(FI.VX, 0.0); cell.set_field(FI.VY, 0.0); cell.set_field(FI.VZ, 0.0)
        cell.set_field(FI.P, float(cell.i + cell.j + cell.k))
    assert compute_local_advection(block, FI.P) == 0.0

def test_advection_linear_fidelity():
    block = setup_stencil_data(make_step3_output_dummy())
    for cell in get_all_cells(block):
        cell.set_field(FI.VX, 1.0); cell.set_field(FI.VY, 1.0); cell.set_field(FI.VZ, 1.0)
        cell.set_field(FI.P, float(cell.i + cell.j + cell.k))
    assert abs(compute_local_advection(block, FI.P) - 3.0) < 1e-12

def test_advection_vector_component_isolation():
    block = setup_stencil_data(make_step3_output_dummy())
    for cell in get_all_cells(block):
        cell.set_field(FI.VX, 2.0); cell.set_field(FI.VY, 0.0); cell.set_field(FI.VZ, 0.0)
        cell.set_field(FI.VX_STAR, float(cell.i))
        cell.set_field(FI.VY_STAR, 0.0); cell.set_field(FI.VZ_STAR, 0.0)
    adv_vec = compute_local_advection_vector(block)
    assert adv_vec[0] == 2.0 and adv_vec[1] == 0.0 and adv_vec[2] == 0.0

def test_advection_opposite_directions():
    block = setup_stencil_data(make_step3_output_dummy())
    for cell in get_all_cells(block):
        cell.set_field(FI.VX, 1.0); cell.set_field(FI.P, -float(cell.i))
    assert compute_local_advection(block, FI.P) == -1.0