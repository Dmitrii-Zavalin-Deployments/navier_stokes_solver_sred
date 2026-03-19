# tests/quality_gates/physics_gate/test_predictor.py

import pytest

from src.common.field_schema import FI

# Corrected Imports based on your grep/head results
from src.step3.ops.advection import compute_local_advection
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.laplacian import compute_local_laplacian
from src.step3.predictor import compute_local_predictor_step
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def setup_predictor_block(dt=1.0, rho=1.0, mu=1.0, dx=1.0):
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]

    params = {
        '_dx': float(dx), '_dy': float(dx), '_dz': float(dx),
        '_dt': float(dt), '_rho': float(rho), '_mu': float(mu),
        '_f_vals': (0.0, 0.0, 10.0)
    }
    for attr, val in params.items():
        object.__setattr__(block, attr, val)

    # Clean out the shared buffer to ensure no leaked values interfere
    block.center.fields_buffer.fill(0.0) 
    return block

def test_predictor_teamwork_diffusion_only():
    """
    Verifies that v* = v_n + (dt/rho) * (mu * lap(v_n))
    Expectation: With mu=1, dt=1, rho=1, and lap=2.0 -> v* = 0 + 2.0 = 2.0
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)
    
    # Create a local curvature in VX: (ip: 1.0, c: 0.0, im: 1.0)
    # lap = (1 - 2*0 + 1) / 1^2 = 2.0
    block.i_plus.set_field(FI.VX, 1.0)
    block.i_minus.set_field(FI.VX, 1.0)
    
    compute_local_predictor_step(block)
    
    # Calculation: v* = 0 + (1/1) * (1.0 * 2.0) = 2.0
    assert block.center.get_field(FI.VX_STAR) == pytest.approx(2.0)

def test_predictor_teamwork_diagnostics():
    """
    Diagnostic suite to find why VX_STAR is -5.0 instead of -1.0.
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)
    
    # 1. Setup Linear Fields
    # VX: ip=2.0, c=1.0, im=0.0  -> grad=1.0, lap=0.0
    block.center.set_field(FI.VX, 1.0)
    block.i_plus.set_field(FI.VX, 2.0)
    block.i_minus.set_field(FI.VX, 0.0)
    
    # Pressure: ip=2.0, im=0.0 -> grad_p = 1.0
    block.i_plus.set_field(FI.P, 2.0)
    block.i_minus.set_field(FI.P, 0.0)

    # --- THE INVESTIGATION ---
    
    # A. Check Laplacian (Expected: 0.0)
    lap = compute_local_laplacian(block, FI.VX)
    assert lap == pytest.approx(0.0), f"Laplacian Leak! Expected 0.0, got {lap}"

    # B. Check Advection (Expected: 1.0)
    # If this is 4.0, it's using dx=0.25
    adv = compute_local_advection(block, FI.VX)
    assert adv == pytest.approx(1.0), f"Advection Leak! Expected 1.0, got {adv}"

    # C. Check Gradient (Expected: 1.0)
    # Using the correct function name: compute_local_gradient_p
    grad_tuple = compute_local_gradient_p(block)
    grad_p_x = grad_tuple[0] 
    assert grad_p_x == pytest.approx(1.0), f"Gradient Leak! Expected 1.0, got {grad_p_x}"

    # --- FINAL MATH CHECK ---
    compute_local_predictor_step(block)
    
    # Formula: v* = v_n + (dt/rho) * [ (mu * lap) - (rho * adv) + F - grad_p ]
    # Expected: 1.0 + 1.0 * [ 0 - 1.0 + 0 - 1.0 ] = -1.0
    obtained = block.center.get_field(FI.VX_STAR)
    assert obtained == pytest.approx(-1.0), f"Math Error! Expected -1.0, got {obtained}"