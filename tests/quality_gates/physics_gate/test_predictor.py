# tests/quality_gates/physics_gate/test_predictor.py

import pytest

from src.common.field_schema import FI
from src.step3.predictor import compute_local_predictor_step
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def setup_predictor_block(dt=1.0, rho=1.0, mu=1.0, dx=1.0):
    """
    Standardizes a StencilBlock and its memory foundation for pure physics testing.
    
    Fixes:
    1. Removes loop over SimpleCellMock (cells lack _dx attribute).
    2. Uses object.__setattr__ to bypass any potential frozen attribute restrictions.
    3. Clears the underlying shared buffer to prevent "0.25-scaling" leaks from dummy.
    """
    # Create the dummy state and pick a central block (index 10 is usually 1,1,1)
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]

    # FORCE UNIT PHYSICS on the Block only (the operators read from here)
    params = {
        '_dx': float(dx),
        '_dy': float(dx),
        '_dz': float(dx),
        '_dt': float(dt),
        '_rho': float(rho),
        '_mu': float(mu),
        '_f_vals': (0.0, 0.0, 10.0)
    }
    
    for attr, val in params.items():
        object.__setattr__(block, attr, val)

    # Nuke the entire field buffer to ensure a clean slate
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

def test_predictor_teamwork_full_integration():
    """
    A 'Kitchen Sink' test where every operator contributes.
    Formula: v* = v_n + (dt/rho) * [ (mu * lap) - (rho * adv) + F - grad_p ]
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)
    
    # 1. Current Velocity (v_n) = 1.0
    block.center.set_field(FI.VX, 1.0)
    
    # 2. Laplacian (Diffusion) 
    # Neighbors (2.0, 1.0, 0.0) -> lap = (2 - 2*1 + 0) / 1^2 = 0.0
    block.i_plus.set_field(FI.VX, 2.0)
    block.i_minus.set_field(FI.VX, 0.0)
    
    # 3. Advection (v ⋅ ∇)v
    # u_c = 1.0, du/dx = (2-0)/(2*1.0) = 1.0
    # Result: 1.0 * 1.0 = 1.0
    
    # 4. Body Force (F) = 10.0 (Applied to Z-axis in setup)
    
    # 5. Pressure Gradient (grad_p)
    # ip=2, im=0 -> grad_p = (2-0)/(2*1.0) = 1.0
    block.i_plus.set_field(FI.P, 2.0)
    block.i_minus.set_field(FI.P, 0.0)
    
    compute_local_predictor_step(block)
    
    # --- VX_STAR Math ---
    # v_n = 1.0
    # term = [ (1.0 * 0.0) - (1.0 * 1.0) + 0.0 - 1.0 ] = -2.0
    # v* = 1.0 + (1/1) * (-2.0) = -1.0
    assert block.center.get_field(FI.VX_STAR) == pytest.approx(-1.0)

    # --- VZ_STAR Math ---
    # v_n = 0.0, lap=0, adv=0, grad_p=0
    # term = [ 10.0 ] (only Body Force)
    # v* = 0 + (1/1) * (10.0) = 10.0
    assert block.center.get_field(FI.VZ_STAR) == pytest.approx(10.0)