# src/step3/corrector.py

from .ops.gradient import gradient_p_n_plus_1
from .ops.scaling import get_dt_over_rho

def apply_velocity_correction(v_star, p_next, dx, dy, dz, dt, rho):
    """
    Projects the intermediate velocity field v* onto a divergence-free space.
    v^{n+1} = v^* - (dt/rho) * grad(p^{n+1})
    """
    # 1. Compute the pressure gradient
    # grad_p returns the spatial derivative of the pressure field
    grad_p = gradient_p_n_plus_1(p_next, dx, dy, dz)
    
    # 2. Scaling factor (dt/rho)
    scaling = get_dt_over_rho(dt, rho)
    
    # 3. Final velocity correction
    # We subtract the scaled gradient from the intermediate velocity
    return v_star - (scaling * grad_p)