# src/step3/ops/scaling.py

def get_dt_over_rho(dt, rho):
    """
    Returns the scaling factor used in the Predictor Step (Section 5.1).
    Formula: dt / rho
    """
    return dt / rho

def get_rho_over_dt(dt, rho):
    """
    Returns the scaling factor used in the Pressure Poisson Equation (Section 5.2).
    Formula: rho / dt
    """
    return rho / dt