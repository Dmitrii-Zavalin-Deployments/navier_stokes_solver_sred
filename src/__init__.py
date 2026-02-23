# src/__init__.py

"""
Navier-Stokes Solver - SRED Edition
Unified API for the 5-Step Numerical Pipeline.
"""

from .main_solver import run_solver
from .solver_state import SolverState

# Define the public-facing API for the src package
__all__ = [
    "run_solver",
    "SolverState",
]