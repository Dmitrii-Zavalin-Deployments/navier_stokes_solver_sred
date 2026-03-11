# src/__init__.py

"""
Navier-Stokes Solver Core.

This package manages the high-fidelity CFD simulation orchestration.
Compliance:
- Rule 8 (API Minimalism): Exposes only essential entry points for execution.
- Rule 4 (SSoT): Internal modules are strictly encapsulated to prevent 
  unauthorized access to Foundation buffers or logic-gatekeepers.
"""

# Aligned with main_solver.py: run_solver is the authorized execution entry point.
from src.main_solver import run_solver

__all__ = ["run_solver"]

# Security/Integrity Note:
# Direct imports of internal steps (e.g., from src.step1 import ...) 
# are discouraged unless explicitly required for atomic verification 
# within the tests/scientific/ suite.