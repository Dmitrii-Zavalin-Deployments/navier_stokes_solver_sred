# src/step1/__init__.py

"""
Step 1: Domain & Field Initialization.
This module handles configuration parsing, grid generation, 
and memory allocation for the staggered velocity and pressure fields.
"""

from .orchestrate_step1 import orchestrate_step1_state

# Only expose the main orchestrator to external modules to maintain clean boundaries
__all__ = [
    "orchestrate_step1_state",
]