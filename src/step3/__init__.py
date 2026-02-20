# src/step3/__init__.py

"""
Step 3: Time Integration & Projection.
This module executes the fractional step method: velocity prediction,
pressure solution (PPE), and divergence-free correction.
"""

from .orchestrate_step3 import orchestrate_step3_state

# Only expose the main orchestrator to keep the interface consistent with Step 2
__all__ = [
    "orchestrate_step3_state",
]