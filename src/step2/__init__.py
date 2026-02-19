# src/step2/__init__.py

"""
Step 2: Operators & PPE Setup.
This module transforms the geometric data from Step 1 into 
computational sparse operators and prepares the linear systems.
"""

from .orchestrate_step2 import orchestrate_step2_state

# Only expose the main orchestrator to external modules
__all__ = [
    "orchestrate_step2_state",
]