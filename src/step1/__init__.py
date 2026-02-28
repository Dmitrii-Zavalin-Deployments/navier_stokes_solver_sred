# src/step1/__init__.py

"""
Step 1: Domain & Field Initialization (The Foundation).

This module handles the transformation of raw JSON configuration into a 
fully-allocated SolverState. It encompasses:
1. Config Parsing & Validation
2. Staggered Grid Generation
3. Physical Property Mapping
4. Initial Condition Deployment

Constitutional Role: The Gatekeeper.
Registry Status: Ready for Step 2 Handoff.
Last Updated: 2026-02-23
"""

from .orchestrate_step1 import orchestrate_step1

# Encapsulation: Only expose the orchestrator to maintain the "Pure Path" 
# mandated by the Project Constitution.
__all__ = [
    "orchestrate_step1",
]