# tests/step3/test_bc_handler_hooks_called.py

import numpy as np
from src.step3.orchestrate_step3 import orchestrate_step3_state
from src.solver_state import SolverState


def make_state(nx=4, ny=4, nz=4):
    """Construct a minimal valid SolverState for Step 3 tests."""
    fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz)),
    }

    mask = np.ones((nx, ny, nz), dtype=int)
    is_fluid = mask == 1
    is_boundary_cell = np.zeros_like(mask, dtype=bool)

    return SolverState(
        config={"external_forces": {}},
        grid={"nx": nx, "ny": ny, "nz": nz},
        fields=fields,
        mask=mask,
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell,
        constants={"rho": 1.0, "mu": 1.0, "dt": 0.1, "dx": 1.0, "dy": 1.0, "dz": 1.0},
        boundary_conditions=None,
        operators={
            "lap_u": lambda u: np.zeros_like(u),
            "lap_v": lambda v: np.zeros_like(v),
            "lap_w": lambda w: np.zeros_like(w),
            "grad_x": lambda p: np.zeros((nx + 1, ny, nz)),
            "grad_y": lambda p: np.zeros((nx, ny + 1, nz)),
            "grad_z": lambda p: np.zeros((nx, ny, nz + 1)),
            "divergence": lambda U, V, W: np.zeros((nx, ny, nz)),
        },
        ppe={
            "solver_type": "SOR",
            "tolerance": 1e-6,
            "max_iterations": 10,
            "ppe_is_singular": True,
            "rhs_builder": "rho/dt * div(U*)",
        },
        health={},
        history={},
    )


def test_bc_handler_hooks_called():
    """
    Step‑3 must call the boundary_conditions handler twice:
        • once in apply_boundary_conditions_pre
        • once in apply_boundary_conditions_post
    """
    state = make_state()

    calls = {"count": 0}

    def bc_handler(state, fields):
        calls["count"] += 1
        return fields

    state.boundary_conditions = bc_handler

    # Run Step‑3
    new_state = orchestrate_step3_state(
        state=state,
        current_time=0.0,
        step_index=0,
    )

    # Expect exactly two calls: pre + post
    assert calls["count"] == 2
