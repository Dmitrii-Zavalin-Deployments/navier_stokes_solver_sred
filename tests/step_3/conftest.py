# tests/step3/conftest.py

import pytest
import numpy as np

@pytest.fixture
def minimal_state():
    """
    Creates a minimal 3×3×3 SimulationState-like dict for Step 3 tests.
    Uses dummy operators that return zeros unless overridden.
    """

    nx = ny = nz = 3

    P = np.zeros((nx, ny, nz))
    U = np.zeros((nx+1, ny, nz))
    V = np.zeros((nx, ny+1, nz))
    W = np.zeros((nx, ny, nz+1))

    mask = np.ones((nx, ny, nz), dtype=int)
    is_fluid = mask == 1
    is_boundary = np.zeros_like(mask, bool)

    class DummyOps:
        @staticmethod
        def divergence(U, V, W, state):
            pattern = state.get("_divergence_pattern", None)
            if pattern is not None:
                return pattern.copy()
            return np.zeros_like(state["P"])

        @staticmethod
        def laplacian_u(U, state): return np.zeros_like(U)
        @staticmethod
        def laplacian_v(V, state): return np.zeros_like(V)
        @staticmethod
        def laplacian_w(W, state): return np.zeros_like(W)

        @staticmethod
        def advection_u(U, V, W, state): return np.zeros_like(U)
        @staticmethod
        def advection_v(U, V, W, state): return np.zeros_like(V)
        @staticmethod
        def advection_w(U, V, W, state): return np.zeros_like(W)

        @staticmethod
        def gradient_p_x(P, state): return np.zeros((nx+1, ny, nz))
        @staticmethod
        def gradient_p_y(P, state): return np.zeros((nx, ny+1, nz))
        @staticmethod
        def gradient_p_z(P, state): return np.zeros((nx, ny, nz+1))

    ops = {
        "divergence": DummyOps.divergence,
        "laplacian_u": DummyOps.laplacian_u,
        "laplacian_v": DummyOps.laplacian_v,
        "laplacian_w": DummyOps.laplacian_w,
        "advection_u": DummyOps.advection_u,
        "advection_v": DummyOps.advection_v,
        "advection_w": DummyOps.advection_w,
        "gradient_p_x": DummyOps.gradient_p_x,
        "gradient_p_y": DummyOps.gradient_p_y,
        "gradient_p_z": DummyOps.gradient_p_z,
    }

    state = {
        "Config": {"external_forces": {}},
        "Mask": mask,
        "is_fluid": is_fluid,
        "is_boundary_cell": is_boundary,
        "P": P,
        "U": U,
        "V": V,
        "W": W,
        "BCs": [],
        "Constants": {
            "rho": 1.0,
            "mu": 0.1,
            "dt": 0.01,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
        },
        "Operators": ops,
        "PPE": {
            "solver": None,
            "tolerance": 1e-6,
            "max_iterations": 100,
            "ppe_is_singular": False,
        },
        "Health": {},
        "History": {},
    }

    return state
