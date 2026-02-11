# tests/helpers/step4_schema_dummy_state.py

import numpy as np


class Step4SchemaDummyState(dict):
    """
    Fully schema‑compliant Step‑4 output dummy.
    Matches the Step 4 Output Schema exactly.
    """

    def __init__(self, nx, ny, nz):
        super().__init__()

        # ---------------------------------------------------------
        # Extended fields with ghost layers
        # ---------------------------------------------------------
        # Minimal ghost layer: +2 in each dimension
        self["p_ext"] = np.zeros((nx + 2, ny + 2, nz + 2))
        self["u_ext"] = np.zeros((nx + 3, ny + 2, nz + 2))
        self["v_ext"] = np.zeros((nx + 2, ny + 3, nz + 2))
        self["w_ext"] = np.zeros((nx + 2, ny + 2, nz + 3))

        # ---------------------------------------------------------
        # Domain metadata
        # ---------------------------------------------------------
        self["domain"] = {
            "coordinates": {
                "x_centers": list(np.linspace(0.5, nx - 0.5, nx)),
                "y_centers": list(np.linspace(0.5, ny - 0.5, ny)),
                "z_centers": list(np.linspace(0.5, nz - 0.5, nz)),
                "x_faces_u": list(np.linspace(0, nx, nx + 1)),
                "y_faces_v": list(np.linspace(0, ny, ny + 1)),
                "z_faces_w": list(np.linspace(0, nz, nz + 1)),
            },

            "ghost_layers": {
                "p_ext": [1, 1, 1, 1, 1, 1],
                "u_ext": [1, 1, 1, 1, 1, 1],
                "v_ext": [1, 1, 1, 1, 1, 1],
                "w_ext": [1, 1, 1, 1, 1, 1],
            },

            "index_ranges": {
                "interior": "1:-1,1:-1,1:-1",
                "ghost_x_lo": "0,:,:",
                "ghost_x_hi": "-1,:,:",
                "ghost_y_lo": ":,0,:",
                "ghost_y_hi": ":,-1,:",
                "ghost_z_lo": "[:,:,0]",
                "ghost_z_hi": "[:,:,-1]",
            },

            "stencil_maps": {
                "xp": [],
                "xm": [],
                "yp": [],
                "ym": [],
                "zp": [],
                "zm": [],
            },

            "interpolation_maps": {
                "interp_u_to_v": {},
                "interp_u_to_w": {},
                "interp_v_to_u": {},
                "interp_v_to_w": {},
                "interp_w_to_u": {},
                "interp_w_to_v": {},
            },
        }

        # ---------------------------------------------------------
        # RHS source terms
        # ---------------------------------------------------------
        self["rhs_source"] = {
            "fx_u": np.zeros((nx + 3, ny + 2, nz + 2)),
            "fy_v": np.zeros((nx + 2, ny + 3, nz + 2)),
            "fz_w": np.zeros((nx + 2, ny + 2, nz + 3)),
        }

        # ---------------------------------------------------------
        # Boundary-condition application metadata
        # ---------------------------------------------------------
        self["bc_applied"] = {
            "initial_velocity_enforced": True,
            "pressure_initial_applied": True,
            "velocity_initial_applied": True,
            "ghost_cells_filled": True,
            "boundary_cells_checked": nx * ny * nz,

            "version": "1.0",
            "timestamp_applied": "2025-01-01T00:00:00Z",

            "boundary_conditions_status": {
                "x_min": "applied",
                "x_max": "applied",
                "y_min": "applied",
                "y_max": "applied",
                "z_min": "applied",
                "z_max": "applied",
            },
        }

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------
        self["diagnostics"] = {
            "total_fluid_cells": nx * ny * nz,
            "grid_volume_per_cell": 1.0,
            "initialized": True,
            "post_bc_max_velocity": 0.0,
            "post_bc_divergence_norm": 0.0,
            "bc_violation_count": 0,
        }

        # ---------------------------------------------------------
        # History (optional but required key)
        # ---------------------------------------------------------
        self["history"] = {
            "times": [],
            "diverggence_norms": [],
            "max_velocity_history": [],
            "ppe_iterations_history": [],
        }

        # ---------------------------------------------------------
        # Final flags
        # ---------------------------------------------------------
        self["initialized"] = True
        self["ready_for_time_loop"] = True
