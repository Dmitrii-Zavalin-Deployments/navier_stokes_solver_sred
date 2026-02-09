# tests/helpers/step2_schema_dummy_state.py

import numpy as np


class Step2SchemaDummyState(dict):
    """
    Fully schema‑compliant Step‑2 output dummy.
    Matches the Step‑2 Output Schema exactly.
    """

    def __init__(
        self,
        nx,
        ny,
        nz,
        *,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        dt=0.1,
        rho=1.0,
        mu=0.1,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # grid (required)
        # ------------------------------------------------------------
        self["grid"] = {
            "x_min": 0.0,
            "x_max": nx * dx,
            "y_min": 0.0,
            "y_max": ny * dy,
            "z_min": 0.0,
            "z_max": nz * dz,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
        }

        # ------------------------------------------------------------
        # fields (required)
        # ------------------------------------------------------------
        self["fields"] = {
            "P": np.zeros((nx, ny, nz)).tolist(),
            "U": np.zeros((nx + 1, ny, nz)).tolist(),
            "V": np.zeros((nx, ny + 1, nz)).tolist(),
            "W": np.zeros((nx, ny, nz + 1)).tolist(),
        }

        # ------------------------------------------------------------
        # mask (required) — tristate [-1, 0, 1]
        # ------------------------------------------------------------
        mask = np.ones((nx, ny, nz), dtype=int)
        self["mask"] = mask.tolist()

        # ------------------------------------------------------------
        # is_fluid (required)
        # ------------------------------------------------------------
        self["is_fluid"] = (mask != 0).tolist()

        # ------------------------------------------------------------
        # is_boundary_cell (required)
        # ------------------------------------------------------------
        self["is_boundary_cell"] = (mask == -1).tolist()

        # ------------------------------------------------------------
        # mask_semantics (required by property‑based tests)
        # ------------------------------------------------------------
        self["mask_semantics"] = {
            "mask": mask.tolist(),
            "is_fluid": (mask == 1).tolist(),
            "is_solid": (mask == 0).tolist(),
        }

        # ------------------------------------------------------------
        # constants (required)
        # ------------------------------------------------------------
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy
        inv_dz = 1.0 / dz

        self["constants"] = {
            "rho": float(rho),
            "mu": float(mu),
            "dt": float(dt),
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            "inv_dx": inv_dx,
            "inv_dy": inv_dy,
            "inv_dz": inv_dz,
            "inv_dx2": inv_dx * inv_dx,
            "inv_dy2": inv_dy * inv_dy,
            "inv_dz2": inv_dz * inv_dz,
        }

        # ------------------------------------------------------------
        # config (required)
        # ------------------------------------------------------------
        self["config"] = {
            "fluid": {"density": rho, "viscosity": mu},
            "simulation": {"dt": dt, "advection_scheme": "central"},
            "domain": {
                "x_min": 0.0, "x_max": nx * dx,
                "y_min": 0.0, "y_max": ny * dy,
                "z_min": 0.0, "z_max": nz * dz,
                "nx": nx, "ny": ny, "nz": nz,
            },
            "forces": {"force_vector": [0, 0, 0]},
            "boundary_conditions": [],
        }

        # ------------------------------------------------------------
        # operators (required) — STRINGS ONLY
        # ------------------------------------------------------------
        self["operators"] = {
            "divergence": "divergence_op",
            "gradient_p_x": "grad_px_op",
            "gradient_p_y": "grad_py_op",
            "gradient_p_z": "grad_pz_op",
            "laplacian_u": "lap_u_op",
            "laplacian_v": "lap_v_op",
            "laplacian_w": "lap_w_op",
            "advection_u": "adv_u_op",
            "advection_v": "adv_v_op",
            "advection_w": "adv_w_op",
            "interpolation_stencils": None,
        }

        # ------------------------------------------------------------
        # ppe (required)
        # ------------------------------------------------------------
        self["ppe"] = {
            "rhs_builder": "rhs_builder_op",
            "solver_type": "PCG",
            "tolerance": 1e-6,
            "max_iterations": 1000,
            "ppe_is_singular": False,
        }

        # ------------------------------------------------------------
        # health (required)
        # ------------------------------------------------------------
        self["health"] = {
            "initial_divergence_norm": 0.0,
            "max_velocity_magnitude": 0.0,
            "cfl_advection_estimate": 0.0,
        }
