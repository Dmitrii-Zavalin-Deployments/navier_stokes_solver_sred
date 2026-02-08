# tests/helpers/step2_schema_dummy_state.py

import numpy as np


class Step2SchemaDummyState(dict):
    """
    Fully schema‑compliant Step‑2 output dummy.
    Matches the Step 2 Output Schema exactly.
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

        # -----------------------------
        # grid (required)
        # -----------------------------
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

        # -----------------------------
        # fields (required)
        # -----------------------------
        self["fields"] = {
            "P": np.zeros((nx, ny, nz)),
            "U": np.zeros((nx + 1, ny, nz)),
            "V": np.zeros((nx, ny + 1, nz)),
            "W": np.zeros((nx, ny, nz + 1)),
        }

        # -----------------------------
        # mask (required)
        # -----------------------------
        mask = np.ones((nx, ny, nz), dtype=int)
        self["mask"] = mask

        # -----------------------------
        # is_fluid (required)
        # -----------------------------
        self["is_fluid"] = (mask == 1)

        # -----------------------------
        # is_boundary_cell (required)
        # -----------------------------
        self["is_boundary_cell"] = np.zeros((nx, ny, nz), dtype=bool)

        # -----------------------------
        # constants (required)
        # -----------------------------
        self["constants"] = {
            "rho": rho,
            "mu": mu,
            "dt": dt,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "inv_dx": 1.0 / dx,
            "inv_dy": 1.0 / dy,
            "inv_dz": 1.0 / dz,
            "inv_dx2": 1.0 / (dx * dx),
            "inv_dy2": 1.0 / (dy * dy),
            "inv_dz2": 1.0 / (dz * dz),
        }

        # -----------------------------
        # config (required)
        # Step‑2 passes Step‑1 input through unchanged.
        # We provide a minimal valid config.
        # -----------------------------
        self["config"] = {
            "domain": {
                "x_min": 0.0, "x_max": nx * dx,
                "y_min": 0.0, "y_max": ny * dy,
                "z_min": 0.0, "z_max": nz * dz,
                "nx": nx, "ny": ny, "nz": nz,
            },
            "fluid": {"density": rho, "viscosity": mu},
            "simulation": {"dt": dt},
            "forces": {"force_vector": [0, 0, 0], "units": "N"},
            "boundary_conditions": [],
            "geometry_definition": {
                "geometry_mask_flat": mask.flatten().tolist(),
                "geometry_mask_shape": [nx, ny, nz],
                "mask_encoding": {"fluid": 1, "solid": -1},
                "flattening_order": "C",
            },
        }

        # -----------------------------
        # operators (required)
        # Schema requires STRINGS, not callables.
        # -----------------------------
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

        # -----------------------------
        # ppe (required)
        # -----------------------------
        self["ppe"] = {
            "rhs_builder": "rhs_builder_op",
            "solver_type": "PCG",
            "tolerance": 1e-6,
            "max_iterations": 100,
            "ppe_is_singular": False,
        }

        # -----------------------------
        # health (required)
        # -----------------------------
        self["health"] = {
            "initial_divergence_norm": 0.0,
            "max_velocity_magnitude": 0.0,
            "cfl_advection_estimate": 0.0,
        }
