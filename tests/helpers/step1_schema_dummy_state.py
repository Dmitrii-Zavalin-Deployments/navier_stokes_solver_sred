# tests/helpers/step1_schema_dummy_state.py

class Step1SchemaDummyState(dict):
    """
    Fully schema‑compliant Step‑1 output dummy.
    Matches the Step 1 Output Schema exactly.
    Produces JSON‑serializable lists (no numpy arrays).
    """

    def __init__(self, nx, ny, nz):
        super().__init__()

        # -----------------------------
        # grid (required)
        # -----------------------------
        self["grid"] = {
            "x_min": 0.0,
            "x_max": float(nx),
            "y_min": 0.0,
            "y_max": float(ny),
            "z_min": 0.0,
            "z_max": float(nz),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
        }

        # -----------------------------
        # helper constructors
        # -----------------------------
        def zeros(shape):
            return [
                [
                    [0.0 for _ in range(shape[2])]
                    for _ in range(shape[1])
                ]
                for _ in range(shape[0])
            ]

        def ones_int(shape):
            return [
                [
                    [1 for _ in range(shape[2])]
                    for _ in range(shape[1])
                ]
                for _ in range(shape[0])
            ]

        # -----------------------------
        # fields (required)
        # Staggered MAC‑grid shapes:
        #   U: (nx+1, ny,   nz)
        #   V: (nx,   ny+1, nz)
        #   W: (nx,   ny,   nz+1)
        #   P: (nx,   ny,   nz)
        # -----------------------------
        self["fields"] = {
            "P": zeros((nx, ny, nz)),
            "U": zeros((nx + 1, ny, nz)),
            "V": zeros((nx, ny + 1, nz)),
            "W": zeros((nx, ny, nz + 1)),
            "Mask": ones_int((nx, ny, nz)),  # values ∈ {-1,0,1}
        }

        # -----------------------------
        # mask_3d (required)
        # -----------------------------
        self["mask_3d"] = ones_int((nx, ny, nz))

        # -----------------------------
        # boundary_table (required)
        # -----------------------------
        self["boundary_table"] = {
            "x_min": [],
            "x_max": [],
            "y_min": [],
            "y_max": [],
            "z_min": [],
            "z_max": [],
        }

        # -----------------------------
        # constants (required)
        # These are Step‑1 constants; Step‑2 may override dt.
        # -----------------------------
        self["constants"] = {
            "rho": 1.0,
            "mu": 0.1,
            "dt": 0.1,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "inv_dx": 1.0,
            "inv_dy": 1.0,
            "inv_dz": 1.0,
            "inv_dx2": 1.0,
            "inv_dy2": 1.0,
            "inv_dz2": 1.0,
        }

        # -----------------------------
        # config (required)
        # -----------------------------
        self["config"] = {
            "domain": {
                "x_min": 0.0, "x_max": float(nx),
                "y_min": 0.0, "y_max": float(ny),
                "z_min": 0.0, "z_max": float(nz),
                "nx": nx, "ny": ny, "nz": nz,
            },
            "fluid": {
                "density": 1.0,
                "viscosity": 0.1,
            },
            "simulation": {
                "time_step": 0.1,
                "total_time": 1.0,
                "output_interval": 1,
            },
            "forces": {
                "force_vector": [0.0, 0.0, 0.0],
                "units": "N",
            },
            "boundary_conditions": [
                {
                    "role": "wall",
                    "type": "dirichlet",
                    "faces": ["x_min"],
                    "apply_to": ["velocity", "pressure"],
                    "velocity": [0.0, 0.0, 0.0],
                    "pressure": 0.0,
                    "pressure_gradient": 0.0,
                    "no_slip": True,
                }
            ],
            "geometry_definition": {
                "geometry_mask_flat": [1] * (nx * ny * nz),
                "geometry_mask_shape": [nx, ny, nz],
                "mask_encoding": {"fluid": 1, "solid": -1},
                "flattening_order": "C",
            },
        }
