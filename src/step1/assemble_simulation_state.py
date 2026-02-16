# src/step1/assemble_simulation_state.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from .types import (
    Config,
    DerivedConstants,
    Fields,
    GridConfig,
)


def assemble_simulation_state(
    config: Config,
    grid: GridConfig,
    fields: Fields,
    mask_3d: np.ndarray,
    bc_table: Dict[str, Any],
    constants: DerivedConstants,
) -> Dict[str, Any]:
    """
    Assemble a solver-ready Step 1 state (dict form).

    All structured objects (Config, GridConfig, Fields, DerivedConstants)
    are converted into plain dictionaries so that the result is JSON-safe
    and compatible with the Step 1 orchestrator and output schema.
    """

    # -----------------------------
    # Convert grid → dict
    # -----------------------------
    grid_dict = {
        "x_min": grid.x_min,
        "x_max": grid.x_max,
        "y_min": grid.y_min,
        "y_max": grid.y_max,
        "z_min": grid.z_min,
        "z_max": grid.z_max,
        "nx": grid.nx,
        "ny": grid.ny,
        "nz": grid.nz,
        "dx": grid.dx,
        "dy": grid.dy,
        "dz": grid.dz,
    }

    # -----------------------------
    # Convert fields → dict
    # -----------------------------
    fields_dict = {
        "P": fields.P,
        "U": fields.U,
        "V": fields.V,
        "W": fields.W,
        "Mask": fields.Mask,
    }

    # -----------------------------
    # Convert constants → dict
    # -----------------------------
    constants_dict = {
        "rho": constants.rho,
        "mu": constants.mu,
        "dt": constants.dt,
        "dx": constants.dx,
        "dy": constants.dy,
        "dz": constants.dz,
        "inv_dx": constants.inv_dx,
        "inv_dy": constants.inv_dy,
        "inv_dz": constants.inv_dz,
        "inv_dx2": constants.inv_dx2,
        "inv_dy2": constants.inv_dy2,
        "inv_dz2": constants.inv_dz2,
    }

    # -----------------------------
    # Convert config → dict (updated schema)
    # -----------------------------
    config_dict = {
        "domain": config.domain,
        "fluid_properties": config.fluid_properties,
        "simulation_parameters": config.simulation_parameters,
        "external_forces": config.external_forces,
        "boundary_conditions": config.boundary_conditions,
        "geometry": config.geometry,
    }

    # -----------------------------
    # Assemble final Step 1 state
    # -----------------------------
    state = {
        "grid": grid_dict,
        "fields": fields_dict,
        "mask": mask_3d,          # UPDATED: correct key name
        "boundary_conditions": bc_table,
        "constants": constants_dict,
        "config": config_dict,
    }

    return state
