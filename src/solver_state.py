# src/solver_state.py

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

# =========================================================
# THE PARENT GUARD
# =========================================================

class ValidatedContainer:
    """
    The 'Security Guard' Parent Class.
    Handles checking for None and Type validation for all sub-classes.
    """
    
    def _get_safe(self, name: str) -> Any:
        """Checks if the internal value is None. If not, returns it."""
        # We look for the 'private' version of the variable
        attr_name = f"_{name}"
        
        # Check if the attribute even exists on the object at all
        if not hasattr(self, attr_name):
            raise AttributeError(
                f"Coding Error: '{attr_name}' is not defined in {self.__class__.__name__}. "
                f"Check your class definition."
            )
            
        val = getattr(self, attr_name)
        
        if val is None:
            raise RuntimeError(
                f"Access Error: '{name}' in {self.__class__.__name__} has not been initialized. "
                f"Did you skip a function in orchestrate_step1?"
            )
        return val

    def _set_safe(self, name: str, value: Any, expected_type: type):
        """Ensures the value is the correct type before saving it."""
        if value is not None and not isinstance(value, expected_type):
            raise TypeError(
                f"Validation Error: '{name}' must be {expected_type}, "
                f"but got {type(value)}."
            )
        setattr(self, f"_{name}", value)

# =========================================================
# STEP 1: THE DEPARTMENT SAFES
# =========================================================

@dataclass
class SolverConfig(ValidatedContainer): # Added Guard for safety
    """Step 0: Global instructions & Numerical Tuning."""
    case_name: str = "default_case"
    method: str = "jacobi"
    precision: str = "float64"
    
    # --- Slots for config.json data ---
    # We initialize as None so the Guard knows they MUST be loaded
    _ppe_tolerance: float = None
    _ppe_atol: float = None
    _ppe_max_iter: int = None

    @property
    def ppe_tolerance(self) -> float: return self._get_safe("ppe_tolerance")
    @ppe_tolerance.setter
    def ppe_tolerance(self, v: float): self._set_safe("ppe_tolerance", v, float)

    @property
    def ppe_atol(self) -> float: return self._get_safe("ppe_atol")
    @ppe_atol.setter
    def ppe_atol(self, v: float): self._set_safe("ppe_atol", v, float)

    @property
    def ppe_max_iter(self) -> int: return self._get_safe("ppe_max_iter")
    @ppe_max_iter.setter
    def ppe_max_iter(self, v: int): self._set_safe("ppe_max_iter", v, int)

@dataclass
class GridContext(ValidatedContainer):
    """Step 1a: The Spatial World."""
    _nx: int = None; _ny: int = None; _nz: int = None
    _x_min: float = None; _x_max: float = None
    _y_min: float = None; _y_max: float = None
    _z_min: float = None; _z_max: float = None

    @property
    def nx(self) -> int: return self._get_safe("nx")
    @nx.setter
    def nx(self, val: int): self._set_safe("nx", val, int)

    @property
    def ny(self) -> int: return self._get_safe("ny")
    @ny.setter
    def ny(self, val: int): self._set_safe("ny", val, int)

    @property
    def nz(self) -> int: return self._get_safe("nz")
    @nz.setter
    def nz(self, val: int): self._set_safe("nz", val, int)

    @property
    def x_min(self) -> float: return self._get_safe("x_min")
    @x_min.setter
    def x_min(self, val: float): self._set_safe("x_min", val, float)

    @property
    def x_max(self) -> float: return self._get_safe("x_max")
    @x_max.setter
    def x_max(self, val: float): self._set_safe("x_max", val, float)

    @property
    def y_min(self) -> float: return self._get_safe("y_min")
    @y_min.setter
    def y_min(self, val: float): self._set_safe("y_min", val, float)

    @property
    def y_max(self) -> float: return self._get_safe("y_max")
    @y_max.setter
    def y_max(self, val: float): self._set_safe("y_max", val, float)

    @property
    def z_min(self) -> float: return self._get_safe("z_min")
    @z_min.setter
    def z_min(self, val: float): self._set_safe("z_min", val, float)

    @property
    def z_max(self) -> float: return self._get_safe("z_max")
    @z_max.setter
    def z_max(self, val: float): self._set_safe("z_max", val, float)

    @property
    def dx(self) -> float: 
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float: 
        return (self.y_max - self.y_min) / self.ny

    @property
    def dz(self) -> float: 
        return (self.z_max - self.z_min) / self.nz

@dataclass
class FieldData(ValidatedContainer):
    """Step 1b: The Memory Map. Staggered arrays."""
    _P: np.ndarray = None; _U: np.ndarray = None
    _V: np.ndarray = None; _W: np.ndarray = None

    @property
    def P(self) -> np.ndarray: return self._get_safe("P")
    @P.setter
    def P(self, val: np.ndarray): self._set_safe("P", val, np.ndarray)

    @property
    def U(self) -> np.ndarray: return self._get_safe("U")
    @U.setter
    def U(self, val: np.ndarray): self._set_safe("U", val, np.ndarray)

    @property
    def V(self) -> np.ndarray: return self._get_safe("V")
    @V.setter
    def V(self, val: np.ndarray): self._set_safe("V", val, np.ndarray)

    @property
    def W(self) -> np.ndarray: return self._get_safe("W")
    @W.setter
    def W(self, val: np.ndarray): self._set_safe("W", val, np.ndarray)

@dataclass
class MaskData(ValidatedContainer):
    """
    Step 1d: The Geometry Blueprint.
    Refined specifically from map_geometry_mask outputs.
    """
    _mask: np.ndarray = None            
    _is_fluid: np.ndarray = None        
    _is_boundary: np.ndarray = None     

    @property
    def mask(self) -> np.ndarray: return self._get_safe("mask")
    @mask.setter
    def mask(self, v: np.ndarray): self._set_safe("mask", v, np.ndarray)

    @property
    def is_fluid(self) -> np.ndarray: return self._get_safe("is_fluid")
    @is_fluid.setter
    def is_fluid(self, v: np.ndarray): self._set_safe("is_fluid", v, np.ndarray)

    @property
    def is_boundary(self) -> np.ndarray: return self._get_safe("is_boundary")
    @is_boundary.setter
    def is_boundary(self, v: np.ndarray): self._set_safe("is_boundary", v, np.ndarray)

@dataclass
class FluidProperties(ValidatedContainer):
    """Step 1c: The Physics Safe."""
    _rho: float = None; _mu: float = None

    @property
    def rho(self) -> float: return self._get_safe("rho")
    @rho.setter
    def rho(self, val: float): self._set_safe("rho", val, float)

    @property
    def mu(self) -> float: return self._get_safe("mu")
    @mu.setter
    def mu(self, val: float): self._set_safe("mu", val, float)

@dataclass
class OperatorStorage(ValidatedContainer):
    """Step 2c: The Sparse Matrix Engine (Calculus Operators)."""
    _divergence: Any = None
    _grad_x: Any = None; _grad_y: Any = None; _grad_z: Any = None
    _laplacian: Any = None

    @property
    def divergence(self) -> Any: return self._get_safe("divergence")
    @divergence.setter
    def divergence(self, v: Any): setattr(self, "_divergence", v)

    @property
    def grad_x(self) -> Any: return self._get_safe("grad_x")
    @grad_x.setter
    def grad_x(self, v: Any): setattr(self, "_grad_x", v)

    @property
    def grad_y(self) -> Any: return self._get_safe("grad_y")
    @grad_y.setter
    def grad_y(self, v: Any): setattr(self, "_grad_y", v)

    @property
    def grad_z(self) -> Any: return self._get_safe("grad_z")
    @grad_z.setter
    def grad_z(self, v: Any): setattr(self, "_grad_z", v)

    @property
    def laplacian(self) -> Any: return self._get_safe("laplacian")
    @laplacian.setter
    def laplacian(self, v: Any): setattr(self, "_laplacian", v)

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

@dataclass
class SolverState:
    """
    The Project Constitution: Article 3 (The Universal State Container).
    This version is strictly mapped to Step 1 & Step 2 Category 1.
    """

    # --- 1. Hardened Safe Objects ---
    config: SolverConfig = field(default_factory=SolverConfig)
    grid: GridContext = field(default_factory=GridContext)
    fields: FieldData = field(default_factory=FieldData)
    masks: MaskData = field(default_factory=MaskData)
    fluid: FluidProperties = field(default_factory=FluidProperties)
    operators: OperatorStorage = field(default_factory=OperatorStorage)

    # --- 2. Orchestrator-Driven Containers ---
    # These hold the dictionaries and tables returned by Step 1 logic
    constants: Dict[str, Any] = field(default_factory=dict)
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)

    # --- 3. Global Simulation State ---
    iteration: int = 0
    time: float = 0.0
    # Logic Gate for Step 2:
    ready_for_time_loop: bool = False

    # ---------------------------------------------------------
    # Attribute Interface (Facade)
    # ---------------------------------------------------------
    # These properties allow the math engine to access data 
    # without worrying about the underlying container structure.

    @property
    def pressure(self) -> np.ndarray:
        return self.fields.P

    @property
    def velocity_u(self) -> np.ndarray:
        return self.fields.U

    @property
    def velocity_v(self) -> np.ndarray:
        return self.fields.V

    @property
    def velocity_w(self) -> np.ndarray:
        return self.fields.W

    @property
    def is_fluid(self) -> np.ndarray:
        """Shortcut to the logical fluid mask."""
        return self.masks.is_fluid

    @property
    def dt(self) -> float:
        """Safe access to the time-step from the constants dict."""
        val = self.constants.get("dt")
        if val is None:
            raise RuntimeError(
                "Access Error: 'dt' not found in constants. "
                "Check compute_derived_constants."
            )
        return val

    # --- Step 2 Numerical Shortcuts ---
    
    @property
    def inv_dx(self) -> float:
        """Returns pre-calculated 1/dx or calculates on the fly."""
        return self.constants.get("inv_dx", 1.0 / self.grid.dx)

    @property
    def inv_dy(self) -> float:
        """Returns pre-calculated 1/dy or calculates on the fly."""
        return self.constants.get("inv_dy", 1.0 / self.grid.dy)

    @property
    def inv_dz(self) -> float:
        """Returns pre-calculated 1/dz or calculates on the fly."""
        return self.constants.get("inv_dz", 1.0 / self.grid.dz)

    # ---------------------------------------------------------
    # Serialization (Scale Guard Compliant)
    # ---------------------------------------------------------
    def to_json_safe(self) -> Dict[str, Any]:
        """Convert state to JSON-safe dict."""
        try:
            from scipy.sparse import issparse
        except ImportError:
            def issparse(obj): return False

        def convert(value):
            if issparse(value):
                return {
                    "type": str(value.format),
                    "shape": list(value.shape),
                    "nnz": int(value.nnz)
                }
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            if callable(value):
                return None
            if hasattr(value, "item") and not isinstance(value, (list, dict, np.ndarray)):
                return value.item()
            if value is None:
                return None
            return value

        result = {}
        for key, value in self.__dict__.items():
            result[key] = convert(value)
        return result