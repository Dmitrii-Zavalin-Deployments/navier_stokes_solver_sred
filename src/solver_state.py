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
class SolverConfig:
    """Step 0: Global instructions."""
    case_name: str = "default_case"
    method: str = "jacobi"
    precision: str = "float64"

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

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

@dataclass
class SolverState:
    """
    The Project Constitution: Article 3 (The Universal State Container).
    """

    # ---------------------------------------------------------
    # Step 0: Input / configuration
    # ---------------------------------------------------------
    config: SolverConfig = field(default_factory=SolverConfig)

    # ---------------------------------------------------------
    # Step 1: Grid & Fields (Solidified Safes)
    # ---------------------------------------------------------
    grid: GridContext = field(default_factory=GridContext)
    fields: FieldData = field(default_factory=FieldData)
    fluid: FluidProperties = field(default_factory=FluidProperties)
    
    # ---------------------------------------------------------
    # Remaining Architecture (Commented out until solidification)
    # ---------------------------------------------------------
    # mask: Optional[np.ndarray] = None                             
    # is_fluid: Optional[np.ndarray] = None
    # is_boundary_cell: Optional[np.ndarray] = None
    # is_solid: Optional[np.ndarray] = None
    
    # constants: Dict[str, Any] = field(default_factory=dict)       
    # fluid_properties: Dict[str, Any] = field(default_factory=dict) 
    # boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # health: Dict[str, Any] = field(default_factory=dict)          

    # operators: Dict[str, Any] = field(default_factory=dict)       
    # ppe: Dict[str, Any] = field(default_factory=dict)             

    # intermediate_fields: Dict[str, np.ndarray] = field(default_factory=dict) 
    # step3_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # history: Dict[str, List[float]] = field(default_factory=lambda: {
    #     "times": [],
    #     "divergence_norms": [],
    #     "max_velocity_history": [],
    #     "ppe_iterations_history": [],
    #     "energy_history": [],
    # })

    # P_ext: Optional[np.ndarray] = None
    # U_ext: Optional[np.ndarray] = None
    # V_ext: Optional[np.ndarray] = None
    # W_ext: Optional[np.ndarray] = None
    # step4_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # step5_outputs: Dict[str, Any] = field(default_factory=dict)

    # ready_for_time_loop: bool = False
    # iteration: int = 0
    # time: float = 0.0

    # ---------------------------------------------------------
    # Attribute Interface (Facade)
    # ---------------------------------------------------------
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