# src/solver_input.py

from dataclasses import dataclass, field
from typing import List
from src.common.base_container import ValidatedContainer

@dataclass
class GridInput(ValidatedContainer):
    """
    Step 1a: The Spatial World (Input Blueprint).
    Maps to the "grid" block in the JSON schema.
    """
    # --- Protected Slots ---
    _x_min: float = None; _x_max: float = None
    _y_min: float = None; _y_max: float = None
    _z_min: float = None; _z_max: float = None
    _nx: int = None; _ny: int = None; _nz: int = None

    # --- X-Axis Properties ---
    @property
    def x_min(self) -> float: return self._get_safe("x_min")
    @x_min.setter
    def x_min(self, v: float): self._set_safe("x_min", v, float)

    @property
    def x_max(self) -> float: return self._get_safe("x_max")
    @x_max.setter
    def x_max(self, v: float): self._set_safe("x_max", v, float)

    @property
    def nx(self) -> int: return self._get_safe("nx")
    @nx.setter
    def nx(self, v: int): self._set_safe("nx", v, int)

    # --- Y-Axis Properties ---
    @property
    def y_min(self) -> float: return self._get_safe("y_min")
    @y_min.setter
    def y_min(self, v: float): self._set_safe("y_min", v, float)

    @property
    def y_max(self) -> float: return self._get_safe("y_max")
    @y_max.setter
    def y_max(self, v: float): self._set_safe("y_max", v, float)

    @property
    def ny(self) -> int: return self._get_safe("ny")
    @ny.setter
    def ny(self, v: int): self._set_safe("ny", v, int)

    # --- Z-Axis Properties ---
    @property
    def z_min(self) -> float: return self._get_safe("z_min")
    @z_min.setter
    def z_min(self, v: float): self._set_safe("z_min", v, float)

    @property
    def z_max(self) -> float: return self._get_safe("z_max")
    @z_max.setter
    def z_max(self, v: float): self._set_safe("z_max", v, float)

    @property
    def nz(self) -> int: return self._get_safe("nz")
    @nz.setter
    def nz(self, v: int): self._set_safe("nz", v, int)

@dataclass
class FluidInput(ValidatedContainer):
    """
    Step 1c: The Physics Safe.
    Maps to the "fluid_properties" block in the JSON schema.
    """
    _density: float = None
    _viscosity: float = None

    @property
    def density(self) -> float:
        """The mass per unit volume (rho)."""
        return self._get_safe("density")
    
    @density.setter
    def density(self, v: float):
        """Schema: exclusiveMinimum: 0."""
        if v is not None and v <= 0:
            raise ValueError(f"Density must be strictly greater than 0, got {v}")
        self._set_safe("density", v, float)

    @property
    def viscosity(self) -> float:
        """The dynamic viscosity (mu)."""
        return self._get_safe("viscosity")
    
    @viscosity.setter
    def viscosity(self, v: float):
        """Schema: minimum: 0."""
        if v is not None and v < 0:
            raise ValueError(f"Viscosity cannot be negative, got {v}")
        self._set_safe("viscosity", v, float)

@dataclass
class InitialConditionsInput(ValidatedContainer):
    """
    Step 1b: The Starting State.
    Maps to the "initial_conditions" block in the JSON schema.
    """
    _velocity: list = None
    _pressure: float = None

    @property
    def velocity(self) -> list:
        """The initial [u, v, w] vector."""
        return self._get_safe("velocity")
    
    @velocity.setter
    def velocity(self, v: list):
        """Strict Schema Enforcement: exactly 3 elements."""
        if v is not None and len(v) != 3:
            raise ValueError(f"Velocity array must have exactly 3 elements, got {len(v)}")
        self._set_safe("velocity", v, list)

    @property
    def pressure(self) -> float:
        """The initial scalar pressure field value."""
        return self._get_safe("pressure")
    
    @pressure.setter
    def pressure(self, v: float):
        self._set_safe("pressure", v, float)

    # --- Tactical Facades for physics logic ---
    @property
    def u_init(self) -> float: return float(self.velocity[0])
    @property
    def v_init(self) -> float: return float(self.velocity[1])
    @property
    def w_init(self) -> float: return float(self.velocity[2])

@dataclass
class SimParamsInput(ValidatedContainer):
    """
    Step 0: Global Instructions & Numerical Tuning.
    Maps to the "simulation_parameters" block in the JSON schema.
    """
    _time_step: float = None
    _total_time: float = None
    _output_interval: int = None

    @property
    def time_step(self) -> float:
        """The discrete time increment (dt)."""
        return self._get_safe("time_step")
    
    @time_step.setter
    def time_step(self, v: float):
        """Schema: exclusiveMinimum: 0."""
        if v is not None and v <= 0:
            raise ValueError(f"time_step must be > 0, got {v}")
        self._set_safe("time_step", v, float)

    @property
    def total_time(self) -> float:
        """The final simulation time (T_end)."""
        return self._get_safe("total_time")
    
    @total_time.setter
    def total_time(self, v: float):
        """Schema: exclusiveMinimum: 0."""
        if v is not None and v <= 0:
            raise ValueError(f"total_time must be > 0, got {v}")
        self._set_safe("total_time", v, float)

    @property
    def output_interval(self) -> int:
        """Frequency of data exports (in iterations)."""
        return self._get_safe("output_interval")
    
    @output_interval.setter
    def output_interval(self, v: int):
        """Schema: minimum: 1."""
        if v is not None and v < 1:
            raise ValueError(f"output_interval must be >= 1, got {v}")
        self._set_safe("output_interval", v, int)

@dataclass
class BoundaryConditionItem(ValidatedContainer):
    """A single boundary face definition from the JSON array."""
    _location: str = None 
    _type: str = None      
    _values: dict = field(default_factory=dict)
    _comment: str = ""

    @property
    def location(self) -> str:
        return self._get_safe("location")

    @location.setter
    def location(self, v: str):
        valid_locations = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        if v not in valid_locations:
            raise ValueError(f"Invalid boundary location '{v}'. Must be one of {valid_locations}")
        self._set_safe("location", v, str)

    @property
    def type(self) -> str:
        return self._get_safe("type")

    @type.setter
    def type(self, v: str):
        valid_types = ["no-slip", "free-slip", "inflow", "outflow", "pressure"]
        if v not in valid_types:
            raise ValueError(f"Invalid BC type '{v}'. Must be one of {valid_types}")
        self._set_safe("type", v, str)

    @property
    def values(self) -> dict:
        return self._get_safe("values")

    @values.setter
    def values(self, v: dict):
        self._set_safe("values", v, dict)

    @property
    def comment(self) -> str:
        return self._get_safe("comment")

    @comment.setter
    def comment(self, v: str):
        self._set_safe("comment", v, str)

@dataclass
class BoundaryConditionsInput(ValidatedContainer):
    """Manager for the list of boundary condition items."""
    _items: List[BoundaryConditionItem] = field(default_factory=list)

    @property
    def items(self) -> List[BoundaryConditionItem]:
        return self._get_safe("items")

    @items.setter
    def items(self, v: list):
        """Converts raw list of dicts from JSON into BoundaryConditionItem objects."""
        processed = []
        if isinstance(v, list):
            for entry in v:
                if isinstance(entry, dict):
                    bc = BoundaryConditionItem()
                    bc.location = entry.get("location")
                    bc.type = entry.get("type")
                    bc.values = entry.get("values", {})
                    bc.comment = entry.get("comment", "")
                    processed.append(bc)
                else:
                    processed.append(entry)
        
        self._set_safe("items", processed, list)

@dataclass
class MaskInput(ValidatedContainer):
    """
    Step 1d: The Geometry Blueprint.
    Maps to the "mask" block in the JSON schema.
    """
    _data: list = None

    @property
    def data(self) -> list:
        """
        The flat integer array of length nx*ny*nz.
        1 = fluid, 0 = solid, -1 = boundary-fluid.
        """
        return self._get_safe("data")

    @data.setter
    def data(self, v: list):
        """Schema: enum check [-1, 0, 1] for all items."""
        if v is not None:
            # Enforce the strict enum values from the schema
            valid_entries = {-1, 0, 1}
            if not all(val in valid_entries for val in v):
                # Identifying the offending value for easier debugging
                offenders = {val for val in v if val not in valid_entries}
                raise ValueError(f"Mask contains invalid values {offenders}. Only -1, 0, 1 allowed.")
        
        self._set_safe("data", v, list)

@dataclass
class ExternalForcesInput(ValidatedContainer):
    """
    Step 2: Source Terms & Body Forces.
    Maps to the "external_forces" block in the JSON schema.
    """
    _force_vector: list = None
    _comment: str = ""

    @property
    def force_vector(self) -> list:
        """Physical source terms like gravity: [fx, fy, fz]."""
        return self._get_safe("force_vector")

    @force_vector.setter
    def force_vector(self, v: list):
        """Strict Schema Enforcement: exactly 3 elements."""
        if v is not None and len(v) != 3:
            raise ValueError(f"force_vector must have exactly 3 elements, got {len(v)}")
        self._set_safe("force_vector", v, list)

    @property
    def comment(self) -> str:
        """Optional metadata describing the force."""
        return self._get_safe("comment")

    @comment.setter
    def comment(self, v: str):
        self._set_safe("comment", v, str)

    # --- Tactical Facades ---
    @property
    def fx(self) -> float: return float(self.force_vector[0])
    @property
    def fy(self) -> float: return float(self.force_vector[1])
    @property
    def fz(self) -> float: return float(self.force_vector[2])