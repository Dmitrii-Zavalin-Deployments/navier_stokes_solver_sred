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
        attr_name = f"_{name}"
        if not hasattr(self, attr_name):
            raise AttributeError(
                f"Coding Error: '{attr_name}' is not defined in {self.__class__.__name__}."
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
                f"Validation Error: '{name}' must be {expected_type}, but got {type(value)}."
            )
        setattr(self, f"_{name}", value)

    def to_dict(self) -> dict:
        """
        Recursively converts the container to a JSON-serializable dictionary.
        Strips leading underscores and converts NumPy arrays to lists.
        """
        out = {}
        # We iterate over instance variables to capture the state
        for attr, val in self.__dict__.items():
            # Clean the key name (strip leading underscore)
            clean_key = attr.lstrip('_')
            
            # 1. Handle nested containers
            if isinstance(val, ValidatedContainer):
                out[clean_key] = val.to_dict()
            # 2. Handle NumPy arrays (Crucial for JSON)
            elif isinstance(val, np.ndarray):
                out[clean_key] = val.tolist()
            # 3. Handle nested dictionaries (like boundary_conditions)
            elif isinstance(val, dict):
                # Ensure no numpy types inside dicts
                out[clean_key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                 for k, v in val.items()}
            # 4. Basic types
            else:
                out[clean_key] = val
        return out

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

    # Derived spacing properties
    @property
    def dx(self) -> float: 
        return float((self.x_max - self.x_min) / self.nx)

    @property
    def dy(self) -> float: 
        return float((self.y_max - self.y_min) / self.ny)

    @property
    def dz(self) -> float: 
        return float((self.z_max - self.z_min) / self.nz)

    # Performance Shortcuts: Inverse Spacing
    @property
    def inv_dx(self) -> float:
        return 1.0 / self.dx

    @property
    def inv_dy(self) -> float:
        return 1.0 / self.dy

    @property
    def inv_dz(self) -> float:
        return 1.0 / self.dz

@dataclass
class FieldData(ValidatedContainer):
    """
    Step 1, 3 & 4: The Memory Map. 
    Contains both the core physical fields and the extended boundary workhorses.

    ## 2. Updated Physical Fields (`state.fields`)
    These are updated **every single iteration** inside the loop by `orchestrate_step3_state`. 
    When Step 5 returns, these contain the "Final Snapshot" of the fluid.

    * **state.fields.U, V, W**: The corrected, divergence-free velocity vectors at t_final.
    * **state.fields.P**: The final pressure field required to maintain incompressibility.
    * **state.fields.U_ext, V_ext, W_ext, P_ext**: Extended arrays (with ghost cells).
    """
    # --- Core Fields (Step 1 & 3) ---
    _P: np.ndarray = None; _U: np.ndarray = None
    _V: np.ndarray = None; _W: np.ndarray = None

    # --- Extended Fields (Step 4) ---
    _P_ext: np.ndarray = None; _U_ext: np.ndarray = None
    _V_ext: np.ndarray = None; _W_ext: np.ndarray = None

    # --- Core Properties ---
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

    # --- Extended Properties (The Step 4 "Workhorses") ---
    @property
    def P_ext(self) -> np.ndarray: 
        """Pressure with ghost cells for Neumann BC enforcement."""
        return self._get_safe("P_ext")
    @P_ext.setter
    def P_ext(self, val: np.ndarray): self._set_safe("P_ext", val, np.ndarray)

    @property
    def U_ext(self) -> np.ndarray: 
        """U-velocity with ghost cells for stencil safety."""
        return self._get_safe("U_ext")
    @U_ext.setter
    def U_ext(self, val: np.ndarray): self._set_safe("U_ext", val, np.ndarray)

    @property
    def V_ext(self) -> np.ndarray: 
        """V-velocity with ghost cells for stencil safety."""
        return self._get_safe("V_ext")
    @V_ext.setter
    def V_ext(self, val: np.ndarray): self._set_safe("V_ext", val, np.ndarray)

    @property
    def W_ext(self) -> np.ndarray: 
        """W-velocity with ghost cells for stencil safety."""
        return self._get_safe("W_ext")
    @W_ext.setter
    def W_ext(self, val: np.ndarray): self._set_safe("W_ext", val, np.ndarray)

    def __setitem__(self, key: str, value: np.ndarray):
        """
        Step 3 & 4 Support: Handles both core and extended assignments.
        Example: state.fields["U_ext"] = padded_array
        """
        key_upper = key.upper()
        # Direct mapping for both standard and extended fields
        mapping = {
            "U": "U", "V": "V", "W": "W", "P": "P",
            "U_EXT": "U_ext", "V_EXT": "V_ext", "W_EXT": "W_ext", "P_EXT": "P_ext"
        }
        
        target = mapping.get(key_upper)
        if target:
            setattr(self, target, value)
        else:
            raise KeyError(f"Field '{key}' is not a recognized core or extended field.")

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

@dataclass
class AdvectionStructure(ValidatedContainer):
    """Step 2b: Momentum Transport Stencils."""
    _weights: Any = None
    _indices: Any = None

    @property
    def weights(self) -> Any: return self._get_safe("weights")
    @weights.setter
    def weights(self, v: Any): self._set_safe("weights", v, (np.ndarray, object))

    @property
    def indices(self) -> Any: return self._get_safe("indices")
    @indices.setter
    def indices(self, v: Any): self._set_safe("indices", v, (np.ndarray, object))

@dataclass
class PPEContext(ValidatedContainer):
    """
    Step 2d & 3: Pressure Poisson Equation System.
    These remain 'Hot' in memory to avoid costly re-assembly during Step 3.
    """
    _A: Any = None
    _preconditioner: Any = None

    @property
    def A(self) -> Any: 
        """
        The system matrix (LHS). Usually a large, sparse Laplacian-based 
        matrix representing the linear equations for pressure.
        """
        return self._get_safe("A")
    @A.setter
    def A(self, v: Any): self._set_safe("A", v, object)

    @property
    def preconditioner(self) -> Any: 
        """
        The mathematical 'shortcut' (e.g., Jacobi, ILU) used to 
        accelerate the convergence of the Pressure solver.
        """
        return self._get_safe("preconditioner")
    @preconditioner.setter
    def preconditioner(self, v: Any): self._set_safe("preconditioner", v, object)

@dataclass
class SimulationHealth(ValidatedContainer):
    """
    Step 2e & 3: Current "Vitals".
    Describes the state of the simulation at this exact moment.

    ## 4. Post-Flight Health Check (`state.health`)

    Generated by `finalize_simulation_health(state)`, this provides a high-level 
    summary of whether the run was a success or a failure.

    * **state.health.is_stable**: A final bool. If a NaN occurred at step 500 of 10,000, this will be False.
    * **state.health.max_u**: The absolute maximum velocity recorded across the entire volume at the final step.
    * **state.health.post_correction_divergence_norm**: The final proof of mass conservation (should be approx 0).
    """
    _max_u: float = None
    _divergence_norm: float = None
    _is_stable: bool = None
    _post_correction_divergence_norm: float = None

    @property
    def max_u(self) -> float: 
        """The speed of the fastest point in the simulation (exploding check)."""
        return self._get_safe("max_u")
    @max_u.setter
    def max_u(self, v: float): self._set_safe("max_u", v, float)

    @property
    def divergence_norm(self) -> float: 
        """Measures how 'un-physical' the initial or current flow field is."""
        return self._get_safe("divergence_norm")
    @divergence_norm.setter
    def divergence_norm(self, v: float): self._set_safe("divergence_norm", v, float)

    @property
    def is_stable(self) -> bool: 
        """Boolean flag: flips to False if NaNs or Infs are detected."""
        return self._get_safe("is_stable")
    @is_stable.setter
    def is_stable(self, v: bool): self._set_safe("is_stable", v, bool)

    @property
    def post_correction_divergence_norm(self) -> float:
        """Proves the pressure correction worked and mass is conserved (Target ~0)."""
        return self._get_safe("post_correction_divergence_norm")
    @post_correction_divergence_norm.setter
    def post_correction_divergence_norm(self, v: float): 
        self._set_safe("post_correction_divergence_norm", v, float)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Step 3 Support: Compatibility shim for dictionary-style access.
        Allows: state.health.get("max_velocity_magnitude")
        """
        mapping = {
            "post_correction_divergence_norm": "_post_correction_divergence_norm",
            "max_velocity_magnitude": "_max_u",
            "divergence_norm": "_divergence_norm",
            "is_stable": "_is_stable"
        }
        attr = mapping.get(key)
        if attr:
            # We use getattr directly to avoid the 'None' check of _get_safe 
            # if the user provided a default.
            val = getattr(self, attr, None)
            return val if val is not None else default
        return default

@dataclass
class SimulationHistory(ValidatedContainer):
    """
    Step 3 & 5: The Black Box Flight Recorder.
    Tracks the evolution of the simulation over time for plotting and 
    post-processing.
    """
    _times: list = field(default_factory=list)
    _divergence_norms: list = field(default_factory=list)
    _max_velocity_history: list = field(default_factory=list)
    _ppe_status_history: list = field(default_factory=list)
    _energy_history: list = field(default_factory=list)

    @property
    def times(self) -> list: 
        """A list of every time-stamp simulated so far."""
        return self._get_safe("times")

    @property
    def divergence_norms(self) -> list: 
        """A record of how the mass conservation error changed over time."""
        return self._get_safe("divergence_norms")

    @property
    def max_velocity_history(self) -> list: 
        """A graph-ready list of peak speeds (Stability tracking)."""
        return self._get_safe("max_velocity_history")

    @property
    def ppe_status_history(self) -> list: 
        """A list of solver outcomes (e.g., 'Converged', 'Failed')."""
        return self._get_safe("ppe_status_history")

    @property
    def energy_history(self) -> list: 
        """The total kinetic energy in the system (Physical decay/growth check)."""
        return self._get_safe("energy_history")

    def __getitem__(self, key: str) -> list:
        """Allows dictionary-style access: history['times']."""
        mapping = {
            "times": self.times,
            "divergence_norms": self.divergence_norms,
            "max_velocity_history": self.max_velocity_history,
            "ppe_status_history": self.ppe_status_history,
            "energy_history": self.energy_history
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(f"History key '{key}' not found.")

@dataclass
class Diagnostics(ValidatedContainer):
    """
    Step 4: Pre-Flight Readiness Audit.
    Summarizes memory usage and BC integrity before the heavy lifting begins.
    """
    _memory_footprint_gb: float = None
    _bc_verification_passed: bool = None
    _initial_cfl_dt: float = None

    @property
    def memory_footprint_gb(self) -> float: 
        """
        Total RAM usage of the extended arrays in GB.
        Calculated as: (Total Voxels * 8 bytes per float64 * Num Fields) / 1e9.
        """
        return self._get_safe("memory_footprint_gb")
    @memory_footprint_gb.setter
    def memory_footprint_gb(self, v: float): self._set_safe("memory_footprint_gb", v, float)

    @property
    def bc_verification_passed(self) -> bool: 
        """Confirms ghost cells were correctly populated by Step 4."""
        return self._get_safe("bc_verification_passed")
    @bc_verification_passed.setter
    def bc_verification_passed(self, v: bool): self._set_safe("bc_verification_passed", v, bool)

    @property
    def initial_cfl_dt(self) -> float: 
        """Maximum allowable time-step based on initial conditions."""
        return self._get_safe("initial_cfl_dt")
    @initial_cfl_dt.setter
    def initial_cfl_dt(self, v: float): self._set_safe("initial_cfl_dt", v, float)

    def __getitem__(self, key: str) -> Any:
        """Step 4 Support: Allows dict-style access for the legacy adapter."""
        mapping = {
            "memory_footprint": self.memory_footprint_gb,
            "bc_verification": self.bc_verification_passed,
            "cfl_limit": self.initial_cfl_dt
        }
        return mapping.get(key)

@dataclass
class OutputManifest(ValidatedContainer):
    """
    ## 5. Output Artifacts (Manifest)
    
    This tracks the physical location of files generated during Step 5.
    Essential for archiving logic and post-processing verification.
    """
    _output_directory: str = "output"
    _saved_snapshots: list[str] = field(default_factory=list)
    _final_checkpoint: str = None
    _log_file: str = None

    @property
    def output_directory(self) -> str:
        """The base folder where all results are stored."""
        return self._get_safe("output_directory")
    
    @output_directory.setter
    def output_directory(self, v: str): 
        self._set_safe("output_directory", v, str)

    @property
    def saved_snapshots(self) -> list[str]:
        """A list of full paths to every VTK/HDF5 file created."""
        return self._get_safe("saved_snapshots")

    @saved_snapshots.setter
    def saved_snapshots(self, v: list):
        """Allows bulk update of the snapshot registry (required by Step 5 assembly)."""
        self._set_safe("saved_snapshots", v, list)

    @property
    def final_checkpoint(self) -> str:
        """The path to the last .npy or .h5 state for restarting."""
        return self._get_safe("final_checkpoint")
    
    @final_checkpoint.setter
    def final_checkpoint(self, v: str): 
        self._set_safe("final_checkpoint", v, str)

    @property
    def log_file(self) -> str:
        """Path to the .txt or .log file containing solver convergence stats."""
        return self._get_safe("log_file")
    
    @log_file.setter
    def log_file(self, v: str): 
        self._set_safe("log_file", v, str)

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

@dataclass
class SolverState:
    """
    The Project Constitution: Article 3 (The Universal State Container).
    
    This central container synchronizes all data across the five-step pipeline:
    1. Domain Setup -> 2. Operator Assembly -> 3. Iterative Loop -> 
    4. Boundary Enforcement -> 5. Artifact Export.
    """

    # --- 1. Hardened Safe Objects ---
    # These containers house all validated numerical and physical data.
    config: SolverConfig = field(default_factory=SolverConfig)
    grid: GridContext = field(default_factory=GridContext)
    fields: FieldData = field(default_factory=FieldData)
    masks: MaskData = field(default_factory=MaskData)
    fluid: FluidProperties = field(default_factory=FluidProperties)
    operators: OperatorStorage = field(default_factory=OperatorStorage)
    advection: AdvectionStructure = field(default_factory=AdvectionStructure)
    ppe: PPEContext = field(default_factory=PPEContext)
    health: SimulationHealth = field(default_factory=SimulationHealth)
    history: SimulationHistory = field(default_factory=SimulationHistory)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    manifest: OutputManifest = field(default_factory=OutputManifest)

    # --- 2. Global Simulation Odometer ---
    # Tracks the physical and temporal progression of the solver.
    # | Attribute          | Type  | Description                                   |
    # |--------------------|-------|-----------------------------------------------|
    # | state.time         | float | Total simulated physical time (seconds).      |
    # | state.iteration    | int   | Current time-step count.                      |
    iteration: int = 0
    time: float = 0.0

    # --- 3. Execution Gate ---
    # ready_for_time_loop: A safety toggle. It must be flipped to True 
    # only after Steps 1-4 successfully initialize all required data.
    ready_for_time_loop: bool = False

    # ---------------------------------------------------------
    # Attribute Interface (Facade)
    # ---------------------------------------------------------
    # These properties provide clean aliases to deeply nested data.
    
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
        return self.masks.is_fluid

    # --- Step 4 Extended Field Shortcuts ---
    @property
    def U_ext(self) -> np.ndarray: return self.fields.U_ext
    @property
    def V_ext(self) -> np.ndarray: return self.fields.V_ext
    @property
    def W_ext(self) -> np.ndarray: return self.fields.W_ext
    @property
    def P_ext(self) -> np.ndarray: return self.fields.P_ext

    # --- Numerical Shortcuts (Validated by Config & Grid) ---
    @property
    def dt(self) -> float:
        """Accesses the validated time-step from the configuration safe."""
        return self.config.dt

    @property
    def inv_dx(self) -> float:
        """Pre-calculated inverse grid spacing for performance."""
        return self.grid.inv_dx

    @property
    def inv_dy(self) -> float:
        return self.grid.inv_dy

    @property
    def inv_dz(self) -> float:
        return self.grid.inv_dz

    # ---------------------------------------------------------
    # Legacy & Pipeline Support
    # ---------------------------------------------------------
    @property
    def step4_diagnostics(self) -> Diagnostics:
        """Alias for the legacy Step 4 adapter."""
        return self.diagnostics

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Extracts a flat dictionary for legacy Step 4 orchestrators 
        that do not yet support the full State object.
        """
        return {
            "U_ext": self.U_ext,
            "V_ext": self.V_ext,
            "W_ext": self.W_ext,
            "P_ext": self.P_ext,
            "step4_diagnostics": self.step4_diagnostics,
            "ready_for_time_loop": self.ready_for_time_loop
        }

    # ---------------------------------------------------------
    # Serialization (Contract Bridge)
    # ---------------------------------------------------------
    def to_json_safe(self) -> dict:
        """
        The Clean Contract: Article 3 Alignment.
        
        Converts the departmentalized state into a structured dictionary.
        This removes all 'Artificial Bridges' and legacy root-level lifting
        to satisfy the Zero-Debt Mandate.
        """
        return {
            # --- Global Metadata ---
            "time": self.time,
            "iteration": self.iteration,
            "ready_for_time_loop": self.ready_for_time_loop,

            # --- Departmental Safes ---
            "config": self.config.to_dict(),
            "grid": self.grid.to_dict(),
            "fields": self.fields.to_dict(),
            "masks": self.masks.to_dict(),
            "fluid": self.fluid.to_dict(),
            "operators": self.operators.to_dict(),
            "advection": self.advection.to_dict(),
            "ppe": self.ppe.to_dict(),
            "health": self.health.to_dict(),
            "history": self.history.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "manifest": self.manifest.to_dict()
        }