# 1. Check the ElasticManager constructor (See if it accepts initial_dt)
grep -n "def __init__" src/common/elasticity.py

# 2. Check the SimulationContext (Verify it matches your requested structure)
cat -n src/common/simulation_context.py

# 3. Check the SolverConfig slots (Ensure we don't have '_dt' causing a conflict)
grep "_dt" src/common/solver_config.py

# 1. Restore the Gold Standard config.json
cat << 'EOF' > config.json
{ 
   "dt_min_limit": 0.01, 
   "ppe_tolerance": 1e-6, 
   "ppe_atol": 1e-10, 
   "ppe_max_iter": 1000, 
   "ppe_omega": 1.1, 
   "divergence_threshold": 1e6 
}
EOF

# 2. Fix SimulationContext: Remove the injection logic and keep it as a clean Factory
cat << 'EOF' > src/common/simulation_context.py
from dataclasses import dataclass
from src.common.solver_config import SolverConfig
from src.common.solver_input import SolverInput

@dataclass
class SimulationContext:
    """
    Acts as the primary dependency injection container for the solver.
    """
    input_data: SolverInput
    config: SolverConfig

    @classmethod
    def create(cls, input_dict: dict, config_dict: dict) -> "SimulationContext":
        # Rule 4: Extract physical data first
        input_data = SolverInput.from_dict(input_dict)

        # Rule 5: Ensure 'dt' doesn't contaminate the static config object
        config_dict.pop("dt", None)
        config = SolverConfig(**config_dict)

        return cls(input_data=input_data, config=config)
EOF

# 3. Fix SolverConfig: Ensure slots match the config.json keys exactly
cat << 'EOF' > src/common/solver_config.py
from dataclasses import dataclass
from src.common.base_container import ValidatedContainer

@dataclass
class SolverConfig(ValidatedContainer):
    """
    Static numerical configuration. No dynamic state allowed.
    """
    __slots__ = [
        '_dt_min_limit', '_ppe_tolerance', '_ppe_atol', 
        '_ppe_max_iter', '_ppe_omega', '_divergence_threshold'
    ]

    def __init__(self, **kwargs):
        # Map JSON keys to slots
        self.dt_min_limit = kwargs.get('dt_min_limit')
        self.ppe_tolerance = kwargs.get('ppe_tolerance')
        self.ppe_atol = kwargs.get('ppe_atol')
        self.ppe_max_iter = kwargs.get('ppe_max_iter')
        self.ppe_omega = kwargs.get('ppe_omega')
        self.divergence_threshold = kwargs.get('divergence_threshold')

    @property
    def dt_min_limit(self) -> float: return self._get_safe("dt_min_limit")
    @dt_min_limit.setter
    def dt_min_limit(self, v: float): self._set_safe("dt_min_limit", v, float)

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

    @property
    def ppe_omega(self) -> float: return self._get_safe("ppe_omega")
    @ppe_omega.setter
    def ppe_omega(self, v: float): self._set_safe("ppe_omega", v, float)

    @property
    def divergence_threshold(self) -> float: return self._get_safe("divergence_threshold")
    @divergence_threshold.setter
    def divergence_threshold(self, v: float): self._set_safe("divergence_threshold", v, float)
EOF

# 4. Final Alignment for main_solver.py line 73:
# In your main_solver.py, you call: elasticity = ElasticManager(context.config)
# BUT your elasticity.py __init__ expects: (self, config, initial_dt)
# We must sync them. Let's update main_solver.py to pass the dt.

sed -i 's/ElasticManager(context.config)/ElasticManager(context.config, context.input_data.simulation_parameters.time_step)/' src/main_solver.py