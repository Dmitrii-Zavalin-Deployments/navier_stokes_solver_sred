# src/common/simulation_context.py

from dataclasses import dataclass

from src.common.solver_config import SolverConfig
from src.common.solver_input import SolverInput


@dataclass
class SimulationContext:
    """
    Acts as the primary dependency injection container for the solver.
    It encapsulates both the physical problem definition and the 
    numerical execution configuration.
    """
    input_data: SolverInput
    config: SolverConfig

    @classmethod
    def create(cls, input_dict: dict, config_dict: dict) -> "SimulationContext":
        """
        Factory method to assemble the context from separate data sources.
        """
        # Validate and create the input container
        input_data = SolverInput.from_dict(input_dict)
        
        # Assemble numerical settings from the 'solver_settings' block
        settings = config_dict.get("solver_settings", {})
        
        # Pass the parameters directly into the constructor.
        # SolverConfig will handle the validation of these values 
        # inside its internal __init__ logic.
        config = SolverConfig(
            ppe_tolerance=settings.get("ppe_tolerance"),
            ppe_atol=settings.get("ppe_atol"),
            ppe_max_iter=settings.get("ppe_max_iter"),
            ppe_omega=settings.get("ppe_omega")
        )
        
        return cls(input_data=input_data, config=config)