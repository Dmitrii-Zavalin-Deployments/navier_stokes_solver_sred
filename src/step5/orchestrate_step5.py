# src/step5/orchestrate_step5.py

from src.step5.io_archivist import save_snapshot


def orchestrate_step5(state) -> object:
    """
    Step 5: The Archivist.
    Coordinates data persistence based on simulation iteration.
    
    Compliance:
    - Maintains the separation between the 'Logic' (the orchestrator's timing) 
      and the 'Foundation' (the buffer saved by the archivist).
    - Ensures zero-copy access to the global fields_buffer via the state object.
    """
    # The output_interval is a configuration property; the decision to save
    # is a logic-layer operation that does not affect memory layout.
    if state.iteration % state.config.output_interval == 0:
        # save_snapshot interacts directly with state.fields_buffer, 
        # leveraging the FI schema for performance-critical data serialization.
        save_snapshot(state)
        
    return state