# tests/step3/test_update_health.py

import numpy as np
from src.step3.update_health import update_health

def test_zero_velocity(minimal_state):
    state = minimal_state
    update_health(state)
    assert state["Health"]["post_correction_divergence_norm"] == 0.0
    assert state["Health"]["max_velocity_magnitude"] == 0.0

def test_uniform_velocity(minimal_state):
    state = minimal_state
    state["U"].fill(2.0)
    update_health(state)
    assert state["Health"]["max_velocity_magnitude"] == 2.0

def test_divergent_field(minimal_state):
    state = minimal_state
    pattern = np.ones_like(state["P"])
    state["_divergence_pattern"] = pattern
    update_health(state)
    assert state["Health"]["post_correction_divergence_norm"] > 0.0

def test_minimal_grid():
    state = {
        "Mask": np.ones((1,1,1), int),
        "is_fluid": np.ones((1,1,1), bool),
        "U": np.zeros((2,1,1)),
        "V": np.zeros((1,2,1)),
        "W": np.zeros((1,1,2)),
        "P": np.zeros((1,1,1)),
        "Constants": {"dt":0.1,"dx":1,"dy":1,"dz":1},
        "Operators": {"divergence": lambda U,V,W,s: np.zeros((1,1,1))},
    }
    update_health(state)
