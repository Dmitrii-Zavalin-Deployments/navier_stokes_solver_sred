# src/common/elasticity.py

import logging

import numpy as np

from src.common.field_schema import FI


class ElasticManager:
    """
    SSoT for Numerical Stability. 
    Acts as the dynamic authority for dt, omega, and max_iter.
    """
    def __init__(self, config):
        self.config = config # SimulationConfig object
        self.logger = logging.getLogger("Elasticity")
        
        # Internal Elastic State
        self._dt = self.config.dt
        self._omega = self.config.ppe_omega
        self._max_iter = self.config.ppe_max_iter
        
        self.is_in_panic = False
        self.stable_streak = 0
        self.cooldown_limit = 5

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def validate_and_commit(self, state) -> bool:
        """Audits trial fields. Returns True if math is sane and committed."""
        audit_fields = [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR, FI.P_NEXT]
        data_slice = state.fields.data[:, audit_fields]
        
        # Access threshold via SSoT (Rule 4 & 5)
        # If the config doesn't have it, Rule 5 mandates we crash with an AttributeError
        limit = self.config.divergence_threshold 

        # Use a more direct check if memory is a concern at scale:
        if not np.isfinite(data_slice).all():
            return False

        # Use the absolute maximum to avoid creating a full boolean mask array
        # Slightly better: find max and min separately to avoid the abs() allocation
        if data_slice.max() > limit or data_slice.min() < -limit:
            return False

        # COMMIT: Star -> Foundation
        state.fields.data[:, [FI.VX, FI.VY, FI.VZ]] = state.fields.data[:, [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR]]
        state.fields.data[:, FI.P] = state.fields.data[:, FI.P_NEXT]
        return True

    def apply_panic_mode(self):
        self.is_in_panic = True
        self.stable_streak = 0
        self._dt *= 0.5
        self._omega = max(0.5, self._omega - 0.2)
        self._max_iter = 5000
        self.logger.warning(f"PANIC: dt reduced to {self._dt:.2e}")
        print("!!! ACTUAL LOG TRIGGERED !!!")

    def gradual_recovery(self):
        if not self.is_in_panic: return
        self.stable_streak += 1
        if self.stable_streak >= self.cooldown_limit:
            self._dt = min(self.config.dt, self._dt * 1.1)
            self._omega = min(self.config.ppe_omega, self._omega + 0.05)
            if self._dt == self.config.dt and self._omega == self.config.ppe_omega:
                self.is_in_panic = False
                self._max_iter = self.config.ppe_max_iter