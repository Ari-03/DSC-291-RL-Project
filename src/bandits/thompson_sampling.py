"""Thompson Sampling contextual bandit with linear payoffs (Agrawal & Goyal, 2013)."""

import numpy as np
from scipy.linalg import cholesky

from .base import ContextualBandit


class ThompsonSampling(ContextualBandit):
    """Linear Thompson Sampling: sample theta from posterior and pick best arm."""

    def __init__(self, d: int, v: float = 0.1, lam: float = 1.0):
        super().__init__(d, lam)
        self.v = v

    def select_arm(self, contexts: np.ndarray, rng: np.random.RandomState) -> int:
        # Sample theta_tilde ~ N(theta_hat, v^2 * A_inv) via Cholesky
        try:
            L = cholesky(self.v**2 * self.A_inv, lower=True)  # (d, d)
        except np.linalg.LinAlgError:
            # Fallback if A_inv is not PD (numerical issues) -- scale-adaptive jitter
            jittered = self.v**2 * self.A_inv + 1e-6 * np.trace(self.A_inv) / self.d * np.eye(self.d)
            L = cholesky(jittered, lower=True)

        z = rng.randn(self.d)
        theta_tilde = self.theta_hat + L @ z  # (d,)

        predicted = contexts @ theta_tilde  # (K,)
        return int(np.argmax(predicted))

    @property
    def name(self) -> str:
        return f"TS(v={self.v})"
