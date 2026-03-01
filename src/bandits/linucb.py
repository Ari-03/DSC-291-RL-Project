"""LinUCB contextual bandit (Li et al., 2010)."""

import numpy as np

from .base import ContextualBandit


class LinUCB(ContextualBandit):
    """LinUCB: select arm with highest UCB = x^T theta_hat + alpha * sqrt(x^T A_inv x)."""

    def __init__(self, d: int, alpha: float = 1.0, lam: float = 1.0):
        super().__init__(d, lam)
        self.alpha = alpha

    def select_arm(self, contexts: np.ndarray, rng: np.random.RandomState) -> int:
        # contexts: (K, d)
        predicted = contexts @ self.theta_hat                        # (K,)
        # Uncertainty: sqrt(x^T A_inv x) for each arm
        # Efficient batch: (K, d) @ (d, d) -> (K, d), then row-wise dot with contexts
        temp = contexts @ self.A_inv                                 # (K, d)
        uncertainty = np.sqrt(np.sum(temp * contexts, axis=1))       # (K,)
        ucb = predicted + self.alpha * uncertainty
        return int(np.argmax(ucb))

    @property
    def name(self) -> str:
        return f"LinUCB(α={self.alpha})"
