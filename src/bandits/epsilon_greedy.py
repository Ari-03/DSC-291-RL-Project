"""Epsilon-greedy contextual bandit."""

import numpy as np

from .base import ContextualBandit


class EpsilonGreedy(ContextualBandit):
    """Epsilon-greedy: explore uniformly with probability epsilon, else exploit."""

    def __init__(self, d: int, epsilon: float = 0.1, lam: float = 1.0):
        super().__init__(d, lam)
        self.epsilon = epsilon

    def select_arm(self, contexts: np.ndarray, rng: np.random.RandomState) -> int:
        K = contexts.shape[0]
        if rng.rand() < self.epsilon:
            return rng.randint(K)
        predicted = contexts @ self.theta_hat  # (K,)
        return int(np.argmax(predicted))

    @property
    def name(self) -> str:
        return f"EpsGreedy(ε={self.epsilon})"
