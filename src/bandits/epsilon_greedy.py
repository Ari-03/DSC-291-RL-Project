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


class DecayingEpsilonGreedy(ContextualBandit):
    """Epsilon-greedy with decaying exploration: epsilon_t = epsilon_0 / sqrt(t+1)."""

    def __init__(self, d: int, epsilon_0: float = 1.0, lam: float = 1.0):
        super().__init__(d, lam)
        self.epsilon_0 = epsilon_0

    def select_arm(self, contexts: np.ndarray, rng: np.random.RandomState) -> int:
        K = contexts.shape[0]
        effective_eps = self.epsilon_0 / np.sqrt(self.t + 1)
        if rng.rand() < effective_eps:
            return rng.randint(K)
        predicted = contexts @ self.theta_hat
        return int(np.argmax(predicted))

    @property
    def name(self) -> str:
        return f"DecayEG(ε₀={self.epsilon_0})"
