"""Abstract base class for contextual bandits with online ridge regression."""

from abc import ABC, abstractmethod

import numpy as np


class ContextualBandit(ABC):
    """Base class with shared ridge regression sufficient statistics.

    Maintains b = X^T y and A_inv via Sherman-Morrison rank-1 updates
    for O(d^2) per step.
    """

    def __init__(self, d: int, lam: float = 1.0):
        self.d = d
        self.lam = lam
        self.reset()

    def reset(self):
        """Reset to initial state for multi-seed experiments."""
        self.b = np.zeros(self.d)
        self.A_inv = (1.0 / self.lam) * np.eye(self.d)
        self.theta_hat = np.zeros(self.d)
        self.t = 0

    @abstractmethod
    def select_arm(self, contexts: np.ndarray, rng: np.random.RandomState) -> int:
        """Select an arm given context matrix (K, d). Returns arm index."""

    def update(self, x: np.ndarray, reward: float):
        """Update sufficient statistics with new observation.

        Uses Sherman-Morrison formula: (A + xx^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        """
        self.t += 1
        # Sherman-Morrison update for A_inv
        Ainv_x = self.A_inv @ x                        # (d,)
        denom = 1.0 + x @ Ainv_x                       # scalar
        self.A_inv -= np.outer(Ainv_x, Ainv_x) / denom # (d, d)

        # Update b
        self.b += reward * x

        # Recompute theta_hat
        self.theta_hat = self.A_inv @ self.b

    def warm_start(self, theta: np.ndarray, A_inv: np.ndarray, b: np.ndarray):
        """Initialize from offline ridge regression fit."""
        self.theta_hat = theta.copy()
        self.A_inv = A_inv.copy()
        self.b = b.copy()

    @property
    def name(self) -> str:
        return self.__class__.__name__
