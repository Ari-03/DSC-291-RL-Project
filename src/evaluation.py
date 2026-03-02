"""Experiment tracking and plotting utilities."""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class ExperimentTracker:
    """Log rewards and oracle rewards, compute cumulative reward and regret."""

    def __init__(self, algo_name: str, T: int):
        self.algo_name = algo_name
        self.T = T
        self.rewards = np.zeros(T)
        self.oracle_rewards = np.zeros(T)
        self.t = 0

    def log(self, reward: float, oracle_reward: float):
        self.rewards[self.t] = reward
        self.oracle_rewards[self.t] = oracle_reward
        self.t += 1

    @property
    def cumulative_reward(self) -> np.ndarray:
        return np.cumsum(self.rewards[:self.t])

    @property
    def cumulative_regret(self) -> np.ndarray:
        return np.cumsum(self.oracle_rewards[:self.t] - self.rewards[:self.t])

    def sliding_avg_reward(self, window: int = 500) -> np.ndarray:
        kernel = np.ones(window) / window
        return np.convolve(self.rewards[:self.t], kernel, mode="valid")


def save_results(results: dict, path: Path):
    """Save experiment results to pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_results(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- Plotting ----------

COLORS = {
    "Random": "#888888",
    "EpsGreedy": "#e74c3c",
    "LinUCB": "#2ecc71",
    "TS": "#3498db",
    "Popularity": "#f39c12",
    "SVD": "#8e44ad",
    "UserCF": "#1abc9c",
}

def _get_color(name: str) -> str:
    for key, color in COLORS.items():
        if key in name:
            return color
    return "#9b59b6"


def plot_cumulative_reward(
    all_results: dict[str, list[ExperimentTracker]],
    title: str = "Cumulative Reward",
    save_path: Path | None = None,
):
    """Plot cumulative reward with mean +/- SE across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, trackers in all_results.items():
        curves = np.array([tr.cumulative_reward for tr in trackers])
        mean = curves.mean(axis=0)
        se = curves.std(axis=0) / np.sqrt(len(trackers))
        T = len(mean)
        x = np.arange(1, T + 1)
        color = _get_color(algo_name)
        ax.plot(x, mean, label=algo_name, color=color)
        ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_cumulative_regret(
    all_results: dict[str, list[ExperimentTracker]],
    title: str = "Cumulative Regret",
    save_path: Path | None = None,
):
    """Plot cumulative regret with mean +/- SE and sqrt(T) reference."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, trackers in all_results.items():
        curves = np.array([tr.cumulative_regret for tr in trackers])
        mean = curves.mean(axis=0)
        se = curves.std(axis=0) / np.sqrt(len(trackers))
        T = len(mean)
        x = np.arange(1, T + 1)
        color = _get_color(algo_name)
        ax.plot(x, mean, label=algo_name, color=color)
        ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)

    # sqrt(T) reference
    T = max(len(next(iter(v)).cumulative_regret) for v in all_results.values())
    x_ref = np.arange(1, T + 1)
    # Scale to roughly match magnitude
    max_regret = max(
        np.array([tr.cumulative_regret for tr in trs]).mean(axis=0)[-1]
        for trs in all_results.values()
    )
    scale = max_regret / np.sqrt(T)
    ax.plot(x_ref, scale * np.sqrt(x_ref), "--", color="gray", alpha=0.5, label=r"$O(\sqrt{T})$ ref")

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_sliding_reward(
    all_results: dict[str, list[ExperimentTracker]],
    window: int = 500,
    title: str = "Sliding-Window Average Reward",
    save_path: Path | None = None,
):
    """Plot sliding-window average reward with mean +/- SE."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, trackers in all_results.items():
        curves = np.array([tr.sliding_avg_reward(window) for tr in trackers])
        mean = curves.mean(axis=0)
        se = curves.std(axis=0) / np.sqrt(len(trackers))
        x = np.arange(len(mean))
        color = _get_color(algo_name)
        ax.plot(x, mean, label=algo_name, color=color)
        ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)

    ax.set_xlabel("Round")
    ax.set_ylabel(f"Avg Reward (window={window})")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
