"""Main experiment runner for contextual bandits anime recommendation."""

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all
from src.feature_engineering import FeatureBuilder
from src.environment import AnimeRecommendationEnv
from src.bandits import EpsilonGreedy, LinUCB, ThompsonSampling
from src.bandits.base import ContextualBandit
from src.baselines import OfflineBaseline, PopularityBaseline, SVDBaseline, UserCFBaseline
from src.evaluation import ExperimentTracker, save_results


# ---- Configuration ----
CONFIG = {
    "T": 50_000,           # rounds per experiment
    "K": 20,               # arms per round
    "seeds": [0, 1, 2, 3, 4],
    "n_users": 5000,
    "min_ratings": 20,
    "min_anime_ratings": 50,
    "lam": 1.0,
    "binary_threshold": 7,
    "reward_type": "continuous",  # "continuous" or "binary"
}


class RandomPolicy(ContextualBandit):
    """Baseline: pick a random arm each round."""

    def __init__(self, d: int, lam: float = 1.0):
        # Don't need ridge regression, but keep the interface
        self.d = d
        self.lam = lam
        self.t = 0

    def reset(self):
        self.t = 0

    def select_arm(self, contexts: np.ndarray, rng: np.random.RandomState) -> int:
        return rng.randint(contexts.shape[0])

    def update(self, x: np.ndarray, reward: float):
        self.t += 1

    @property
    def name(self) -> str:
        return "Random"


def build_algorithms(d: int, lam: float) -> list[ContextualBandit]:
    """Instantiate all algorithms to evaluate."""
    return [
        RandomPolicy(d, lam),
        EpsilonGreedy(d, epsilon=0.05, lam=lam),
        EpsilonGreedy(d, epsilon=0.1, lam=lam),
        LinUCB(d, alpha=0.5, lam=lam),
        LinUCB(d, alpha=1.0, lam=lam),
        ThompsonSampling(d, v=0.1, lam=lam),
        ThompsonSampling(d, v=0.5, lam=lam),
    ]


def build_baselines(rating_map: dict[str, dict[int, int]]) -> list[OfflineBaseline]:
    """Instantiate offline CF baselines (these see the full rating matrix)."""
    return [
        PopularityBaseline(rating_map),
        SVDBaseline(rating_map),
        UserCFBaseline(rating_map),
    ]


def main():
    print("=" * 60)
    print("Contextual Bandits for Anime Recommendation")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    data = load_all(
        n_users=CONFIG["n_users"],
        min_ratings=CONFIG["min_ratings"],
        min_anime_ratings=CONFIG["min_anime_ratings"],
        seed=42,
    )
    print(f"  Users: {len(data['sampled_users'])}, Anime: {len(data['candidate_anime_ids'])}")
    print(f"  Interactions: {len(data['interactions'])}")

    # Build features
    print("\n[2/4] Building feature vectors...")
    fb = FeatureBuilder(data["anime_df"], data["users_df"], data["rating_map"])
    d = fb.dim
    print(f"  Feature dimension: {d}")

    # Create environment
    print("\n[3/4] Setting up environment...")
    env = AnimeRecommendationEnv(
        feature_builder=fb,
        rating_map=data["rating_map"],
        users=data["sampled_users"],
        K=CONFIG["K"],
        binary_threshold=CONFIG["binary_threshold"],
        reward_type=CONFIG["reward_type"],
    )
    print(f"  Valid users for K={CONFIG['K']}: {len(env.valid_users)}")

    # Run experiments
    print(f"\n[4/4] Running experiments (T={CONFIG['T']}, {len(CONFIG['seeds'])} seeds)...")
    algorithms = build_algorithms(d, CONFIG["lam"])
    print("  Building offline baselines (may take a minute)...")
    baselines = build_baselines(data["rating_map"])
    all_policies: list[ContextualBandit | OfflineBaseline] = algorithms + baselines
    algo_names = [a.name for a in all_policies]
    print(f"  Policies: {algo_names}")

    # results: algo_name -> list of ExperimentTracker (one per seed)
    results: dict[str, list[ExperimentTracker]] = {a.name: [] for a in all_policies}
    algo_idx = {a.name: i for i, a in enumerate(all_policies)}

    for seed in CONFIG["seeds"]:
        print(f"\n  Seed {seed}: generating sequence...")
        sequence = env.generate_sequence(CONFIG["T"], seed=seed)

        for algo in all_policies:
            start = time.time()
            # Use per-algorithm RNG with deterministic index (hash() is session-random)
            algo_rng_seed = seed * 1000 + algo_idx[algo.name]
            # Patch the select_arm call to use a proper RNG
            tracker = _run_with_rng(algo, sequence, env, CONFIG["T"], algo_rng_seed)
            elapsed = time.time() - start
            results[algo.name].append(tracker)
            final_reward = tracker.cumulative_reward[-1]
            final_regret = tracker.cumulative_regret[-1]
            print(f"    {algo.name:25s} | reward={final_reward:.1f} | regret={final_regret:.1f} | {elapsed:.1f}s")

    # Save results
    results_path = PROJECT_ROOT / "results" / f"results_{CONFIG['reward_type']}.pkl"
    save_results(results, results_path)
    print(f"\nResults saved to {results_path}")

    # Also save config
    config_path = PROJECT_ROOT / "results" / "config.pkl"
    save_results(CONFIG, config_path)

    # Quick summary
    print("\n" + "=" * 60)
    print("SUMMARY (mean over seeds)")
    print("=" * 60)
    for name, trackers in results.items():
        rewards = [tr.cumulative_reward[-1] for tr in trackers]
        regrets = [tr.cumulative_regret[-1] for tr in trackers]
        print(f"  {name:25s} | reward={np.mean(rewards):.1f}±{np.std(rewards):.1f} | regret={np.mean(regrets):.1f}±{np.std(regrets):.1f}")


def _run_with_rng(
    algo: ContextualBandit | OfflineBaseline,
    sequence: list,
    env: AnimeRecommendationEnv,
    T: int,
    rng_seed: int,
) -> ExperimentTracker:
    """Run one algorithm with a dedicated RNG."""
    tracker = ExperimentTracker(algo.name, T)
    algo.reset()
    rng = np.random.RandomState(rng_seed)
    is_offline = isinstance(algo, OfflineBaseline)

    for t in tqdm(range(T), desc=f"    {algo.name:25s}", leave=False, ncols=80):
        username, anime_ids, contexts, oracle = sequence[t]
        if is_offline:
            arm_idx = algo.select_arm(username, anime_ids, rng)
        else:
            arm_idx = algo.select_arm(contexts, rng)
        chosen_anime = anime_ids[arm_idx]
        reward = env.get_reward(username, chosen_anime)
        if not is_offline:
            algo.update(contexts[arm_idx], reward)
        tracker.log(reward, oracle)

    return tracker


if __name__ == "__main__":
    main()
