"""Main experiment runner for contextual bandits anime recommendation."""

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all, split_rating_map, compute_user_means
from src.feature_engineering import FeatureBuilder
from src.environment import AnimeRecommendationEnv
from src.bandits import EpsilonGreedy, DecayingEpsilonGreedy, LinUCB, ThompsonSampling
from src.bandits.base import ContextualBandit
from src.baselines import OfflineBaseline, SVDBaseline, UserCFBaseline
from src.evaluation import ExperimentTracker, save_results


# ---- Configuration ----
CONFIG = {
    "T": 50_000,           # rounds per experiment
    "K": 50,               # arms per round
    "seeds": [0, 1, 2, 3, 4],
    "n_users": 5000,
    "min_ratings": 20,
    "min_anime_ratings": 50,
    "lam": 0.1,
    "warm_start_n": 10000,  # samples for warm-start ridge regression
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
        DecayingEpsilonGreedy(d, epsilon_0=1.0, lam=lam),
        LinUCB(d, alpha=0.1, lam=lam),
        LinUCB(d, alpha=0.5, lam=lam),
        LinUCB(d, alpha=1.0, lam=lam),
        ThompsonSampling(d, v=0.1, lam=lam),
        ThompsonSampling(d, v=0.5, lam=lam),
    ]


def build_baselines(
    rating_map_train: dict[str, dict[int, int]],
    rating_map_test: dict[str, dict[int, int]],
) -> list[OfflineBaseline]:
    """Instantiate offline CF baselines (trained on train split only)."""
    # Extract test anime per user for UserCF precomputation
    predict_for_anime: dict[str, set[int]] = {}
    for u, test_ratings in rating_map_test.items():
        if test_ratings:
            predict_for_anime[u] = set(test_ratings.keys())
    return [
        SVDBaseline(rating_map_train),
        UserCFBaseline(rating_map_train, predict_for_anime=predict_for_anime),
    ]


def main():
    print("=" * 60)
    print("Contextual Bandits for Anime Recommendation")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    data = load_all(
        n_users=CONFIG["n_users"],
        min_ratings=CONFIG["min_ratings"],
        min_anime_ratings=CONFIG["min_anime_ratings"],
        seed=42,
    )
    print(f"  Users: {len(data['sampled_users'])}, Anime: {len(data['candidate_anime_ids'])}")
    print(f"  Interactions: {len(data['interactions'])}")

    # Train/test split (per user, 70/30)
    print("\n  Splitting ratings 70/30 per user...")
    rating_map_train, rating_map_test = split_rating_map(data["rating_map"], test_frac=0.3, seed=42)
    user_means = compute_user_means(rating_map_train)
    _train_total = sum(len(r) for r in rating_map_train.values())
    _test_total = sum(len(r) for r in rating_map_test.values())
    print(f"  Train ratings: {_train_total}, Test ratings: {_test_total}")

    # Build features (using training data only)
    print("\n[2/5] Building feature vectors...")
    fb = FeatureBuilder(data["anime_df"], data["users_df"], rating_map_train)
    d = fb.dim
    print(f"  Feature dimension: {d}")

    # Create environment (using test data for rewards, centered by train means)
    print("\n[3/5] Setting up environment...")
    env = AnimeRecommendationEnv(
        feature_builder=fb,
        rating_map=rating_map_test,
        users=data["sampled_users"],
        K=CONFIG["K"],
        binary_threshold=CONFIG["binary_threshold"],
        reward_type=CONFIG["reward_type"],
        user_means=user_means,
    )
    print(f"  Valid users for K={CONFIG['K']}: {len(env.valid_users)}")

    # Warm-start: fit offline ridge regression on training data (with centering)
    print("\n[4/5] Computing warm-start from offline data...")
    warm_theta, warm_A_inv, warm_b = _compute_warm_start(
        fb, rating_map_train, CONFIG["lam"], CONFIG["warm_start_n"],
        reward_type=CONFIG["reward_type"],
        user_means=user_means,
    )
    print(f"  Warm-start fitted on {CONFIG['warm_start_n']} samples")

    # Run experiments
    print(f"\n[5/5] Running experiments (T={CONFIG['T']}, {len(CONFIG['seeds'])} seeds)...")
    algorithms = build_algorithms(d, CONFIG["lam"])
    print("  Building offline baselines (may take a minute)...")
    baselines = build_baselines(rating_map_train, rating_map_test)
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
            tracker = _run_with_rng(
                algo, sequence, env, CONFIG["T"], algo_rng_seed,
                warm_start=(warm_theta, warm_A_inv, warm_b),
            )
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


def _compute_warm_start(
    fb: FeatureBuilder,
    rating_map: dict,
    lam: float,
    n_samples: int,
    reward_type: str = "continuous",
    user_means: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ridge regression on a sample of (user, anime, rating) triples."""
    rng = np.random.RandomState(42)
    d = fb.dim

    # Collect all (user, anime) pairs
    pairs = []
    for u, ratings in rating_map.items():
        for aid, rating in ratings.items():
            if fb.has_anime(aid) and u in fb._user_features:
                pairs.append((u, aid, rating))

    # Sample
    indices = rng.choice(len(pairs), size=min(n_samples, len(pairs)), replace=False)

    X = np.zeros((len(indices), d))
    y = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        u, aid, rating = pairs[idx]
        X[i] = fb.build_context(u, aid)
        if reward_type == "binary":
            y[i] = 1.0 if rating >= 7 else 0.0
        elif user_means is not None:
            y[i] = (rating - user_means.get(u, 5.0)) / 10.0
        else:
            y[i] = rating / 10.0

    # Ridge regression: theta = (X^T X + lam I)^{-1} X^T y
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    A_inv = np.linalg.inv(A)
    theta = A_inv @ b

    return theta, A_inv, b


def _run_with_rng(
    algo: ContextualBandit | OfflineBaseline,
    sequence: list,
    env: AnimeRecommendationEnv,
    T: int,
    rng_seed: int,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> ExperimentTracker:
    """Run one algorithm with a dedicated RNG."""
    tracker = ExperimentTracker(algo.name, T)
    algo.reset()
    is_offline = isinstance(algo, OfflineBaseline)

    # Apply warm-start to online bandits (except Random)
    if warm_start is not None and not is_offline and hasattr(algo, 'warm_start') and algo.name != "Random":
        theta, A_inv, b = warm_start
        algo.warm_start(theta, A_inv, b)

    rng = np.random.RandomState(rng_seed)

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
