"""Semi-synthetic offline simulator for anime recommendation."""

import numpy as np

from .feature_engineering import FeatureBuilder


class AnimeRecommendationEnv:
    """At each round, sample a user and present K candidate anime that the user
    has actually rated. Return the real rating as reward.

    This is a "committed" strategy: we only offer anime for which we have
    ground-truth ratings, enabling exact reward and regret computation.
    """

    def __init__(
        self,
        feature_builder: FeatureBuilder,
        rating_map: dict[str, dict[int, int]],
        users: list[str],
        K: int = 20,
        binary_threshold: int = 7,
        reward_type: str = "continuous",
    ):
        self.fb = feature_builder
        self.rating_map = rating_map
        self.users = users
        self.K = K
        self.binary_threshold = binary_threshold
        self.reward_type = reward_type

        # Precompute per-user anime lists for fast sampling
        self._user_anime: dict[str, list[int]] = {}
        for u in users:
            rated = rating_map.get(u, {})
            # Only include anime for which we have precomputed features
            valid = [aid for aid in rated if self.fb.has_anime(aid)]
            if len(valid) >= K:
                self._user_anime[u] = valid
        self.valid_users = sorted(self._user_anime.keys())

    def _reward(self, rating: int) -> float:
        if self.reward_type == "binary":
            return 1.0 if rating >= self.binary_threshold else 0.0
        return rating / 10.0

    def step(self, rng: np.random.RandomState) -> tuple[str, list[int], np.ndarray]:
        """Sample a round: user, K candidate anime, context matrix.

        Returns:
            username, anime_ids (list of K), contexts (K, d)
        """
        user = self.valid_users[rng.randint(len(self.valid_users))]
        all_anime = self._user_anime[user]
        indices = rng.choice(len(all_anime), size=self.K, replace=False)
        anime_ids = [all_anime[i] for i in indices]
        contexts = self.fb.build_context_batch(user, anime_ids)
        return user, anime_ids, contexts

    def get_reward(self, username: str, anime_id: int) -> float:
        """Get reward for a (user, anime) pair."""
        rating = self.rating_map[username][anime_id]
        return self._reward(rating)

    def oracle_reward(self, username: str, anime_ids: list[int]) -> float:
        """Best possible reward among the K candidates."""
        return max(self._reward(self.rating_map[username][aid]) for aid in anime_ids)

    def generate_sequence(self, T: int, seed: int) -> list[tuple[str, list[int], np.ndarray, float]]:
        """Pre-generate T rounds for fair comparison across algorithms.

        Each tuple contains (username, anime_ids, contexts, oracle_reward).
        Oracle is pre-computed once here instead of per-algorithm.
        """
        rng = np.random.RandomState(seed)
        seq = []
        for _ in range(T):
            user, anime_ids, contexts = self.step(rng)
            oracle = self.oracle_reward(user, anime_ids)
            seq.append((user, anime_ids, contexts, oracle))
        return seq
