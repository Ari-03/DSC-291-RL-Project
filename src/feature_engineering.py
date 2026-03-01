"""Construct context vectors for the contextual bandits experiment."""

import numpy as np
import pandas as pd


# Canonical ordered lists for one-hot encoding
ALL_GENRES = [
    "Action", "Adventure", "Cars", "Comedy", "Dementia", "Demons", "Drama",
    "Ecchi", "Fantasy", "Game", "Harem", "Hentai", "Historical", "Horror",
    "Josei", "Kids", "Magic", "Martial Arts", "Mecha", "Military", "Music",
    "Mystery", "Parody", "Police", "Psychological", "Romance", "Samurai",
    "School", "Sci-Fi", "Seinen", "Shoujo", "Shoujo Ai", "Shounen",
    "Shounen Ai", "Slice of Life", "Space", "Sports", "Super Power",
    "Supernatural", "Thriller", "Vampire", "Yaoi", "Yuri",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(ALL_GENRES)}
NUM_GENRES = len(ALL_GENRES)

ALL_TYPES = ["Movie", "Music", "ONA", "OVA", "Special", "TV"]
TYPE_TO_IDX = {t: i for i, t in enumerate(ALL_TYPES)}
NUM_TYPES = len(ALL_TYPES)

# Group rare sources to keep dimensionality manageable
SOURCE_GROUPS = {
    "Manga": "Manga",
    "Light novel": "LightNovel",
    "Original": "Original",
    "Visual novel": "VisualNovel",
    "Game": "Game",
}
ALL_SOURCE_GROUPS = ["Manga", "LightNovel", "Original", "VisualNovel", "Game", "Other"]
SOURCE_TO_IDX = {s: i for i, s in enumerate(ALL_SOURCE_GROUPS)}
NUM_SOURCES = len(ALL_SOURCE_GROUPS)


class FeatureBuilder:
    """Precomputes and caches feature vectors for users and anime."""

    def __init__(self, anime_df: pd.DataFrame, users_df: pd.DataFrame, rating_map: dict):
        self.anime_df = anime_df.set_index("anime_id")
        self.users_df = users_df.set_index("username")
        self.rating_map = rating_map

        # Compute normalization stats from training data
        self._compute_normalization_stats()
        # Precompute static anime features
        self._anime_features: dict[int, np.ndarray] = {}
        self._anime_genre_vec: dict[int, np.ndarray] = {}
        for aid in self.anime_df.index:
            self._anime_features[aid] = self._build_anime_features(aid)
            self._anime_genre_vec[aid] = self._build_genre_vec(aid)

        # Precompute user features and genre profiles
        self._user_features: dict[str, np.ndarray] = {}
        self._user_genre_profile: dict[str, np.ndarray] = {}
        for uname in self.users_df.index:
            self._user_features[uname] = self._build_user_features(uname)
            self._user_genre_profile[uname] = self._build_user_genre_profile(uname)

        self._d = self._compute_dim()

    @property
    def dim(self) -> int:
        return self._d

    def _compute_normalization_stats(self):
        """Compute means and stds for normalization."""
        a = self.anime_df
        self.anime_score_mean = a["score"].mean()
        self.anime_score_std = a["score"].std() + 1e-8
        self.anime_pop_mean = np.log1p(a["popularity"]).mean()
        self.anime_pop_std = np.log1p(a["popularity"]).std() + 1e-8
        self.anime_ep_mean = np.log1p(a["episodes"]).mean()
        self.anime_ep_std = np.log1p(a["episodes"]).std() + 1e-8
        self.anime_dur_mean = a["duration_min"].mean()
        self.anime_dur_std = a["duration_min"].std() + 1e-8
        self.anime_year_mean = a["aired_from_year"].mean()
        self.anime_year_std = a["aired_from_year"].std() + 1e-8

        u = self.users_df
        self.user_score_mean = u["stats_mean_score"].mean()
        self.user_score_std = u["stats_mean_score"].std() + 1e-8
        self.user_comp_mean = np.log1p(u["user_completed"]).mean()
        self.user_comp_std = np.log1p(u["user_completed"]).std() + 1e-8
        self.user_days_mean = np.log1p(u["user_days_spent_watching"]).mean()
        self.user_days_std = np.log1p(u["user_days_spent_watching"]).std() + 1e-8
        self.user_list_mean = np.log1p(u["total_list_size"]).mean()
        self.user_list_std = np.log1p(u["total_list_size"]).std() + 1e-8

    def _build_user_features(self, username: str) -> np.ndarray:
        """Build static user feature block (~4 dims, normalized)."""
        row = self.users_df.loc[username]
        return np.array([
            (row["stats_mean_score"] - self.user_score_mean) / self.user_score_std,
            (np.log1p(row["user_completed"]) - self.user_comp_mean) / self.user_comp_std,
            (np.log1p(row["user_days_spent_watching"]) - self.user_days_mean) / self.user_days_std,
            (np.log1p(row["total_list_size"]) - self.user_list_mean) / self.user_list_std,
        ], dtype=np.float64)

    def _build_genre_vec(self, anime_id: int) -> np.ndarray:
        """Multi-hot genre vector for an anime."""
        vec = np.zeros(NUM_GENRES, dtype=np.float64)
        row = self.anime_df.loc[anime_id]
        for g in row["genre_list"]:
            if g in GENRE_TO_IDX:
                vec[GENRE_TO_IDX[g]] = 1.0
        return vec

    def _build_anime_features(self, anime_id: int) -> np.ndarray:
        """Build static anime feature block.

        Layout: [genre_multihot(43), score(1), log_pop(1), log_ep(1),
                 type_onehot(6), source_onehot(6), duration(1), year(1)]
        Total: 43 + 1 + 1 + 1 + 6 + 6 + 1 + 1 = 60
        """
        row = self.anime_df.loc[anime_id]

        # Genre multi-hot
        genre = self._build_genre_vec(anime_id)

        # Normalized continuous
        score = (row["score"] - self.anime_score_mean) / self.anime_score_std
        log_pop = (np.log1p(row["popularity"]) - self.anime_pop_mean) / self.anime_pop_std
        log_ep = (np.log1p(row["episodes"]) - self.anime_ep_mean) / self.anime_ep_std
        dur = (row["duration_min"] - self.anime_dur_mean) / self.anime_dur_std
        year = (row["aired_from_year"] - self.anime_year_mean) / self.anime_year_std

        # Type one-hot
        type_vec = np.zeros(NUM_TYPES, dtype=np.float64)
        t = row.get("type", None)
        if isinstance(t, str) and t in TYPE_TO_IDX:
            type_vec[TYPE_TO_IDX[t]] = 1.0

        # Source one-hot (grouped)
        source_vec = np.zeros(NUM_SOURCES, dtype=np.float64)
        s = row.get("source", None)
        if isinstance(s, str):
            group = SOURCE_GROUPS.get(s, "Other")
            source_vec[SOURCE_TO_IDX[group]] = 1.0

        return np.concatenate([
            genre,                          # 43
            [score, log_pop, log_ep],       # 3
            type_vec,                       # 6
            source_vec,                     # 6
            [dur, year],                    # 2
        ])  # total = 60

    def _build_user_genre_profile(self, username: str) -> np.ndarray:
        """Weighted average of genre vectors for anime the user has rated."""
        profile = np.zeros(NUM_GENRES, dtype=np.float64)
        ratings = self.rating_map.get(username, {})
        total_weight = 0.0
        for aid, score in ratings.items():
            if aid in self._anime_genre_vec:
                profile += score * self._anime_genre_vec[aid]
                total_weight += score
        if total_weight > 0:
            profile /= total_weight
        return profile

    def build_context(self, username: str, anime_id: int) -> np.ndarray:
        """Build full context vector x_{a,t} = phi(user, anime).

        Layout: [user_feats(4), anime_feats(60), interaction(2), bias(1)]
        Total: 4 + 60 + 2 + 1 = 67
        """
        user_feats = self._user_features[username]
        anime_feats = self._anime_features[anime_id]

        # Interaction features
        user_genre = self._user_genre_profile[username]
        anime_genre = self._anime_genre_vec[anime_id]

        # Genre overlap cosine similarity
        norm_u = np.linalg.norm(user_genre)
        norm_a = np.linalg.norm(anime_genre)
        if norm_u > 0 and norm_a > 0:
            genre_cos = np.dot(user_genre, anime_genre) / (norm_u * norm_a)
        else:
            genre_cos = 0.0

        # Score deviation: user_mean - anime_score (both raw, normalized)
        user_mean = self.users_df.loc[username, "stats_mean_score"]
        anime_score = self.anime_df.loc[anime_id, "score"]
        score_dev = (user_mean - anime_score) / self.anime_score_std

        interaction = np.array([genre_cos, score_dev], dtype=np.float64)

        return np.concatenate([user_feats, anime_feats, interaction, [1.0]])  # bias

    def build_context_batch(self, username: str, anime_ids: list[int]) -> np.ndarray:
        """Build context matrix for multiple anime for one user. Shape: (K, d)."""
        return np.vstack([self.build_context(username, aid) for aid in anime_ids])

    def _compute_dim(self) -> int:
        """Compute feature dimensionality from a sample."""
        uname = next(iter(self._user_features))
        aid = next(iter(self._anime_features))
        return len(self.build_context(uname, aid))
