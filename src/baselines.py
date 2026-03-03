"""Offline baselines for comparison: SVD-CF, UserCF.

These baselines are trained on the training split (70% of each user's
ratings) and evaluated on the held-out test split. This is a fair
comparison — CF methods must generalize, not memorize.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


class OfflineBaseline(ABC):
    """Base class for offline baselines that use the full rating matrix."""

    @abstractmethod
    def select_arm(self, username: str, anime_ids: list[int], rng: np.random.RandomState) -> int:
        """Select an arm given a username and candidate anime IDs."""

    def update(self, x: np.ndarray, reward: float):
        pass

    def reset(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str: ...


class SVDBaseline(OfflineBaseline):
    """Matrix factorization via truncated SVD on the centered rating matrix."""

    def __init__(self, rating_map: dict[str, dict[int, int]], k: int = 100):
        users = sorted(rating_map.keys())
        # Collect all anime IDs
        all_anime: set[int] = set()
        for ur in rating_map.values():
            all_anime.update(ur.keys())
        anime_list = sorted(all_anime)

        self._user_idx = {u: i for i, u in enumerate(users)}
        self._anime_idx = {a: i for i, a in enumerate(anime_list)}

        n_users = len(users)
        n_anime = len(anime_list)

        # Build sparse matrix
        rows, cols, vals = [], [], []
        for u, ur in rating_map.items():
            ui = self._user_idx[u]
            for aid, rating in ur.items():
                ai = self._anime_idx[aid]
                rows.append(ui)
                cols.append(ai)
                vals.append(float(rating))

        R = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_anime))

        # Compute per-user mean (only over rated items)
        row_sums = np.array(R.sum(axis=1)).ravel()
        row_counts = np.array((R != 0).sum(axis=1)).ravel()
        row_counts[row_counts == 0] = 1
        self._user_means = row_sums / row_counts

        # Center the matrix (subtract user mean from nonzero entries)
        R_centered = R.copy().astype(np.float64)
        for i in range(n_users):
            start, end = R_centered.indptr[i], R_centered.indptr[i + 1]
            R_centered.data[start:end] -= self._user_means[i]

        # Truncated SVD
        k = min(k, min(n_users, n_anime) - 1)
        U, sigma, Vt = svds(R_centered, k=k)
        self._U_sigma = U * sigma  # (n_users, k)
        self._Vt = Vt              # (k, n_anime)

    def select_arm(self, username: str, anime_ids: list[int], rng: np.random.RandomState) -> int:
        ui = self._user_idx.get(username)
        if ui is None:
            return rng.randint(len(anime_ids))
        user_mean = self._user_means[ui]
        u_vec = self._U_sigma[ui]  # (k,)

        best_idx, best_score = 0, -np.inf
        for i, aid in enumerate(anime_ids):
            ai = self._anime_idx.get(aid)
            if ai is None:
                score = user_mean
            else:
                score = user_mean + u_vec @ self._Vt[:, ai]
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    @property
    def name(self) -> str:
        return "SVD-CF"


class UserCFBaseline(OfflineBaseline):
    """User-based collaborative filtering with cosine similarity."""

    def __init__(
        self,
        rating_map: dict[str, dict[int, int]],
        k_neighbors: int = 50,
        predict_for_anime: dict[str, set[int]] | None = None,
    ):
        users = sorted(rating_map.keys())
        all_anime: set[int] = set()
        for ur in rating_map.values():
            all_anime.update(ur.keys())
        anime_list = sorted(all_anime)

        user_idx = {u: i for i, u in enumerate(users)}
        anime_idx = {a: i for i, a in enumerate(anime_list)}

        n_users = len(users)
        n_anime = len(anime_list)

        # Build sparse matrix
        rows, cols, vals = [], [], []
        for u, ur in rating_map.items():
            ui = user_idx[u]
            for aid, rating in ur.items():
                ai = anime_idx[aid]
                rows.append(ui)
                cols.append(ai)
                vals.append(float(rating))

        R = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_anime))

        # Cosine similarity: normalize rows, then S = R_norm @ R_norm.T
        row_norms = sp.linalg.norm(R, axis=1)
        row_norms[row_norms == 0] = 1.0
        R_norm = R.multiply(1.0 / row_norms[:, np.newaxis])
        # S is dense (n_users x n_users) — 5000x5000 = 25M floats ~ 200MB, manageable
        S = (R_norm @ R_norm.T).toarray()

        # Zero out self-similarity
        np.fill_diagonal(S, 0.0)

        # Precompute predictions for all (user, anime) pairs in rating_map
        # plus any extra test anime from predict_for_anime
        # For each user, only keep top-k neighbors for efficiency
        self._pred: dict[str, dict[int, float]] = {}
        for u, ur in rating_map.items():
            ui = user_idx[u]
            sim_row = S[ui]
            # Top-k neighbors
            if k_neighbors < n_users:
                top_k = np.argpartition(sim_row, -k_neighbors)[-k_neighbors:]
            else:
                top_k = np.arange(n_users)

            # Predict for train anime + any extra test anime
            anime_to_predict = set(ur.keys())
            if predict_for_anime:
                anime_to_predict |= predict_for_anime.get(u, set())

            user_preds: dict[int, float] = {}
            for aid in anime_to_predict:
                ai = anime_idx.get(aid)
                if ai is None:
                    continue  # anime not in training matrix, skip
                # Get neighbors who rated this anime
                col = R[:, ai].toarray().ravel()
                mask = (col[top_k] != 0)
                neighbor_indices = top_k[mask]
                if len(neighbor_indices) == 0:
                    user_preds[aid] = 5.0  # fallback to global mean-ish
                    continue
                sims = sim_row[neighbor_indices]
                ratings = col[neighbor_indices]
                denom = np.abs(sims).sum()
                if denom < 1e-10:
                    user_preds[aid] = 5.0
                else:
                    user_preds[aid] = (sims * ratings).sum() / denom
            self._pred[u] = user_preds

    def select_arm(self, username: str, anime_ids: list[int], rng: np.random.RandomState) -> int:
        user_preds = self._pred.get(username, {})
        scores = [user_preds.get(aid, 5.0) for aid in anime_ids]
        return int(np.argmax(scores))

    @property
    def name(self) -> str:
        return "UserCF"
