"""Load and preprocess MyAnimeList data for the contextual bandits experiment."""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "cache"


def load_anime(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load anime_cleaned.csv and parse genre lists."""
    df = pd.read_csv(data_dir / "anime_cleaned.csv")
    df["genre_list"] = df["genre"].fillna("").apply(
        lambda g: [x.strip() for x in g.split(",") if x.strip()]
    )
    # Fill missing numeric columns
    for col in ["score", "popularity", "episodes", "duration_min", "aired_from_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["score"] = df["score"].fillna(df["score"].median())
    df["popularity"] = df["popularity"].fillna(df["popularity"].median())
    df["episodes"] = df["episodes"].fillna(1)
    df["duration_min"] = df["duration_min"].fillna(df["duration_min"].median())
    df["aired_from_year"] = df["aired_from_year"].fillna(df["aired_from_year"].median())
    return df


def load_users(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load users_cleaned.csv and fill missing stats."""
    df = pd.read_csv(data_dir / "users_cleaned.csv")
    df["stats_mean_score"] = pd.to_numeric(df["stats_mean_score"], errors="coerce")
    df["stats_mean_score"] = df["stats_mean_score"].fillna(df["stats_mean_score"].median())
    df["user_days_spent_watching"] = pd.to_numeric(df["user_days_spent_watching"], errors="coerce")
    df["user_days_spent_watching"] = df["user_days_spent_watching"].fillna(0)
    df["user_completed"] = pd.to_numeric(df["user_completed"], errors="coerce").fillna(0)
    # Total list size
    for col in ["user_watching", "user_onhold", "user_dropped", "user_plantowatch"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["total_list_size"] = (
        df["user_watching"] + df["user_completed"] + df["user_onhold"]
        + df["user_dropped"] + df["user_plantowatch"]
    )
    return df


def load_interactions(
    data_dir: Path = DATA_DIR,
    n_users: int = 5000,
    min_ratings: int = 20,
    min_anime_ratings: int = 50,
    seed: int = 42,
) -> tuple[pd.DataFrame, set[str], set[int]]:
    """Load animelists_cleaned.csv, filter, and subsample users.

    Returns:
        interactions: filtered DataFrame
        sampled_users: set of selected usernames
        candidate_anime_ids: set of anime_ids with enough ratings
    """
    cache_path = CACHE_DIR / f"interactions_n{n_users}_min{min_ratings}_seed{seed}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Loading interactions (this may take a minute on first run)...")
    df = pd.read_csv(
        data_dir / "animelists_cleaned.csv",
        usecols=["username", "anime_id", "my_score", "my_status"],
    )
    # Filter to scored, watching or completed
    df = df[(df["my_score"] > 0) & (df["my_status"].isin([1, 2]))]

    # Users with enough ratings
    user_counts = df["username"].value_counts()
    eligible_users = set(user_counts[user_counts >= min_ratings].index)
    rng = np.random.RandomState(seed)
    sampled_users = set(rng.choice(sorted(eligible_users), size=min(n_users, len(eligible_users)), replace=False))

    df = df[df["username"].isin(sampled_users)]

    # Anime with enough ratings among sampled users
    anime_counts = df["anime_id"].value_counts()
    candidate_anime_ids = set(anime_counts[anime_counts >= min_anime_ratings].index)
    df = df[df["anime_id"].isin(candidate_anime_ids)]

    # Re-filter users who still have enough ratings after anime filtering
    user_counts2 = df["username"].value_counts()
    sampled_users = set(user_counts2[user_counts2 >= min_ratings].index)
    df = df[df["username"].isin(sampled_users)]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((df, sampled_users, candidate_anime_ids), f)

    print(f"Loaded {len(df)} interactions, {len(sampled_users)} users, {len(candidate_anime_ids)} anime")
    return df, sampled_users, candidate_anime_ids


def build_rating_map(interactions: pd.DataFrame) -> dict[str, dict[int, int]]:
    """Build username -> {anime_id -> rating} lookup."""
    rating_map: dict[str, dict[int, int]] = {}
    for username, anime_id, score in zip(
        interactions["username"], interactions["anime_id"], interactions["my_score"]
    ):
        if username not in rating_map:
            rating_map[username] = {}
        rating_map[username][anime_id] = int(score)
    return rating_map


def load_all(
    n_users: int = 5000,
    min_ratings: int = 20,
    min_anime_ratings: int = 50,
    seed: int = 42,
) -> dict:
    """Load everything and return a data bundle."""
    anime_df = load_anime()
    users_df = load_users()
    interactions, sampled_users, candidate_anime_ids = load_interactions(
        n_users=n_users, min_ratings=min_ratings,
        min_anime_ratings=min_anime_ratings, seed=seed,
    )
    rating_map = build_rating_map(interactions)

    # Filter DataFrames to relevant subset
    anime_df = anime_df[anime_df["anime_id"].isin(candidate_anime_ids)].copy()
    users_df = users_df[users_df["username"].isin(sampled_users)].copy()

    return {
        "anime_df": anime_df,
        "users_df": users_df,
        "interactions": interactions,
        "rating_map": rating_map,
        "sampled_users": sorted(sampled_users),
        "candidate_anime_ids": sorted(candidate_anime_ids),
    }
