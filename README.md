# Contextual Bandits for Anime Recommendation

**DSC 291: Reinforcement Learning -- Course Project**
**Author:** Aritra

## Motivation

Recommendation systems face a fundamental *explore-exploit dilemma*: should the system recommend items it already knows the user likes (exploit), or try unfamiliar items to discover new preferences (explore)? Classical collaborative filtering methods greedily exploit past behavior, often trapping users in filter bubbles.

This project frames anime recommendation as a **contextual bandit** problem and compares principled exploration strategies on real user-preference data from [MyAnimeList](https://myanimelist.net/).

## Algorithms

All algorithms share an online ridge regression backbone with Sherman-Morrison rank-1 updates (O(d^2) per step):

| Algorithm | Strategy | Key Parameter |
|-----------|----------|---------------|
| **Random** | Uniform random arm selection (baseline) | -- |
| **Epsilon-Greedy** | Exploit with probability 1-epsilon, explore uniformly otherwise | epsilon in {0.05, 0.1} |
| **LinUCB** ([Li et al., 2010](https://arxiv.org/abs/1003.0146)) | Upper confidence bound on linear payoff | alpha in {0.5, 1.0} |
| **Thompson Sampling** ([Agrawal & Goyal, 2013](https://arxiv.org/abs/1209.3352)) | Posterior sampling via Cholesky decomposition | v in {0.1, 0.5} |

## Feature Engineering

Each (user, anime) pair is encoded as a **67-dimensional context vector**:

| Block | Dimensions | Description |
|-------|-----------|-------------|
| User features | 4 | Mean score, completed count, days watched, list size (z-scored) |
| Anime features | 60 | Genre multi-hot (43), score/popularity/episodes (3), type one-hot (6), source one-hot (6), duration + year (2) |
| Interaction features | 2 | User-anime genre cosine similarity, score deviation |
| Bias | 1 | Constant 1 |

All continuous features are z-score normalized. Log transforms are applied to skewed features (popularity, episodes, watch time, list size).

## Semi-Synthetic Environment

The environment uses real MyAnimeList ratings to create a faithful offline simulation:

1. **Data loading** -- Sample 5,000 users with at least 20 ratings each; keep anime with at least 50 ratings among sampled users.
2. **Round generation** -- Each round: sample a random user, draw K=20 anime that the user has actually rated, and build context vectors.
3. **Reward** -- The user's real rating serves as the reward (continuous: rating/10, or binary: 1 if rating >= 7).
4. **Oracle** -- The best achievable reward among the K candidates, enabling exact regret computation.
5. **Reproducibility** -- Pre-generate the full interaction sequence per seed so all algorithms see identical rounds.

Experiments run for T=50,000 rounds across 5 seeds.

## Project Structure

```
.
├── data/                        # MyAnimeList datasets (raw + cleaned)
│   ├── AnimeList.csv
│   ├── UserAnimeList.csv
│   ├── UserList.csv
│   ├── anime_cleaned.csv
│   ├── anime_filtered.csv
│   ├── animelists_cleaned.csv
│   ├── animelists_filtered.csv
│   ├── users_cleaned.csv
│   └── users_filtered.csv
├── src/                         # Core library
│   ├── bandits/
│   │   ├── base.py              # Abstract ContextualBandit with ridge regression
│   │   ├── epsilon_greedy.py    # Epsilon-Greedy
│   │   ├── linucb.py            # LinUCB
│   │   └── thompson_sampling.py # Linear Thompson Sampling
│   ├── data_loader.py           # Data loading, filtering, caching
│   ├── environment.py           # Semi-synthetic recommendation environment
│   ├── evaluation.py            # ExperimentTracker, plotting utilities
│   └── feature_engineering.py   # 67-dim context vector construction
├── scripts/
│   └── run_experiment.py        # Main experiment runner
├── notebooks/                   # Analysis notebooks
├── presentation/                # Marimo presentation slides
├── results/                     # Saved experiment results (pickle)
├── pyproject.toml
└── README.md
```

## Setup

**Requirements:** Python >= 3.12, [uv](https://docs.astral.sh/uv/)

```bash
# Clone and install
git clone <repo-url>
cd Project
uv sync
```

## Running Experiments

```bash
uv run python scripts/run_experiment.py
```

This will:
1. Load and preprocess MyAnimeList data (cached after first run)
2. Build 67-dim feature vectors for all user-anime pairs
3. Run all 7 algorithm variants across 5 seeds (T=50,000 rounds each)
4. Print a summary table and save results to `results/`

## Results

Experiments evaluate cumulative reward, cumulative regret, and sliding-window average reward (window=500). Results are averaged over 5 seeds with standard error bands.

Key findings:
- **LinUCB** and **Thompson Sampling** significantly outperform the random baseline and epsilon-greedy in cumulative reward.
- Principled exploration (UCB, posterior sampling) yields sublinear regret growth, consistent with theoretical O(sqrt(T)) bounds.
- Hyperparameter sensitivity: smaller exploration parameters (alpha=0.5, v=0.1) tend to perform better on this dataset, suggesting the linear model captures enough signal that aggressive exploration is unnecessary.
