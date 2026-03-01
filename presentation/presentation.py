import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import pickle
    import sys
    import numpy as np
    from pathlib import Path
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = _THIS_DIR.parent
    RESULTS_PATH = PROJECT_ROOT / "results" / "results_continuous.pkl"
    # Ensure src is importable (needed for unpickling ExperimentTracker)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    COLORS = {
        "Random": "#888888",
        "EpsGreedy(ε=0.05)": "#e74c3c",
        "EpsGreedy(ε=0.1)": "#c0392b",
        "LinUCB(α=0.5)": "#2ecc71",
        "LinUCB(α=1.0)": "#27ae60",
        "TS(v=0.1)": "#3498db",
        "TS(v=0.5)": "#2980b9",
    }
    return COLORS, RESULTS_PATH, go, make_subplots, mo, np, pickle


@app.cell
def load_results(RESULTS_PATH, pickle):
    with open(RESULTS_PATH, "rb") as _f:
        all_results = pickle.load(_f)
    return (all_results,)


@app.cell
def slide_title(mo):
    mo.md("""
    # Contextual Bandits for Anime Recommendation

    **Aritra Das**

    DSC 291 — Reinforcement Learning | Winter 2026

    ---

    *Balancing exploration and exploitation in sequential anime recommendation using linear contextual bandits on MyAnimeList data.*
    """)
    return


@app.cell
def slide_motivation(mo):
    mo.md("""
    ## Problem & Motivation

    **Anime recommendation is a sequential decision problem:**

    - A user arrives → system shows an anime → observes feedback (rating) → adapts
    - Static recommenders (collaborative filtering, content-based) train once and serve — they get stuck in **filter bubbles**
    - A Naruto fan might love *Violet Evergarden*, but the system never tries it

    **The explore-exploit tension:**

    | | Exploit | Explore |
    |---|---|---|
    | **Action** | Recommend anime similar to known preferences | Try something new to discover hidden gems |
    | **Risk** | Filter bubble, missed discoveries | Short-term suboptimal recommendations |

    **Contextual bandits** give a principled framework for this tradeoff — provably balancing exploration with exploitation using context about both the user and the anime.
    """)
    return


@app.cell
def slide_formulation(mo):
    mo.md(r"""
    ## Formulation

    **Protocol:** At each round $t = 1, \ldots, T$:

    1. A user arrives; system observes $K$ candidate anime with context vectors $\{\mathbf{x}_{a,t}\}_{a=1}^{K}$
    2. System selects one anime $a_t$
    3. System observes reward $r_t$ (the user's rating)

    **Context vector:** $\mathbf{x}_{a,t} = \phi(\text{user}_t, \text{anime}_a) \in \mathbb{R}^{d}$ with $d = 67$

    **Linear reward model:**
    $$\mathbb{E}[r_t \mid \mathbf{x}_{a_t, t}] = \mathbf{x}_{a_t, t}^\top \boldsymbol{\theta}^*$$

    **Goal:** Minimize cumulative pseudo-regret:
    $$R(T) = \sum_{t=1}^{T} \left[ \max_{a \in \mathcal{A}_t} \mathbf{x}_{a,t}^\top \boldsymbol{\theta}^* - \mathbf{x}_{a_t,t}^\top \boldsymbol{\theta}^* \right]$$
    """)
    return


@app.cell
def slide_data(mo):
    mo.md(r"""
    ## Data & Features

    **MyAnimeList Dataset:**
    - **5,000 users** (subsampled, each with ≥ 20 rated anime)
    - **~3,100 anime** (each with ≥ 50 ratings among sampled users)
    - **~940K rated interactions** used as ground truth

    **Context vector** $\mathbf{x} \in \mathbb{R}^{67}$ — three blocks:

    | Block | Dims | Features |
    |-------|------|----------|
    | **User** | 4 | mean score, log(completed), log(days watching), log(list size) |
    | **Anime** | 60 | multi-hot genres (43), score, log(popularity), log(episodes), type (6), source (6), duration, year |
    | **Interaction** | 2 | genre-overlap cosine similarity, score deviation |
    | **Bias** | 1 | constant 1 |

    **Semi-synthetic simulator:** real user ratings serve as ground truth rewards. At each round, we only present anime the user has actually rated → exact reward and regret computation.
    """)
    return


@app.cell
def slide_algorithms(mo):
    mo.md(r"""
    ## Algorithms

    All three algorithms share **online ridge regression** for learning $\boldsymbol{\theta}$:

    $$\mathbf{A}_t = \lambda \mathbf{I} + \sum_{s=1}^{t} \mathbf{x}_s \mathbf{x}_s^\top, \quad \mathbf{b}_t = \sum_{s=1}^{t} r_s \mathbf{x}_s, \quad \hat{\boldsymbol{\theta}}_t = \mathbf{A}_t^{-1} \mathbf{b}_t$$

    Updated in $O(d^2)$ per step via **Sherman-Morrison** rank-1 updates.

    ---

    **ε-Greedy:** With probability $\varepsilon$, pick a random arm; otherwise exploit.
    $$a_t = \begin{cases} \text{Uniform}(\mathcal{A}_t) & \text{w.p. } \varepsilon \\ \arg\max_a \mathbf{x}_{a,t}^\top \hat{\boldsymbol{\theta}}_t & \text{w.p. } 1 - \varepsilon \end{cases}$$

    **LinUCB** *(Li et al., 2010):* Optimism in the face of uncertainty.
    $$a_t = \arg\max_a \left[ \mathbf{x}_{a,t}^\top \hat{\boldsymbol{\theta}}_t + \alpha \sqrt{\mathbf{x}_{a,t}^\top \mathbf{A}_t^{-1} \mathbf{x}_{a,t}} \right]$$

    **Thompson Sampling** *(Agrawal & Goyal, 2013):* Posterior sampling.
    $$\tilde{\boldsymbol{\theta}}_t \sim \mathcal{N}(\hat{\boldsymbol{\theta}}_t, \, v^2 \mathbf{A}_t^{-1}), \quad a_t = \arg\max_a \mathbf{x}_{a,t}^\top \tilde{\boldsymbol{\theta}}_t$$
    """)
    return


@app.cell
def slide_setup(mo):
    mo.md("""
    ## Experiment Setup

    | Parameter | Value |
    |-----------|-------|
    | Rounds $T$ | 50,000 |
    | Arms per round $K$ | 20 |
    | Random seeds | 5 (for mean ± SE) |
    | Feature dimension $d$ | 67 |
    | Ridge regularization $\lambda$ | 1.0 |
    | Reward | Continuous: rating / 10 ∈ [0, 1] |

    **Algorithms tested:**

    | Algorithm | Hyperparameter |
    |-----------|---------------|
    | Random (baseline) | — |
    | ε-Greedy | ε ∈ {0.05, 0.1} |
    | LinUCB | α ∈ {0.5, 1.0} |
    | Thompson Sampling | v ∈ {0.1, 0.5} |

    **Fair comparison:** All algorithms see the **same user/candidate sequence** per seed.
    """)
    return


@app.cell
def slide_results_reward(COLORS, all_results, go, mo, np):
    reward_fig = go.Figure()

    for _name, _trackers in all_results.items():
        _curves = np.array([tr.cumulative_reward for tr in _trackers])
        _mean = _curves.mean(axis=0)
        _se = _curves.std(axis=0) / np.sqrt(len(_trackers))
        _T = len(_mean)
        _step = max(1, _T // 1000)
        _x = np.arange(0, _T, _step)
        _color = COLORS.get(_name, "#9b59b6")

        reward_fig.add_trace(
            go.Scatter(
                x=_x,
                y=_mean[_x],
                mode="lines",
                name=_name,
                line=dict(color=_color, width=2),
            )
        )
        reward_fig.add_trace(
            go.Scatter(
                x=np.concatenate([_x, _x[::-1]]),
                y=np.concatenate(
                    [_mean[_x] + _se[_x], (_mean[_x] - _se[_x])[::-1]]
                ),
                fill="toself",
                fillcolor=_color,
                opacity=0.15,
                line=dict(width=0),
                showlegend=False,
                name=_name + " SE",
            )
        )

    reward_fig.update_layout(
        title="Cumulative Reward Over Time",
        xaxis_title="Round",
        yaxis_title="Cumulative Reward",
        template="plotly_white",
        height=500,
        legend=dict(x=0.02, y=0.98),
    )

    mo.md("## Results: Cumulative Reward")
    mo.ui.plotly(reward_fig)
    return


@app.cell
def slide_results_regret(COLORS, all_results, go, make_subplots, mo, np):
    regret_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cumulative Regret", "Sliding-Window Avg Reward (w=500)"),
    )

    # Left: Regret curves
    for _name, _trackers in all_results.items():
        _curves = np.array([tr.cumulative_regret for tr in _trackers])
        _mean = _curves.mean(axis=0)
        _T = len(_mean)
        _step = max(1, _T // 500)
        _x = np.arange(0, _T, _step)
        _color = COLORS.get(_name, "#9b59b6")
        regret_fig.add_trace(
            go.Scatter(
                x=_x,
                y=_mean[_x],
                mode="lines",
                name=_name,
                line=dict(color=_color, width=2),
            ),
            row=1,
            col=1,
        )

    # sqrt(T) reference
    _T_ref = 50000
    _x_ref = np.arange(0, _T_ref, 100)
    _max_regret = max(
        np.array([tr.cumulative_regret for tr in _trs]).mean(axis=0)[-1]
        for _trs in all_results.values()
    )
    _scale = _max_regret / np.sqrt(_T_ref)
    regret_fig.add_trace(
        go.Scatter(
            x=_x_ref,
            y=_scale * np.sqrt(_x_ref),
            mode="lines",
            name="O(√T) ref",
            line=dict(color="gray", dash="dash", width=1),
        ),
        row=1,
        col=1,
    )

    # Right: Sliding window average reward
    _window = 500
    for _name2, _trackers2 in all_results.items():
        _curves2 = np.array([tr.sliding_avg_reward(_window) for tr in _trackers2])
        _mean2 = _curves2.mean(axis=0)
        _step2 = max(1, len(_mean2) // 500)
        _x2 = np.arange(0, len(_mean2), _step2)
        _color2 = COLORS.get(_name2, "#9b59b6")
        regret_fig.add_trace(
            go.Scatter(
                x=_x2,
                y=_mean2[_x2],
                mode="lines",
                name=_name2,
                line=dict(color=_color2, width=2),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    regret_fig.update_layout(
        template="plotly_white",
        height=450,
        legend=dict(x=0.02, y=0.98),
    )
    regret_fig.update_xaxes(title_text="Round", row=1, col=1)
    regret_fig.update_xaxes(title_text="Round", row=1, col=2)
    regret_fig.update_yaxes(title_text="Cumulative Regret", row=1, col=1)
    regret_fig.update_yaxes(title_text="Avg Reward", row=1, col=2)

    mo.md("## Results: Regret & Learning Dynamics")
    mo.ui.plotly(regret_fig)
    return


@app.cell
def slide_examples(mo, np):
    from src.data_loader import load_all as _load_all
    from src.feature_engineering import FeatureBuilder as _FeatureBuilder
    from src.bandits import LinUCB as _LinUCB

    _data = _load_all()
    _fb = _FeatureBuilder(_data["anime_df"], _data["users_df"], _data["rating_map"])
    _anime_df = _data["anime_df"].set_index("anime_id")

    _rating_map = _data["rating_map"]
    _example_user = sorted(_rating_map.keys())[0]
    _user_ratings = _rating_map[_example_user]

    _agent = _LinUCB(_fb.dim, alpha=1.0)
    _rng = np.random.RandomState(42)

    _all_users = _data["sampled_users"]
    for _ in range(1000):
        _u = _all_users[_rng.randint(len(_all_users))]
        _u_ratings = _rating_map[_u]
        _aids = [_a for _a in _u_ratings if _a in _fb._anime_features]
        if len(_aids) < 2:
            continue
        _aid = _aids[_rng.randint(len(_aids))]
        _xv = _fb.build_context(_u, _aid)
        _r = _u_ratings[_aid] / 10.0
        _agent.update(_xv, _r)

    _user_anime = [_a for _a in _user_ratings if _a in _fb._anime_features]
    _ctxs = _fb.build_context_batch(_example_user, _user_anime)
    _scores = _ctxs @ _agent.theta_hat
    _top_idx = np.argsort(_scores)[::-1][:5]
    _bottom_idx = np.argsort(_scores)[:5]

    _top_rows = []
    for _i in _top_idx:
        _aid = _user_anime[_i]
        _title = (
            _anime_df.loc[_aid, "title"] if _aid in _anime_df.index else str(_aid)
        )
        _actual = _user_ratings[_aid]
        _pred = _scores[_i]
        _top_rows.append(f"| {_title[:40]} | {_actual} | {_pred:.3f} |")

    _bottom_rows = []
    for _i in _bottom_idx:
        _aid = _user_anime[_i]
        _title = (
            _anime_df.loc[_aid, "title"] if _aid in _anime_df.index else str(_aid)
        )
        _actual = _user_ratings[_aid]
        _pred = _scores[_i]
        _bottom_rows.append(f"| {_title[:40]} | {_actual} | {_pred:.3f} |")

    _top_table = "\n".join(_top_rows)
    _bottom_table = "\n".join(_bottom_rows)

    mo.md(
        f"""
    ## Example: LinUCB Recommendations for User `{_example_user}`

    **Top-5 predicted (after 1,000 training steps):**

    | Anime | Actual Rating | Predicted Score |
    |-------|:---:|:---:|
    {_top_table}

    **Bottom-5 predicted:**

    | Anime | Actual Rating | Predicted Score |
    |-------|:---:|:---:|
    {_bottom_table}

    The model learns to predict higher scores for anime that match the user's genre preferences and are generally well-rated.
    """
    )
    return


@app.cell
def slide_conclusion(all_results, mo, np):
    _rows = []
    for _name, _trackers in all_results.items():
        _rewards = [tr.cumulative_reward[-1] for tr in _trackers]
        _regrets = [tr.cumulative_regret[-1] for tr in _trackers]
        _rows.append(
            f"| {_name} | {np.mean(_rewards):.0f} ± {np.std(_rewards):.0f} "
            f"| {np.mean(_regrets):.0f} ± {np.std(_regrets):.0f} |"
        )
    _table = "\n".join(_rows)

    mo.md(
        f"""
    ## Conclusion & Future Work

    **Final results (T = 50,000):**

    | Algorithm | Cumulative Reward | Cumulative Regret |
    |-----------|:-:|:-:|
    {_table}

    **Key takeaways:**
    - LinUCB and Thompson Sampling consistently outperform ε-Greedy and Random
    - Structured exploration (UCB bonus / posterior sampling) > undirected exploration (ε-Greedy)
    - Regret grows sublinearly — the algorithms are learning effectively

    **Future directions:**
    - **Non-stationarity:** user preferences drift over time → sliding-window or discounted bandits
    - **Neural bandits:** replace linear model with neural network for richer representations
    - **Online deployment:** A/B testing with real users on a recommendation platform
    - **Combinatorial bandits:** recommend a *slate* of anime, not just one
    """
    )
    return


if __name__ == "__main__":
    app.run()
