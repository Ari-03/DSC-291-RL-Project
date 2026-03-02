import marimo

__generated_with = "0.20.2"
app = marimo.App(
    width="medium",
    app_title="Contextual Bandits for Anime Recommendation",
    layout_file="layouts/presentation.slides.json",
)


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
        "DecayEG(ε₀=1.0)": "#d35400",
        "LinUCB(α=0.1)": "#1abc9c",
        "LinUCB(α=0.5)": "#2ecc71",
        "LinUCB(α=1.0)": "#27ae60",
        "TS(v=0.1)": "#3498db",
        "TS(v=0.5)": "#2980b9",
        "Popularity": "#f39c12",
        "SVD-CF": "#8e44ad",
        "UserCF": "#16a085",
    }
    return COLORS, RESULTS_PATH, go, make_subplots, mo, np, pickle


@app.cell
def load_results(RESULTS_PATH, pickle):
    with open(RESULTS_PATH, "rb") as _f:
        _raw = pickle.load(_f)
    all_results = _raw
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

    **Context vector:** $\mathbf{x}_{a,t} = \phi(\text{user}_t, \text{anime}_a) \in \mathbb{R}^{d}$ with $d = 72$

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

    **Context vector** $\mathbf{x} \in \mathbb{R}^{72}$ — three blocks:

    | Block | Dims | Features |
    |-------|------|----------|
    | **User** | 4 | mean score, log(completed), log(days watching), log(list size) |
    | **Anime** | 60 | multi-hot genres (43), score, log(popularity), log(episodes), type (6), source (6), duration, year |
    | **Interaction** | 7 | genre cosine sim, score deviation, genre\_cos², genre\_cos×score, genre\_cos×score\_dev, engagement×popularity, score\_dev² |
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
    mo.md(r"""
    ## Experiment Setup

    | Parameter | Value |
    |-----------|-------|
    | Rounds $T$ | 50,000 |
    | Arms per round $K$ | 50 |
    | Random seeds | 5 (for mean ± SE) |
    | Feature dimension $d$ | 72 |
    | Ridge regularization $\lambda$ | 0.1 |
    | Warm-start | Ridge regression on 5,000 offline samples |
    | Reward | Continuous: rating / 10 ∈ [0, 1] |

    **Algorithms tested:**

    | Algorithm | Hyperparameter |
    |-----------|---------------|
    | Random (baseline) | — |
    | ε-Greedy | ε ∈ {0.05, 0.1} |
    | Decaying ε-Greedy | ε₀ = 1.0, ε_t = ε₀/√t |
    | LinUCB | α ∈ {0.1, 0.5, 1.0} |
    | Thompson Sampling | v ∈ {0.1, 0.5} |
    | **Offline CF baselines** | Popularity, SVD, UserCF |

    **Fair comparison:** All algorithms see the **same user/candidate sequence** per seed. CF baselines see the full rating matrix upfront.
    """)
    return


@app.cell
def slide_why_random(go, mo, np):
    """Why does Random's % of oracle decrease with K?"""
    from src.data_loader import load_all as _load_all

    _data = _load_all()
    _rating_map = _data["rating_map"]

    # Collect all ratings
    _all_ratings = []
    for _ur in _rating_map.values():
        _all_ratings.extend(_ur.values())
    _all_ratings = np.array(_all_ratings)

    # Histogram of ratings
    _hist_fig = go.Figure()
    _counts, _edges = np.histogram(_all_ratings, bins=np.arange(0.5, 11.5, 1))
    _hist_fig.add_trace(go.Bar(
        x=list(range(1, 11)),
        y=_counts,
        marker_color="#3498db",
        text=[f"{c/len(_all_ratings)*100:.1f}%" for c in _counts],
        textposition="outside",
    ))
    _hist_fig.update_layout(
        title="Rating Distribution (All 940K Interactions)",
        xaxis_title="Rating", yaxis_title="Count",
        template="plotly_white", height=350,
        xaxis=dict(tickmode="linear", dtick=1),
    )

    # Monte Carlo: compare K=20 vs K=50
    _n_sim = 100_000
    _rng = np.random.RandomState(42)

    _mc_results = {}
    for _K in [20, 50]:
        _means = np.empty(_n_sim)
        _maxes = np.empty(_n_sim)
        for _i in range(_n_sim):
            _sample = _rng.choice(_all_ratings, size=_K, replace=True)
            _means[_i] = _sample.mean()
            _maxes[_i] = _sample.max()
        _mc_results[_K] = {
            "e_mean": _means.mean(),
            "e_max": _maxes.mean(),
            "ratio": _means.mean() / _maxes.mean(),
        }

    _r20 = _mc_results[20]
    _r50 = _mc_results[50]

    mo.md(f"""
    ## Why Does Random's Performance Depend on K?

    The rating distribution is **left-skewed and concentrated** in the 6–9 range.

    - **Global mean rating:** {_all_ratings.mean():.2f} / 10 | **Std dev:** {_all_ratings.std():.2f}

    **Monte Carlo simulation** (100K trials per K):

    | K | E[mean of K] | E[max of K] | E[mean]/E[max] |
    |---|:---:|:---:|:---:|
    | 20 | {_r20['e_mean']:.2f} | {_r20['e_max']:.2f} | **{_r20['ratio']:.1%}** |
    | 50 | {_r50['e_mean']:.2f} | {_r50['e_max']:.2f} | **{_r50['ratio']:.1%}** |

    With K=50, max grows faster than mean → Random drops to **{_r50['ratio']:.1%}** of oracle.
    This widens the gap where **smart algorithms compete**.
    """)
    mo.ui.plotly(_hist_fig)
    return


@app.cell
def slide_results_reward(COLORS, all_results, go, mo, np):
    reward_fig = go.Figure()

    # Compute oracle cumulative reward (same across all algorithms)
    _first_trackers = next(iter(all_results.values()))
    _oracle_curves = np.array([
        np.cumsum(tr.oracle_rewards[:tr.t]) for tr in _first_trackers
    ])
    _oracle_mean = _oracle_curves.mean(axis=0)

    for _name, _trackers in all_results.items():
        _curves = np.array([tr.cumulative_reward for tr in _trackers])
        _oracle_per_seed = np.array([
            np.cumsum(tr.oracle_rewards[:tr.t]) for tr in _trackers
        ])
        # % of oracle reward per seed, then average
        _pct_curves = 100.0 * _curves / _oracle_per_seed
        _mean = _pct_curves.mean(axis=0)
        _se = _pct_curves.std(axis=0) / np.sqrt(len(_trackers))
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
        title="Cumulative Reward as % of Oracle",
        xaxis_title="Round",
        yaxis_title="% of Oracle Reward",
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
        subplot_titles=("Normalized Regret R(t)/t", "Sliding-Window Avg Reward (w=500)"),
    )

    # Left: Normalized regret R(t)/t — average regret per round
    for _name, _trackers in all_results.items():
        _curves = np.array([tr.cumulative_regret for tr in _trackers])
        _mean = _curves.mean(axis=0)
        _T = len(_mean)
        _rounds = np.arange(1, _T + 1)
        _normalized = _mean / _rounds  # R(t)/t
        _step = max(1, _T // 500)
        _x = np.arange(0, _T, _step)
        _color = COLORS.get(_name, "#9b59b6")
        regret_fig.add_trace(
            go.Scatter(
                x=_x,
                y=_normalized[_x],
                mode="lines",
                name=_name,
                line=dict(color=_color, width=2),
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
    regret_fig.update_yaxes(title_text="Avg Regret Per Round", row=1, col=1)
    regret_fig.update_yaxes(title_text="Avg Reward", row=1, col=2)

    mo.md("## Results: Regret & Learning Dynamics")
    mo.ui.plotly(regret_fig)
    return


@app.cell
def _(np):
    """Shared computation: train LinUCB and log uncertainty metrics."""
    from src.data_loader import load_all as _load_all
    from src.feature_engineering import (
        FeatureBuilder as _FeatureBuilder,
        ALL_GENRES as _ALL_GENRES,
        ALL_TYPES as _ALL_TYPES,
        ALL_SOURCE_GROUPS as _ALL_SOURCE_GROUPS,
    )
    from src.environment import AnimeRecommendationEnv as _Env
    from src.bandits import LinUCB as _LinUCB

    _data = _load_all()
    _fb = _FeatureBuilder(_data["anime_df"], _data["users_df"], _data["rating_map"])
    _env = _Env(_fb, _data["rating_map"], _data["sampled_users"], K=50, reward_type="continuous")
    _seq = _env.generate_sequence(T=50_000, seed=0)

    _agent = _LinUCB(_fb.dim, alpha=0.5, lam=0.1)
    _rng = np.random.RandomState(0)

    # Sample 100 fixed contexts for tracking UCB width
    _fixed_contexts = np.array([_seq[i][2][0] for i in range(0, 1000, 10)])  # 100 contexts

    trace_history = []
    ucb_width_history = []
    log_rounds = []

    for _t, (_user, _aids, _ctxs, _oracle) in enumerate(_seq):
        _arm = _agent.select_arm(_ctxs, _rng)
        _r = _env.get_reward(_user, _aids[_arm])
        _agent.update(_ctxs[_arm], _r)

        if (_t + 1) % 100 == 0:
            log_rounds.append(_t + 1)
            trace_history.append(np.trace(_agent.A_inv))
            _temp = _fixed_contexts @ _agent.A_inv
            _widths = np.sqrt(np.sum(_temp * _fixed_contexts, axis=1))
            ucb_width_history.append(_widths.mean())

    learned_theta = _agent.theta_hat.copy()

    # Build feature names (72 total)
    feature_names = (
        ["user: mean_score", "user: log_completed", "user: log_days", "user: log_list_size"]
        + [f"genre: {g}" for g in _ALL_GENRES]
        + ["anime: score", "anime: log_pop", "anime: log_episodes"]
        + [f"type: {t}" for t in _ALL_TYPES]
        + [f"source: {s}" for s in _ALL_SOURCE_GROUPS]
        + ["anime: duration", "anime: year"]
        + ["interaction: genre_cos_sim", "interaction: score_dev",
           "interaction: genre_cos²", "interaction: genre_cos×score",
           "interaction: genre_cos×score_dev", "interaction: engagement×pop",
           "interaction: score_dev²"]
        + ["bias"]
    )

    trace_history = np.array(trace_history)
    ucb_width_history = np.array(ucb_width_history)
    log_rounds = np.array(log_rounds)
    return (
        feature_names,
        learned_theta,
        log_rounds,
        trace_history,
        ucb_width_history,
    )


@app.cell
def _(COLORS, all_results, go, mo, np):
    """Slide A: Regret Growth Rate Analysis (log-log)."""
    _KEY_ALGOS = ["Random", "EpsGreedy(ε=0.05)", "DecayEG(ε₀=1.0)", "LinUCB(α=0.5)", "TS(v=0.1)"]
    _fig = go.Figure()
    _skip = 1000  # skip early transient

    _rows = []
    for _name in _KEY_ALGOS:
        _trackers = all_results[_name]
        _curves = np.array([tr.cumulative_regret for tr in _trackers])
        _mean = _curves.mean(axis=0)
        _T = len(_mean)
        _x = np.arange(1, _T + 1)
        _color = COLORS.get(_name, "#9b59b6")

        # Log-log plot
        _log_x = np.log10(_x[_skip:])
        _log_y = np.log10(np.maximum(_mean[_skip:], 1e-10))
        _step = max(1, len(_log_x) // 500)
        _fig.add_trace(
            go.Scatter(
                x=_log_x[::_step], y=_log_y[::_step],
                mode="lines", name=_name,
                line=dict(color=_color, width=2),
            )
        )

        # Fit power law: log R = alpha * log t + log c
        _coeffs = np.polyfit(_log_x, _log_y, 1)
        _alpha = _coeffs[0]
        _rows.append(f"| {_name} | {_alpha:.3f} | {'Sublinear' if _alpha < 0.95 else 'Linear'} |")

        # Annotate fitted alpha on the curve
        _mid_idx = len(_log_x) // 2
        _fig.add_annotation(
            x=_log_x[_mid_idx], y=_log_y[_mid_idx],
            text=f"α={_alpha:.2f}",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            font=dict(size=11, color=_color),
            ax=30, ay=-25,
        )

    # Add reference slopes
    _x_ref = np.linspace(np.log10(_skip + 1), np.log10(50000), 100)
    _fig.add_trace(go.Scatter(x=_x_ref, y=0.5 * _x_ref + 1.0, mode="lines",
                              name="slope=0.5 (√T)", line=dict(color="gray", dash="dash", width=1)))
    _fig.add_trace(go.Scatter(x=_x_ref, y=1.0 * _x_ref - 0.5, mode="lines",
                              name="slope=1.0 (linear)", line=dict(color="gray", dash="dot", width=1)))

    _fig.update_layout(
        title="Cumulative Regret: Log-Log Scale",
        xaxis_title="log₁₀(Round)", yaxis_title="log₁₀(Cumulative Regret)",
        template="plotly_white", height=400,
        legend=dict(x=0.02, y=0.98),
    )
    _table = "\n".join(_rows)

    mo.md(f"""
    ## Regret Growth Rate Analysis

    Fit $R(t) = c \\cdot t^\\alpha$ on log-log scale (skipping first 1,000 rounds):

    | Algorithm | Fitted α | Interpretation |
    |-----------|:---:|----------------|
    {_table}

    **Theory:** LinUCB regret bound is $R(T) = O(d\\sqrt{{T}} \\log T)$, so $\\alpha \\approx 0.5$.
    Random policy has $\\alpha \\approx 1$ (linear regret).
    """)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(COLORS, all_results, go, mo, np):
    """Slide B: Instantaneous Regret Over Time."""
    _KEY_ALGOS_B = ["Random", "EpsGreedy(ε=0.05)", "DecayEG(ε₀=1.0)", "LinUCB(α=0.5)", "TS(v=0.1)"]
    _fig = go.Figure()
    _window = 1000

    for _name in _KEY_ALGOS_B:
        _trackers = all_results[_name]
        _inst_regrets = np.array([
            tr.oracle_rewards[:tr.t] - tr.rewards[:tr.t] for tr in _trackers
        ])
        _mean_inst = _inst_regrets.mean(axis=0)
        _kernel = np.ones(_window) / _window
        _smoothed = np.convolve(_mean_inst, _kernel, mode="valid")
        _x = np.arange(len(_smoothed))
        _step = max(1, len(_x) // 500)
        _color = COLORS.get(_name, "#9b59b6")

        _fig.add_trace(
            go.Scatter(
                x=_x[::_step], y=_smoothed[::_step],
                mode="lines", name=_name,
                line=dict(color=_color, width=2),
            )
        )

    _fig.update_layout(
        title="Instantaneous Regret (Smoothed, window=1000)",
        xaxis_title="Round", yaxis_title="Per-Round Regret",
        template="plotly_white", height=450,
        legend=dict(x=0.75, y=0.98),
    )

    mo.md(r"""
    ## Instantaneous Regret Over Time

    Per-round regret $r_t^* - r_t$ smoothed with a 1,000-round moving average.

    - **Random:** flat — no learning, constant expected regret
    - **LinUCB / TS:** decreasing — converging to optimal policy
    - **ε-Greedy:** plateaus above zero — residual $\varepsilon$-exploration cost

    **Theory:** For sublinear cumulative regret $R(T) = o(T)$, we need instantaneous regret $\to 0$.
    This connects to the Azuma-Hoeffding martingale concentration analysis of the regret process.
    """)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(feature_names, go, learned_theta, mo, np):
    """Slide C: Learned Feature Weights (theta_hat)."""
    _theta = learned_theta
    _names = feature_names

    # Sort by weight
    _sorted_idx = np.argsort(_theta)
    _top15 = _sorted_idx[-15:][::-1]  # highest 15
    _bot5 = _sorted_idx[:5]            # lowest 5
    _show_idx = np.concatenate([_top15, _bot5])

    _fig = go.Figure()
    _colors = ["#2ecc71" if _theta[i] >= 0 else "#e74c3c" for i in _show_idx]

    _fig.add_trace(go.Bar(
        y=[_names[i] for i in _show_idx],
        x=[_theta[i] for i in _show_idx],
        orientation="h",
        marker_color=_colors,
    ))
    _fig.update_layout(
        title="LinUCB(α=0.5): Learned Feature Weights θ̂ (50K rounds)",
        xaxis_title="Weight", yaxis_title="",
        template="plotly_white", height=550,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200),
    )

    mo.md(r"""
    ## Learned Feature Weights ($\hat{\boldsymbol{\theta}}$)

    Top-15 positive and bottom-5 negative weights from LinUCB after 50,000 rounds.

    **Theory:** $\hat{\boldsymbol{\theta}}_T = \mathbf{A}_T^{-1}\mathbf{b}_T$ is the online ridge regression
    solution, converging to $\boldsymbol{\theta}^*$ at rate $O(1/\sqrt{T})$ (lecture 2).
    """)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(go, log_rounds, make_subplots, mo, trace_history, ucb_width_history):
    """Slide D: Confidence Ellipsoid Shrinkage."""
    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("trace(A⁻¹) Over Time", "Avg UCB Width Over Time"),
    )

    _fig.add_trace(
        go.Scatter(x=log_rounds, y=trace_history, mode="lines",
                   line=dict(color="#2ecc71", width=2), name="trace(A⁻¹)"),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scatter(x=log_rounds, y=ucb_width_history, mode="lines",
                   line=dict(color="#3498db", width=2), name="Avg UCB width"),
        row=1, col=2,
    )

    _fig.update_layout(template="plotly_white", height=400, showlegend=False)
    _fig.update_xaxes(title_text="Round", row=1, col=1)
    _fig.update_xaxes(title_text="Round", row=1, col=2)
    _fig.update_yaxes(title_text="trace(A⁻¹)", type="log", row=1, col=1)
    _fig.update_yaxes(title_text="√(xᵀA⁻¹x)", type="log", row=1, col=2)

    mo.md(r"""
    ## Confidence Ellipsoid Shrinkage

    As data accumulates, $\mathbf{A}_t = \lambda\mathbf{I} + \sum_{s=1}^{t}\mathbf{x}_s\mathbf{x}_s^\top$ grows,
    so $\mathbf{A}_t^{-1}$ shrinks. This drives LinUCB from **exploration → exploitation**.

    - **Left:** $\text{trace}(\mathbf{A}_t^{-1})$ — overall uncertainty scale (sum of eigenvalues of $\mathbf{A}_t^{-1}$)
    - **Right:** $\sqrt{\mathbf{x}^\top \mathbf{A}_t^{-1} \mathbf{x}}$ — the UCB bonus term, averaged over 100 fixed contexts

    **Theory:** The confidence set $\mathcal{C}_t = \{\boldsymbol{\theta}: \|\boldsymbol{\theta} - \hat{\boldsymbol{\theta}}_t\|_{\mathbf{A}_t} \le \beta_t\}$
    contracts as $\mathbf{A}_t$ grows. LinUCB's regret bound follows from this contraction (lecture 2/3).
    """)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(COLORS, all_results, go, mo, np):
    """Slide E: Per-Round Regret Distribution (box plots)."""
    _KEY_ALGOS_E = ["Random", "EpsGreedy(ε=0.05)", "DecayEG(ε₀=1.0)", "LinUCB(α=0.5)", "TS(v=0.1)"]
    _fig = go.Figure()

    for _name in _KEY_ALGOS_E:
        _trackers = all_results[_name]
        # Compute per-round regret: oracle - actual
        _regret_all = np.concatenate([
            tr.oracle_rewards[:tr.t] - tr.rewards[:tr.t] for tr in _trackers
        ])
        _color = COLORS.get(_name, "#9b59b6")
        _fig.add_trace(go.Box(
            y=_regret_all[::10],  # subsample for performance
            name=_name,
            marker_color=_color,
            boxmean=True,
        ))

    _fig.update_layout(
        title="Per-Round Regret Distribution (across all seeds)",
        yaxis_title="Per-Round Regret (oracle − actual)",
        template="plotly_white", height=500,
        showlegend=False,
    )

    mo.md(r"""
    ## Regret Distribution

    Box plots of per-round regret $r_t^* - r_t$ (subsampled 10x for rendering). Diamond shows mean.

    - **Random:** wide distribution, high mean regret — never learns
    - **EpsGreedy:** moderate regret, residual exploration cost
    - **LinUCB / TS:** concentrated near zero with thin right tails — mostly picking near-optimal arms

    **Theory:** Sub-Gaussian rewards with parameter $\sigma$ satisfy
    $\Pr[|\bar{X}_n - \mu| > \varepsilon] \le 2\exp(-n\varepsilon^2/2\sigma^2)$ — tighter concentration
    as algorithms learn to select better arms (lecture 1).
    """)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(COLORS, all_results, go, mo, np):
    """Slide F: Hyperparameter Sensitivity (bar chart)."""
    _names = list(all_results.keys())
    _final_regrets_mean = []
    _final_regrets_se = []
    _colors = []

    for _name in _names:
        _trackers = all_results[_name]
        _finals = np.array([tr.cumulative_regret[-1] for tr in _trackers])
        _final_regrets_mean.append(_finals.mean())
        _final_regrets_se.append(_finals.std() / np.sqrt(len(_finals)))
        _colors.append(COLORS.get(_name, "#9b59b6"))

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=_names,
        y=_final_regrets_mean,
        error_y=dict(type="data", array=_final_regrets_se, visible=True),
        marker_color=_colors,
    ))

    _fig.update_layout(
        title="Final Cumulative Regret (T=50,000) — Mean ± SE over 5 seeds",
        xaxis_title="Algorithm", yaxis_title="Cumulative Regret",
        template="plotly_white", height=450,
    )

    mo.md(r"""
    ## Hyperparameter Sensitivity

    Final cumulative regret at $T = 50{,}000$ for all algorithm variants. Error bars show ± 1 SE (5 seeds).

    - **Within families**, lower exploration parameters perform better on this dataset:
      - LinUCB: $\alpha=0.5$ < $\alpha=1.0$
      - TS: $v=0.1$ < $v=0.5$
      - ε-Greedy: $\varepsilon=0.05$ < $\varepsilon=0.1$
    - **Across families:** LinUCB ≈ TS ≪ ε-Greedy ≪ Random

    **Theory:** The regret bound for LinUCB scales linearly with $\alpha$: $R(T) = O(\alpha d \sqrt{T})$.
    Thompson Sampling's Bayes regret scales with $v$ similarly.
    """)
    mo.ui.plotly(_fig)
    return


@app.cell
def slide_bandits_vs_cf(COLORS, all_results, go, make_subplots, mo, np):
    """Bandits vs. Collaborative Filtering comparison."""
    _KEY = ["Random", "Popularity", "SVD-CF", "UserCF", "LinUCB(α=0.5)", "TS(v=0.1)"]
    _available = [k for k in _KEY if k in all_results]

    # Left: cumulative reward as % of oracle over time for key methods
    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Cumulative Reward (% of Oracle) Over Time",
            "Final Cumulative Reward (T=50K)",
        ),
        column_widths=[0.6, 0.4],
    )

    # Oracle curve (same across all)
    _first_trackers = next(iter(all_results.values()))
    _oracle_per_seed = np.array([
        np.cumsum(tr.oracle_rewards[:tr.t]) for tr in _first_trackers
    ])

    for _name in _available:
        _trackers = all_results[_name]
        _curves = np.array([tr.cumulative_reward for tr in _trackers])
        _oracle_s = np.array([
            np.cumsum(tr.oracle_rewards[:tr.t]) for tr in _trackers
        ])
        _pct = 100.0 * _curves / _oracle_s
        _mean = _pct.mean(axis=0)
        _T = len(_mean)
        _step = max(1, _T // 500)
        _x = np.arange(0, _T, _step)
        _color = COLORS.get(_name, "#9b59b6")

        _fig.add_trace(
            go.Scatter(
                x=_x, y=_mean[_x],
                mode="lines", name=_name,
                line=dict(color=_color, width=2),
            ),
            row=1, col=1,
        )

    # Right: bar chart of final cumulative reward, grouped
    _all_names = list(all_results.keys())
    _offline = [n for n in _all_names if n in ("Popularity", "SVD-CF", "UserCF")]
    _online = [n for n in _all_names if n not in _offline]
    _order = _online + _offline

    _final_rewards = []
    _bar_colors = []
    for _name in _order:
        _trackers = all_results[_name]
        _finals = np.array([tr.cumulative_reward[-1] for tr in _trackers])
        _final_rewards.append(_finals.mean())
        _bar_colors.append(COLORS.get(_name, "#9b59b6"))

    _fig.add_trace(
        go.Bar(
            x=_order, y=_final_rewards,
            marker_color=_bar_colors,
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Add a vertical separator annotation
    _fig.add_vline(
        x=len(_online) - 0.5, line_dash="dash", line_color="gray",
        row=1, col=2,
    )
    _fig.add_annotation(
        x=len(_online) - 0.5, y=max(_final_rewards) * 1.05,
        text="← Online | Offline →", showarrow=False,
        font=dict(size=10, color="gray"),
        row=1, col=2,
    )

    _fig.update_layout(
        template="plotly_white", height=500,
        legend=dict(x=0.02, y=0.02, yanchor="bottom"),
    )
    _fig.update_xaxes(title_text="Round", row=1, col=1)
    _fig.update_yaxes(title_text="% of Oracle", row=1, col=1)
    _fig.update_xaxes(title_text="", tickangle=-30, row=1, col=2)
    _fig.update_yaxes(title_text="Cumulative Reward", row=1, col=2)

    # Compute key comparison numbers
    _linucb_pct = 0.0
    _ucf_pct = 0.0
    if "LinUCB(α=0.5)" in all_results:
        _tr = all_results["LinUCB(α=0.5)"]
        _c = np.array([tr.cumulative_reward for tr in _tr])
        _o = np.array([np.cumsum(tr.oracle_rewards[:tr.t]) for tr in _tr])
        _linucb_pct = (100.0 * _c / _o).mean(axis=0)[-1]
    if "UserCF" in all_results:
        _tr = all_results["UserCF"]
        _c = np.array([tr.cumulative_reward for tr in _tr])
        _o = np.array([np.cumsum(tr.oracle_rewards[:tr.t]) for tr in _tr])
        _ucf_pct = (100.0 * _c / _o).mean(axis=0)[-1]

    _comparison = ""
    if _linucb_pct > 0 and _ucf_pct > 0:
        _diff = _linucb_pct - _ucf_pct
        if abs(_diff) < 1.0:
            _comparison = f"LinUCB ({_linucb_pct:.1f}%) ≈ UserCF ({_ucf_pct:.1f}%) — **online bandits match pretrained CF** despite starting from scratch, and they can adapt to preference drift."
        elif _diff > 0:
            _comparison = f"LinUCB ({_linucb_pct:.1f}%) > UserCF ({_ucf_pct:.1f}%) — **online bandits outperform pretrained CF**, a strong result given bandits start from zero."
        else:
            _comparison = f"UserCF ({_ucf_pct:.1f}%) > LinUCB ({_linucb_pct:.1f}%) — CF has an offline advantage, but bandits **close the gap over time** and handle cold-start / non-stationarity."

    mo.md(f"""
    ## Bandits vs. Collaborative Filtering

    **Key caveat:** CF baselines see the **entire 940K-interaction rating matrix** upfront.
    Bandits start from scratch and learn online. This is an **unfair comparison by design** — if bandits match CF, it's a strong result.

    {_comparison}

    **When to prefer bandits over CF:**
    - **Cold-start:** new users/items with no rating history
    - **Non-stationarity:** user preferences drift over time
    - **Exploration:** CF exploits known patterns; bandits discover new ones
    """)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(all_results, go, mo, np):
    """Slide G: Statistical Significance (heatmap)."""
    from scipy.stats import ttest_rel as _ttest_rel

    _SHORT_NAMES = {
        "Random": "Rnd",
        "EpsGreedy(ε=0.05)": "EG.05",
        "EpsGreedy(ε=0.1)": "EG.1",
        "DecayEG(ε₀=1.0)": "DEG",
        "LinUCB(α=0.1)": "UCB.1",
        "LinUCB(α=0.5)": "UCB.5",
        "LinUCB(α=1.0)": "UCB1",
        "TS(v=0.1)": "TS.1",
        "TS(v=0.5)": "TS.5",
        "Popularity": "Pop",
        "SVD-CF": "SVD",
        "UserCF": "UCF",
    }

    _names = list(all_results.keys())
    _short = [_SHORT_NAMES.get(_n, _n) for _n in _names]
    _n = len(_names)

    # Final cumulative reward per seed for each algorithm
    _final_rewards = {}
    for _name, _trackers in all_results.items():
        _final_rewards[_name] = np.array([tr.cumulative_reward[-1] for tr in _trackers])

    _pvals = np.ones((_n, _n))
    for _i in range(_n):
        for _j in range(_n):
            if _i != _j:
                _, _p = _ttest_rel(_final_rewards[_names[_i]], _final_rewards[_names[_j]])
                _pvals[_i, _j] = _p

    # Format annotations
    _annot = []
    for _i in range(_n):
        _row = []
        for _j in range(_n):
            if _i == _j:
                _row.append("—")
            elif _pvals[_i, _j] < 0.001:
                _row.append(f"{_pvals[_i, _j]:.1e}")
            else:
                _row.append(f"{_pvals[_i, _j]:.3f}")
        _annot.append(_row)

    _fig = go.Figure(data=go.Heatmap(
        z=_pvals,
        x=_short, y=_short,
        colorscale=[[0, "#2ecc71"], [0.05, "#f1c40f"], [0.2, "#e74c3c"], [1.0, "#e74c3c"]],
        zmin=0, zmax=0.2,
        text=_annot,
        texttemplate="%{text}",
        colorbar=dict(title="p-value"),
    ))

    _fig.update_layout(
        title="Pairwise Paired t-test p-values (Final Cumulative Reward)",
        template="plotly_white", height=500,
        xaxis=dict(tickangle=0),
    )

    mo.md(r"""
    ## Statistical Significance

    Paired t-test on final cumulative reward across 5 seeds (same random sequences → natural pairing).

    - **Green cells** ($p < 0.05$): statistically significant difference
    - **Red/yellow cells** ($p > 0.05$): not enough evidence to distinguish

    With only 5 seeds, we have limited statistical power. Differences between similar algorithms
    (e.g., LinUCB vs TS) may not be significant, while Random vs. others clearly is.

    **Design choice:** Paired tests control for sequence randomness, increasing power vs. unpaired tests.
    """)
    mo.ui.plotly(_fig)
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
    - **Sublinear regret confirmed:** Fitted growth rates show LinUCB/TS achieve $\\alpha \\approx 0.5$–$0.7$ (sublinear), while Random has $\\alpha \\approx 1$ (linear) — matching the theoretical $O(d\\sqrt{{T}})$ bound
    - **K=50 widens the gap:** With more arms, Random drops significantly while bandits maintain high oracle %, creating a much clearer separation
    - **Interaction features + warm-start** give bandits richer signal and a head start, pushing them closer to oracle performance
    - **Structured > undirected exploration:** UCB bonus and posterior sampling outperform $\\varepsilon$-Greedy; decaying $\\varepsilon$-greedy bridges the gap by reducing exploration over time
    - **Competitive with CF baselines:** Online bandits match or exceed pretrained collaborative filtering (Popularity, SVD, UserCF) despite starting from scratch — and offer cold-start/non-stationarity advantages CF cannot

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
