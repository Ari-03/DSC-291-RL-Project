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
    - **~940K rated interactions**, split **70/30 per user** into train/test
    - Train set: features, warm-start, CF baselines | Test set: environment rewards

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
    | Train/test split | 70/30 per user |
    | Warm-start | Ridge regression on 10,000 offline samples |
    | Reward | User-centered: (rating − user_mean) / 10 |

    **Algorithms tested:**

    | Algorithm | Hyperparameter |
    |-----------|---------------|
    | Random (baseline) | — |
    | ε-Greedy | ε ∈ {0.05, 0.1} |
    | Decaying ε-Greedy | ε₀ = 1.0, ε_t = ε₀/√t |
    | LinUCB | α ∈ {0.1, 0.5, 1.0} |
    | Thompson Sampling | v ∈ {0.1, 0.5} |
    | **Offline CF baselines** | SVD, UserCF |

    **Fair comparison:** All algorithms see the **same user/candidate sequence** per seed. CF baselines are trained on 70% of ratings; rewards come from the held-out 30%.
    """)
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
def slide_bandits_vs_cf(COLORS, all_results, go, make_subplots, mo, np):
    """Bandits vs. Collaborative Filtering comparison."""
    _KEY = ["Random", "SVD-CF", "UserCF", "LinUCB(α=0.5)", "TS(v=0.1)"]
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
    _offline = [n for n in _all_names if n in ("SVD-CF", "UserCF")]
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
            _comparison = f"LinUCB ({_linucb_pct:.1f}%) ≈ UserCF ({_ucf_pct:.1f}%) — **online bandits match CF** in a fair train/test evaluation."
        elif _diff > 0:
            _comparison = f"LinUCB ({_linucb_pct:.1f}%) > UserCF ({_ucf_pct:.1f}%) — **online bandits outperform CF** when CF must generalize to unseen ratings."
        else:
            _comparison = f"UserCF ({_ucf_pct:.1f}%) > LinUCB ({_linucb_pct:.1f}%) — CF retains an edge, but bandits **close the gap** and handle cold-start / non-stationarity."

    mo.md(f"""
    ## Bandits vs. Collaborative Filtering

    **Fair evaluation:** CF baselines are trained on 70% of each user's ratings; rewards come from the held-out 30%.
    CF must **generalize** — it can no longer memorize the test set. Bandits learn online with user-centered rewards.

    {_comparison}

    **When to prefer bandits over CF:**
    - **Cold-start:** new users/items with no rating history
    - **Non-stationarity:** user preferences drift over time
    - **Exploration:** CF exploits known patterns; bandits discover new ones
    """)
    mo.ui.plotly(_fig)
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
    - **Fair evaluation via train/test split:** CF baselines trained on 70% of ratings, evaluated on held-out 30% — no data leakage
    - **User-centered rewards** remove per-user bias so the linear model learns relative preferences, not absolute ratings
    - **Structured > undirected exploration:** UCB bonus and posterior sampling outperform $\\varepsilon$-Greedy; decaying $\\varepsilon$-greedy bridges the gap by reducing exploration over time
    - **Competitive with CF baselines:** With fair evaluation, online bandits are competitive with collaborative filtering — and offer cold-start/non-stationarity advantages CF cannot

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
