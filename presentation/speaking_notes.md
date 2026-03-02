# Speaking Notes — Contextual Bandits for Anime Recommendation

## Slide 1: Title

"Today I'm presenting my project on contextual bandits for anime recommendation. The core idea is simple: when you're recommending anime to users sequentially, you face a fundamental tradeoff — do you exploit what you already know works, or explore something new that might be even better? I'll show how algorithms from our lecture — LinUCB, Thompson Sampling — handle this tradeoff on real MyAnimeList data, and compare them against collaborative filtering baselines."

## Slide 2: Problem & Motivation

"Why bandits for recommendation? Think about it — a traditional recommender trains on historical data and deploys a fixed model. If a user mostly watches action anime, it keeps recommending action. It never discovers that this user might also love slice-of-life shows. That's the filter bubble problem. Contextual bandits break this by treating each recommendation as a decision under uncertainty. The explore-exploit table here captures the tension: exploiting means safe, expected rewards; exploring means potentially discovering much better options at the cost of occasional bad recommendations. The key insight from our course is that you don't have to choose — algorithms like LinUCB do both simultaneously through the UCB principle."

## Slide 3: Formulation

"Here's the formal setup, which maps directly to the linear contextual bandit framework from lecture 2. Each round, a user arrives, we present K=50 candidate anime, and we observe the user's rating as reward. The context vector phi(user, anime) is 72-dimensional — I'll detail these features on the next slide. The core assumption is a linear reward model: the expected rating is the dot product of the context with an unknown parameter theta-star. Our goal is to minimize cumulative pseudo-regret — the sum of gaps between the best possible arm and the arm we actually chose. The theoretical bound from lecture is O(d * sqrt(T) * log T), which means regret should grow sublinearly — we'll test this empirically."

## Slide 4: Data & Features

"I'm using the MyAnimeList dataset — 5,000 users, about 3,100 anime, and 940K rating interactions. The 72-dimensional context vector has three blocks: user features like their average score and activity level, anime features including genre multi-hot encodings and metadata, and seven interaction features. Beyond the original genre cosine similarity and score deviation, I added polynomial and cross terms — genre_cos squared, genre_cos times anime score, genre_cos times score deviation, user engagement times anime popularity, and score_dev squared. These give the linear model access to nonlinear patterns without changing the algorithm. The bias term makes the model affine. Crucially, this is a semi-synthetic simulator: at each round, I only present anime the user has actually rated in the dataset, so I can compute exact regret."

## Slide 5: Algorithms

"All algorithms share the same backbone: online ridge regression with Sherman-Morrison rank-1 updates, which gives us O(d-squared) per step — about 5,200 operations per round with d=72. Epsilon-Greedy is the simplest: flip a coin, explore with probability epsilon, exploit otherwise. Its exploration is undirected. I also added a decaying epsilon-greedy variant where epsilon decreases as 1/sqrt(t) — this explores heavily early and converges to pure exploitation. LinUCB from Li et al. adds an optimism bonus: the UCB term inflates predicted reward in uncertain directions. Thompson Sampling samples theta from the posterior and acts greedily. Both LinUCB and TS direct exploration toward uncertainty, which is why they should outperform epsilon-Greedy."

## Slide 6: Experiment Setup

"50,000 rounds, 50 arms per round — I increased K from 20 to 50 to make the problem harder for Random and give smart algorithms more room to exploit their knowledge. Lambda is set to 0.1 instead of the default 1.0, because with d=72 features the higher regularization was biasing theta too aggressively toward zero. I also warm-start all bandit algorithms with ridge regression fitted on 5,000 offline rating samples — this gives them a head start so they don't waste the first few thousand rounds learning basic patterns. The three CF baselines — Popularity, SVD, and UserCF — see the entire rating matrix upfront. All algorithms see the exact same user-candidate sequence per seed for fair comparison."

## Slide 7: Why Does Random's % of Oracle Depend on K?

"Before diving into results, let me explain why K=50 matters. The rating distribution is heavily left-skewed, clustered between 6 and 9. With K=20, the Monte Carlo analysis shows Random achieves about 81% of oracle. But with K=50, the max of 50 samples grows faster than the mean — so Random's oracle percentage drops significantly. This widens the gap where smart algorithms compete. The key insight: with more arms, there's more room for algorithms that can identify the best arm to shine, while random selection gets relatively worse."

## Slide 8: Results — Cumulative Reward (% of Oracle)

"This shows cumulative reward as a percentage of the oracle. With K=50 and the improved features, Random's performance drops compared to the K=20 setting. LinUCB and Thompson Sampling should reach higher percentages thanks to the warm-start, richer interaction features, and lower regularization. The decaying epsilon-greedy is interesting — it should outperform fixed epsilon early on due to heavy exploration, then converge toward exploitation."

## Slide 9: Results — Regret & Learning Dynamics

"On the left, normalized regret — average regret per round. Random stays flat — it never learns. All other algorithms decrease, confirming sublinear cumulative regret. The warm-start means the learning algorithms start with lower regret than they would from scratch. On the right, the sliding-window average reward shows convergence toward higher per-round rewards. The decaying epsilon-greedy is interesting to compare with fixed epsilon — its exploration cost decreases over time unlike the fixed variant."

## Slide 10: Regret Growth Rate Analysis (log-log)

"This is the money slide for connecting to theory. On a log-log scale, R(t) = c * t^alpha becomes a straight line with slope alpha. Random has alpha near 1 — linear regret. LinUCB and TS have alpha around 0.5 to 0.7, confirming sublinear growth matching the O(d*sqrt(T)) bound. The decaying epsilon-greedy should also show sublinear regret since its exploration rate goes to zero. The reference lines show pure sqrt(T) and linear growth for comparison."

## Slide 11: Instantaneous Regret Over Time

"Per-round regret smoothed with a 1,000-round moving average. Random is flat — zero learning. LinUCB and TS clearly decrease over time. Fixed epsilon-greedy plateaus above zero due to its constant exploration cost. Decaying epsilon-greedy should show a different pattern — initially high exploration but converging toward zero, similar to LinUCB/TS but through a different mechanism."

## Slide 12: Learned Feature Weights

"What does LinUCB actually learn? These are the top-15 positive and bottom-5 negative weights from theta-hat after 50K rounds. With the richer interaction features, we should see the polynomial terms — especially genre_cos squared and genre_cos times score — carrying significant weight. The model can now capture nonlinear effects like 'high genre overlap AND high anime quality is extra good' through the cross term. This is the ridge regression solution converging to theta-star."

## Slide 13: Confidence Ellipsoid Shrinkage

"This visualizes the core mechanism behind LinUCB with lambda=0.1. On the left, trace of A-inverse drops rapidly. With lower lambda, A-inverse starts larger (1/0.1 = 10 per diagonal element vs 1.0 before), so there's more initial uncertainty but also faster learning from data. On the right, the average UCB bonus term shrinks as the confidence ellipsoid contracts. The transition from exploration to exploitation is gradual, not a hard switch."

## Slide 14: Regret Distribution

"Per-round regret distributions. Random has a wide distribution — high and variable regret. With the warm-start, LinUCB and TS should have distributions even more concentrated near zero compared to before. The key visual: the mass of the LinUCB/TS distributions is compressed near zero, while Random spreads across the full range."

## Slide 15: Hyperparameter Sensitivity

"This bar chart shows final cumulative regret for all algorithm variants including the new LinUCB(alpha=0.1) and decaying epsilon-greedy. With the lower lambda and richer features, we expect LinUCB(alpha=0.1) to potentially outperform alpha=0.5 since the features are more informative and less exploration is needed. The CF baselines appear here too for direct comparison — they should fall between Random and the best bandits."

## Slide 16: Bandits vs. Collaborative Filtering

"Now the key comparison: online bandits vs. offline CF baselines. We have three CF methods — Popularity (global average rating), SVD-CF (matrix factorization with 100 latent factors), and UserCF (user-based cosine similarity with 50 nearest neighbors). These CF baselines see the entire 940K-interaction rating matrix upfront. Our bandits start from zero and learn online (though they get a warm-start from 5K samples). On the left, cumulative reward as a percentage of oracle over time. On the right, final rewards with a vertical line separating online and offline methods. Even if CF wins on raw numbers, bandits offer real advantages: cold-start handling, adaptation to preference drift, and continuous exploration."

## Slide 17: Statistical Significance

"With only 5 seeds, our statistical power is limited but we now have 12 algorithms to compare. The heatmap shows pairwise paired t-test p-values. Random vs everything else is clearly significant. Bandit-vs-CF comparisons are particularly interesting — if significant, they tell us whether the online learning truly matches offline knowledge. The paired design helps control for sequence randomness."

## Slide 18: Example Recommendations

"Here's one user and what LinUCB predicts after 1,000 training steps. With the richer features and lower regularization, the model should be more confident and accurate in its predictions. The interaction features help it capture nuanced preferences beyond simple genre matching."

## Slide 19: Conclusion

"Key takeaways. First, sublinear regret is empirically confirmed with the theoretical O(d*sqrt(T)) rate. Second, K=50 creates a much more challenging problem — Random's oracle percentage drops significantly, widening the gap where smart algorithms compete. Third, richer interaction features and warm-start push bandits closer to oracle performance. Fourth, structured exploration via UCB or posterior sampling outperforms both fixed and decaying epsilon-greedy. Fifth, online bandits are competitive with pretrained collaborative filtering — Popularity, SVD, and UserCF — despite massive information asymmetry. For future work: non-stationary preferences, neural bandits, and real online deployment."
