# Speaking Notes — Contextual Bandits for Anime Recommendation

## Slide 1: Title

"Today I'm presenting my project on contextual bandits for anime recommendation. The core idea is simple: when you're recommending anime to users sequentially, you face a fundamental tradeoff — do you exploit what you already know works, or explore something new that might be even better? I'll show how algorithms from our lecture — LinUCB, Thompson Sampling — handle this tradeoff on real MyAnimeList data."

## Slide 2: Problem & Motivation

"Why bandits for recommendation? Think about it — a traditional recommender trains on historical data and deploys a fixed model. If a user mostly watches action anime, it keeps recommending action. It never discovers that this user might also love slice-of-life shows. That's the filter bubble problem. Contextual bandits break this by treating each recommendation as a decision under uncertainty. The explore-exploit table here captures the tension: exploiting means safe, expected rewards; exploring means potentially discovering much better options at the cost of occasional bad recommendations. The key insight from our course is that you don't have to choose — algorithms like LinUCB do both simultaneously through the UCB principle."

## Slide 3: Formulation

"Here's the formal setup, which maps directly to the linear contextual bandit framework from lecture 2. Each round, a user arrives, we present K=20 candidate anime, and we observe the user's rating as reward. The context vector phi(user, anime) is 67-dimensional — I'll detail these features on the next slide. The core assumption is a linear reward model: the expected rating is the dot product of the context with an unknown parameter theta-star. Our goal is to minimize cumulative pseudo-regret — the sum of gaps between the best possible arm and the arm we actually chose. The theoretical bound from lecture is O(d * sqrt(T) * log T), which means regret should grow sublinearly — we'll test this empirically."

## Slide 4: Data & Features

"I'm using the MyAnimeList dataset — 5,000 users, about 3,100 anime, and 940K rating interactions. The 67-dimensional context vector has three blocks: user features like their average score and activity level, anime features including genre multi-hot encodings and metadata, and two interaction features — genre cosine similarity between the user's profile and the anime, and score deviation. The bias term makes the model affine. Crucially, this is a semi-synthetic simulator: at each round, I only present anime the user has actually rated in the dataset. This means I have the exact ground-truth reward for every possible arm, so I can compute exact regret — something you can't do in a real deployment."

## Slide 5: Algorithms

"All three algorithms share the same backbone: online ridge regression with Sherman-Morrison rank-1 updates, which gives us O(d-squared) per step — that's about 4,500 operations per round, very fast. Epsilon-Greedy is the simplest: flip a coin, explore with probability epsilon, exploit otherwise. It's simple but its exploration is undirected — it doesn't know where the uncertainty is. LinUCB from Li et al. adds an optimism bonus: the UCB term alpha times sqrt(x^T A-inverse x) inflates the predicted reward in uncertain directions. This is the 'optimism in the face of uncertainty' principle from lecture 2. Thompson Sampling takes a Bayesian approach: sample theta from the posterior and act greedily on the sample. Both LinUCB and TS direct exploration toward where they're uncertain, which is why they should theoretically outperform epsilon-Greedy."

## Slide 6: Experiment Setup

"50,000 rounds, 20 arms per round, 5 random seeds for statistical reliability. Continuous reward — the rating divided by 10, so rewards are in [0,1]. Critical design choice: all algorithms see the exact same user-candidate sequence per seed. This paired design means we can attribute performance differences to the algorithm, not to sequence randomness. I'm testing two hyperparameter values per algorithm to understand sensitivity."

## Slide 7: Why Does Random Achieve ~81% of Oracle?

"Before diving into algorithm results, let me address something that might seem surprising: the Random baseline achieves about 81% of oracle performance. Is that a problem? No — it's a property of the data. Look at the rating distribution: it's heavily left-skewed, with most ratings clustering between 6 and 9. The mean rating is about 7.6 out of 10. When you randomly pick one of 20 candidates, the expected rating is close to the population mean, while the oracle picks the maximum. A Monte Carlo simulation confirms: E[mean of 20 samples] / E[max of 20 samples] is approximately 81%. So the 'easy' 81% comes for free from the data distribution. The interesting question is: can algorithms capture the remaining 19% gap? That's where exploration-exploitation matters."

## Slide 8: Results — Cumulative Reward (% of Oracle)

"This shows cumulative reward as a percentage of the oracle — the best possible algorithm that always picks the highest-rated anime. Random achieves about 81% of oracle performance — as we just explained, that's a property of the data distribution. But LinUCB and Thompson Sampling reach about 91-92%, a meaningful 10+ percentage point gap. Epsilon-Greedy lands in between. The key observation: the smart algorithms' curves are still rising at T=50K, suggesting they'd continue to improve with more data."

## Slide 9: Results — Regret & Learning Dynamics

"On the left, normalized regret — average regret per round — tells us the learning rate. Random stays flat at about 0.185 regret per round — it never learns. All other algorithms decrease, confirming sublinear cumulative regret. LinUCB and TS decrease fastest. On the right, the sliding-window average reward shows the same story from the reward perspective: smart algorithms converge toward higher per-round rewards. The gap between LinUCB/TS and epsilon-Greedy widens over time — that's the residual exploration cost of undirected epsilon-exploration."

## Slide 10: Regret Growth Rate Analysis (log-log)

"This is the money slide for connecting to theory. On a log-log scale, R(t) = c * t^alpha becomes a straight line with slope alpha. The table shows the fitted exponents. Random has alpha near 1 — linear regret, as expected for a policy that never learns. LinUCB and TS have alpha around 0.5 to 0.7, confirming sublinear growth. The theoretical bound from lecture predicts alpha = 0.5 for the O(d*sqrt(T)) rate. Our empirical values are slightly higher — this is expected because the bound assumes worst-case and our feature dimension d=67 is moderately large. The reference lines show what pure sqrt(T) and linear growth look like."

## Slide 11: Instantaneous Regret Over Time

"This is my favorite visualization because it directly shows whether an algorithm is learning. Per-round regret is noisy, so I smooth with a 1,000-round moving average. Random is flat at about 0.185 — zero learning. LinUCB and TS clearly decrease over time, meaning they're selecting increasingly better arms. Epsilon-Greedy is interesting: it decreases initially but then plateaus above zero. That plateau is the epsilon-exploration cost — even after the model is well-trained, it still randomly explores 5-10% of the time. This is the core weakness of undirected exploration. From lecture: for R(T) = o(T) we need instantaneous regret converging to zero, which only LinUCB and TS achieve."

## Slide 12: Learned Feature Weights

"What does LinUCB actually learn? These are the top-15 positive and bottom-5 negative weights from theta-hat after 50K rounds. The interaction features dominate: genre cosine similarity has the highest positive weight — if a user's genre profile matches the anime, predict a higher rating. Score deviation (user_mean - anime_score) is also important. Among genres, we see that certain genres like 'Thriller' or 'Psychological' correlate with higher predicted ratings in our population. The anime score feature also contributes positively — popular anime tend to get higher ratings. This is the ridge regression solution A_inverse times b, converging to theta-star as we discussed in lecture 2."

## Slide 13: Confidence Ellipsoid Shrinkage

"This visualizes the core mechanism behind LinUCB. On the left, trace of A-inverse — a scalar summary of total uncertainty — drops rapidly in the first few thousand rounds as the algorithm gathers diverse data, then continues decreasing more slowly. On log scale you can see it's still shrinking at 50K rounds. On the right, the average UCB bonus term sqrt(x^T A-inverse x) — this is what LinUCB actually adds to its predictions. It shrinks from about 0.4 initially to under 0.02 by round 50K. This is the confidence ellipsoid from lecture 2/3 contracting: the set of plausible theta values shrinks as A_t grows, and LinUCB's exploration bonus shrinks with it. The transition from exploration to exploitation is gradual, not a hard switch."

## Slide 14: Regret Distribution

"Instead of reward distributions, I'm showing per-round regret distributions — the gap between the oracle and each algorithm. This makes the differences more dramatic. Random has a wide distribution centered around 0.185 — high and variable regret. LinUCB and TS have distributions concentrated near zero with thin right tails — they mostly pick near-optimal arms. The key visual: the mass of the LinUCB/TS distributions is compressed near zero, while Random spreads across the full range. This connects to the sub-Gaussian concentration from lecture 1."

## Slide 15: Hyperparameter Sensitivity

"This bar chart shows final cumulative regret for all seven variants. The story is clear: within each algorithm family, lower exploration parameters win — alpha=0.5 beats alpha=1.0 for LinUCB, v=0.1 beats v=0.5 for TS, epsilon=0.05 beats epsilon=0.1. This makes sense: our 67-dimensional feature space with 50K rounds gives enough data that aggressive exploration hurts more than it helps. The theoretical regret bound scales linearly with alpha for LinUCB: R(T) = O(alpha * d * sqrt(T)). Across families: LinUCB and TS are statistically indistinguishable and both clearly beat epsilon-Greedy, which beats Random. Structured exploration wins."

## Slide 16: Bandits vs. Collaborative Filtering

"Now the question everyone is thinking: why not just use collaborative filtering? This slide compares our online bandits against three offline CF baselines — Popularity (global average rating), SVD-CF (matrix factorization with 50 latent factors), and UserCF (user-based cosine similarity with 50 nearest neighbors). The critical caveat: these CF baselines see the entire 940K-interaction rating matrix upfront. Our bandits start from zero and learn online. So this is an intentionally unfair comparison. On the left, you can see how cumulative reward as a percentage of oracle evolves over time. On the right, the bar chart shows final rewards with online methods on the left and offline baselines on the right of the dashed line. The key finding here is whether our best bandits — LinUCB and TS — can match or approach CF despite this massive information disadvantage. Even if CF wins on raw numbers, bandits offer real advantages: they handle cold-start users naturally, they adapt to preference drift, and they continuously explore to discover new content. In production, you'd likely want a hybrid: CF for warm-start, bandits for ongoing personalization."

## Slide 17: Statistical Significance

"Being honest about statistics: with only 5 seeds, our statistical power is limited. This heatmap shows pairwise paired t-test p-values. Green means significant (p<0.05), red means not. Random vs everything else is clearly significant — no surprise. But can we distinguish LinUCB(alpha=0.5) from TS(v=0.1)? Probably not — they're too close with only 5 seeds. The paired design helps: since all algorithms see the same sequence per seed, the pairing controls for sequence randomness. If I had 20+ seeds, I could tease apart finer differences. This is an honest limitation."

## Slide 18: Example Recommendations

"Let's make this concrete. Here's one user — Dullahan — and what LinUCB predicts after just 1,000 training steps. The top-5 predicted anime include Gintama (a popular, well-rated series) and Fullmetal Alchemist: Brotherhood (actual rating 10, predicted 0.757). The bottom-5 are lower-rated, niche titles. The model has already learned the basic pattern: popular anime matching the user's genre preferences get higher predicted scores. With more training, these predictions would sharpen further."

## Slide 19: Conclusion

"Five key takeaways. First, sublinear regret is empirically confirmed — the log-log analysis shows growth rates matching the O(d*sqrt(T)) theoretical bound. Second, we can visualize the theory: the confidence ellipsoid shrinks over time, driving LinUCB from exploration to exploitation. Third, the learned weights are interpretable — genre overlap and anime quality drive recommendations. Fourth, structured exploration via UCB or posterior sampling consistently outperforms undirected epsilon-exploration. Fifth, online bandits are competitive with pretrained collaborative filtering despite starting from scratch — and they offer cold-start and non-stationarity advantages that CF cannot. For future work: handling non-stationary preferences, scaling beyond linear models with neural bandits, and real online deployment."
