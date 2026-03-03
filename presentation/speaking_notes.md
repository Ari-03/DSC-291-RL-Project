# Speaking Notes — Contextual Bandits for Anime Recommendation

## Slide 1: Title (0:30)

"Today I'm presenting my project on contextual bandits for anime recommendation. The core question: when recommending anime sequentially, how do we balance exploiting known preferences with exploring potentially better options? I'll show how LinUCB and Thompson Sampling handle this on real MyAnimeList data, and compare against collaborative filtering baselines."

## Slide 2: Problem & Motivation (2:00)

"Why bandits for recommendation? A traditional recommender trains once and deploys a fixed model — if a user watches action anime, it keeps recommending action, creating a filter bubble. Contextual bandits treat each recommendation as a decision under uncertainty. The explore-exploit table captures the tension: exploiting means safe, expected rewards; exploring means discovering better options at the cost of occasional misses. The key insight from our course is that algorithms like LinUCB handle both simultaneously through the UCB principle — no hard tradeoff needed."

## Slide 3: Formulation (2:00)

"This maps directly to the linear contextual bandit framework from lecture. Each round, a user arrives, we present K=50 candidate anime, and observe the rating as reward. The 72-dimensional context vector encodes user, anime, and interaction features. The core assumption is a linear reward model — expected rating equals the dot product of context with unknown theta-star. Our goal is minimizing cumulative pseudo-regret, with a theoretical bound of O(d * sqrt(T) * log T) — sublinear regret that we'll verify empirically."

## Slide 4: Data & Features (1:30)

"I'm using MyAnimeList — 5,000 users, 3,100 anime, 940K interactions split 70/30 per user. Train set provides features and CF baselines; test set provides rewards, preventing data leakage. The 72-dimensional context has user features, anime features including genre encodings, and seven interaction features with polynomial cross-terms. It's a semi-synthetic simulator where we only present anime the user has actually rated, giving us exact regret computation."

## Slide 5: Algorithms (2:00)

"All algorithms share online ridge regression with Sherman-Morrison updates — O(d-squared) per step. Epsilon-Greedy explores randomly with fixed probability — simple but undirected. I also test a decaying variant where epsilon shrinks as 1/sqrt(t). LinUCB adds an optimism bonus that inflates predicted reward in uncertain directions — directed exploration. Thompson Sampling achieves the same effect by sampling from the posterior. Both LinUCB and TS direct exploration toward uncertainty, which is why they should outperform epsilon-greedy."

## Slide 6: Experiment Setup (1:00)

"50,000 rounds, 50 arms per round, 5 seeds for error bars. All algorithms are warm-started with ridge regression on 10,000 offline samples. The critical methodological choice: 70/30 train/test split per user means CF baselines must generalize — they can't memorize answers. All algorithms see the exact same user-candidate sequence per seed for fair comparison."

## Slide 7: Cumulative Reward — % of Oracle (2:00)

"This is cumulative reward as a percentage of the oracle over time. LinUCB and Thompson Sampling converge to the highest percentages. The warm-start gives all learning algorithms a head start. Random stays flat — it never learns. Fixed epsilon-greedy plateaus due to constant exploration cost, while decaying epsilon-greedy improves over time as exploration diminishes. The shaded bands show standard error across 5 seeds."

## Slide 8: Regret Growth Rate — log-log (1:30)

"This is the key slide connecting to theory. On a log-log scale, R(t) = c * t^alpha becomes a straight line with slope alpha. Random has alpha near 1 — linear regret, as expected. LinUCB and TS have alpha around 0.5 to 0.7, confirming sublinear growth matching the O(d*sqrt(T)) bound from lecture. The dashed reference lines show pure sqrt(T) and linear growth for comparison."

## Slide 9: Bandits vs. Collaborative Filtering (1:30)

"The key comparison: online bandits versus offline CF with a fair train/test split. CF baselines — SVD and UserCF — train on 70% of ratings and must generalize to the held-out 30%. On the left, cumulative reward over time; on the right, final rewards with online and offline methods separated. With fair evaluation, bandits are competitive with CF — and they offer real advantages: cold-start handling, adaptation to preference drift, and continuous exploration that CF cannot provide."

## Slide 10: Conclusion & Future Work (1:30)

"The summary table shows final rewards and regret for all algorithms. Key takeaways: sublinear regret is empirically confirmed at the theoretical rate. The train/test split ensures fair CF comparison with no data leakage. Structured exploration via UCB and posterior sampling outperforms epsilon-greedy. And online bandits are competitive with collaborative filtering while offering cold-start and non-stationarity advantages. Future work includes non-stationary bandits, neural reward models, and real online A/B testing."
