# Adaptive Music Queue: Reinforcement Learning for Session-Aware Song Recommendation

**CS5100: Foundations of Artificial Intelligence, Group 3**

Jiang Aitong · Shloka Shreyans Trivedi · Aryan Rakeshbhai Shah

---

## Overview

Listeners' preferences often change within a single session. A static recommender can keep suggesting similar songs even after the listener has skipped several in a row. This project builds an **adaptive next-song queue** that learns from:

- **Implicit feedback:** early skips, completions, replays
- **Explicit feedback:** likes

The goal is to improve **session-level satisfaction** while balancing familiar music, discovery, and coherent transitions between songs. The main challenge is telling apart a listener **rejecting one specific track** from a listener **temporarily shifting** their preferred genre, mood, or energy.

## Background

| Approach | Idea | Limitation |
|---|---|---|
| Content-based filtering | Rank songs by metadata similarity | Doesn't adapt within a session |
| Collaborative filtering | Use patterns across many listeners | Needs large interaction data |
| Contextual bandits | Adapt to immediate feedback | Doesn't optimize long-term sequences |
| Reinforcement learning | Treat recommendation as sequential decisions | The approach this project explores |

This project is a **small, inspectable, session-aware agent trained from scratch**. It is not a full music streaming platform.

## Existing Methodologies

This section surveys the main families of approaches to music and sequential recommendation, roughly in order of how much they account for the *sequential, interactive* nature of a listening session. For each one we note what it means for this project.

### 1. Content-Based Filtering

Content-based recommenders represent each item by its attributes and recommend items similar to those the user has liked before [2]. In music, the attributes are typically genre and artist tags or audio descriptors such as tempo, energy, valence, danceability and acousticness. Deep models can also learn these representations directly from audio, which helps with the *cold-start* problem for new tracks that have no listening history [3]. Many classical automatic playlist generators work this way: they order tracks so that neighboring songs are acoustically similar, which keeps transitions smooth [4].

*Strengths:* it needs no data from other users, it works on a small catalog, and it is easy to interpret.
*Limitations:* it is static and tends to over-specialize (a "filter bubble"). It has no built-in way to react when a user skips several similar tracks in a row.
*Relevance:* our track metadata and audio features will define the similarity space. A content-similarity recommender is one of our **baselines**.

### 2. Collaborative Filtering

Collaborative filtering (CF) infers preferences from patterns across many users. Matrix factorization learns latent user and item vectors whose dot product predicts preference [6]. Because music data is mostly *implicit* (plays, skips) rather than explicit ratings, the implicit-feedback formulation of Hu et al. [5] treats play counts as confidence-weighted observations, and it became the standard approach for music. Surveys of music recommender systems point out problems specific to music: very large catalogs, short item duration, lots of repeated listening, and strong dependence on context [7].

*Strengths:* it captures taste that metadata cannot describe, and it scales well.
*Limitations:* it needs large interaction datasets. The classical form models long-term taste and ignores order and context within a session.
*Relevance:* with no real user population, we will not train CF directly. However, the idea of a **long-term preference profile** carries over to our listener state.

### 3. Session-Based and Sequential Recommendation

Session-based recommenders predict the next item from the sequence of the current session. GRU4Rec [8] was the first to apply recurrent neural networks to this task. SASRec [9] later used self-attention to model which past items matter for the next one. At Spotify, CoSeRNN [11] showed that recent consumption and session context (time of day, device) predict what a user streams next, and it models a user as a *sequence of session-level embeddings*. This matches the long-term vs. short-term split in our design. A large benchmark study by Ludewig and Jannach [10] found that simple nearest-neighbor and rule-based session methods often match or beat deep models.

*Strengths:* they model order and short-term intent explicitly.
*Limitations:* these are *supervised predictors* of the next item. They imitate logged behavior rather than choose actions that optimize future feedback, and they need large logs to train.
*Relevance:* [10] supports keeping strong **heuristic baselines** (e.g., "after k skips, switch genre"). Our session state can borrow the recent-history representation these models use.

### 4. Multi-Armed and Contextual Bandits

Bandit algorithms address the **exploration–exploitation** trade-off directly. Contextual bandits such as LinUCB [12] choose an arm (item) based on a context vector and update after each reward. Thompson sampling [13] is a Bayesian alternative that performs well in practice. In music, Wang et al. [14] framed interactive music recommendation as a bandit problem with a Bayesian model that combines audio content with a **novelty** term, so recently played songs lose value and recover over time. Spotify's BaRT [15] used contextual bandits in production to decide when to explore versus exploit on the home page.

A key point for us is **non-stationarity**. Standard bandits assume fixed reward distributions, but a listener whose mood changes mid-session violates that assumption. Discounted-UCB and sliding-window UCB [16] down-weight older observations so they can track rewards that change abruptly.

*Strengths:* they adapt online, explore in a principled way, are sample-efficient, and are simple to implement.
*Limitations:* they are *myopic*. They maximize immediate reward and do not reason about how the current song changes future states (e.g., a "bridge" song that sets up a style transition).
*Relevance:* a contextual bandit, possibly with a sliding window, is a natural **intermediate baseline** between heuristics and full RL. Comparing them tells us whether planning over multiple steps adds anything.

### 5. Reinforcement Learning and MDP-Based Recommenders

Treating recommendation as a Markov Decision Process goes back at least to Shani et al. [17], who used n-gram purchase histories as states and solved for a policy that maximizes long-term value. The general machinery (value functions, temporal-difference learning, Q-learning [18]) is covered in Sutton and Barto [1]. Afsar et al. [22] survey modern RL-based recommenders. Notable examples include:

- **DEERS** [19], which models positive and negative feedback (e.g., skips) as *separate* state components. This idea fits our problem directly, because skips far outnumber likes and could otherwise drown out the positive signal.
- **Top-K off-policy REINFORCE** [20], from YouTube, which trains a policy-gradient recommender from logged data with importance-weight correction.
- **SlateQ** [21], which decomposes the value of a recommended *set* into per-item Q-values, making large action spaces tractable.

Hierarchical RL, in particular the *options* framework [34], lets an agent choose temporally extended actions such as "stay in this style" or "transition toward style X" and then pick specific tracks within them. This is the hierarchical decision-making described in our proposal.

*Strengths:* they optimize **long-term, session-level** reward and can learn to take a short-term loss for a later gain.
*Limitations:* they need many interactions, the state and action spaces can grow large, and evaluating them offline is hard.
*Relevance:* our tabular or approximate Q-learning agent belongs here. A small catalog, short sessions and a simulator keep the problem tractable without deep RL.

### 6. Reinforcement Learning for Music Playlists

Two papers are closest to this project:

- **DJ-MC** [23] formulates playlist generation as an MDP whose reward depends on both the individual song *and the transition* from previous songs. Using song features and model-based planning, it learns personalized sequences within a **single 25–50 song session** and outperforms recommenders that score songs independently. The scale and setting are close to ours, so this is the most direct precedent.
- **Spotify's simulation-based RL** [24] trains a modified DQN (AH-DQN) against a *learned user simulator* to generate playlists. It shows that simulator-based offline evaluation correlates with online A/B results. This supports our decision to train and evaluate against simulated listeners.

### 7. Skip Behavior and Implicit Feedback

Skips are the main implicit signal in music streaming. The **Music Streaming Sessions Dataset** [25] (~160M Spotify sessions with skip labels and audio features) was released for the **Spotify Sequential Skip Prediction Challenge** at the WSDM Cup 2019 [26]. That challenge asked participants to predict whether each remaining track in a session would be skipped, given the first half of the session. Meggetto et al. [27] applied deep RL to skip prediction and found that *user-behavior features* (recent skip patterns) predict skips much better than content or context features.

*Relevance:* this supports giving **recent skip/complete history** a central role in our state. It also shows that skip timing matters (an early skip is a stronger negative signal than a late one), which we can use when designing the reward.

### 8. User Simulators for Training and Evaluation

Since online experiments with real users are expensive, simulators have become standard tools for RL recommenders. **RecSim** [28] (Google) is a configurable platform with pluggable user-choice models, latent user state and document models. **RecoGym** [29] simulates organic and bandit feedback for e-commerce recommendation. Both use the same design pattern we need: a hidden user state (e.g., current mood or energy preference) that generates observable feedback and can change over time.

*Relevance:* our listener simulator follows this design. Its profiles have a long-term taste vector, a short-term mood that can **shift mid-session**, and a stochastic feedback model (skip / complete / replay / like).

### 9. Diversity, Novelty, and Repetition

Optimizing for accuracy alone leads to repetitive recommendations. Maximal Marginal Relevance [30] and topic diversification [31] re-rank candidates to balance relevance against redundancy. Kaminskas and Bridge [32] survey "beyond-accuracy" objectives (diversity, novelty, serendipity, coverage). In music, Hansen et al. [33] show that recommendations can shift users toward more diverse listening without hurting satisfaction.

*Relevance:* this motivates our **repetition penalty** and **exploration** terms, and gives us beyond-accuracy metrics (intra-session diversity, novelty) to report alongside skip rate.

### Summary and Research Gap

| Family | Adapts within session | Plans ahead | Explores | Data needs | Example refs |
|---|:---:|:---:|:---:|---|---|
| Content-based | ✗ | ✗ | ✗ | Low (metadata only) | [2], [3], [4] |
| Collaborative filtering | ✗ | ✗ | ✗ | High (many users) | [5], [6] |
| Session-based sequential | ✓ | ✗ | ✗ | High (logged sessions) | [8], [9], [10], [11] |
| Contextual bandits | ✓ | ✗ | ✓ | Low–medium | [12]–[16] |
| RL / MDP | ✓ | ✓ | ✓ | Medium–high (or a simulator) | [17]–[24] |

Most existing work either optimizes long-term taste with large proprietary datasets or predicts skips without acting on the prediction. Bandits adapt quickly but plan only one step ahead. Large-scale RL systems ([20], [24]) depend on infrastructure and data beyond the scope of a course project. The gap this project targets is a **small, inspectable agent** that explicitly separates:

1. *rejection of a single track*, which calls for staying in the same style with a different song, from
2. *a shift in the listener's mood or style*, which calls for a transition,

and is tested against controlled simulated listeners whose ground-truth preference shifts are known. This makes it possible to measure adaptation directly, which real logged data does not allow.

## Scope

**Minimum scope:**

- A catalog of **~500–1,000 tracks** with genre and audio-feature metadata
- Simulated listening sessions of **15–30 recommendations**
- Tracking of short-term feedback and a longer-term preference profile
- Choosing the next track from a limited candidate set
- Reacting to repeated skips, avoiding too much repetition, and sometimes exploring unfamiliar styles
- Several configurable **simulated listener profiles**, including ones whose preferences change during a session
- A lightweight **interactive demo** where a user gives feedback by hand

**Out of scope:** audio streaming integration.

## AI Formulation

TBD

### Methods

TBD

## Evaluation

TBD

## Roadmap

- [ ] Track catalog and metadata preparation
- [ ] Listener simulator and listener profiles
- [ ] Baseline recommenders (content-similarity and heuristic)
- [ ] Q-learning agent
- [ ] Controlled experiments and ablations
- [ ] Evaluation plots
- [ ] Interactive demo

## Deliverables

- Reproducible code
- Evaluation plots
- A short demo

## References

**Foundations**

1. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.

**Content-based and collaborative filtering**

2. M. J. Pazzani and D. Billsus, "Content-Based Recommendation Systems," in *The Adaptive Web*, LNCS 4321, Springer, 2007, pp. 325–341.
3. A. van den Oord, S. Dieleman, and B. Schrauwen, "Deep Content-Based Music Recommendation," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2013.
4. G. Bonnin and D. Jannach, "Automated Generation of Music Playlists: Survey and Experiments," *ACM Computing Surveys*, vol. 47, no. 2, 2014.
5. Y. Hu, Y. Koren, and C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," in *Proc. IEEE ICDM*, 2008.
6. Y. Koren, R. Bell, and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," *IEEE Computer*, vol. 42, no. 8, 2009.
7. M. Schedl, H. Zamani, C.-W. Chen, Y. Deldjoo, and M. Elahi, "Current Challenges and Visions in Music Recommender Systems Research," *International Journal of Multimedia Information Retrieval*, vol. 7, no. 2, 2018.

**Session-based and sequential recommendation**

8. B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk, "Session-based Recommendations with Recurrent Neural Networks," in *Proc. ICLR*, 2016.
9. W.-C. Kang and J. McAuley, "Self-Attentive Sequential Recommendation," in *Proc. IEEE ICDM*, 2018.
10. M. Ludewig and D. Jannach, "Evaluation of Session-based Recommendation Algorithms," *User Modeling and User-Adapted Interaction*, vol. 28, pp. 331–390, 2018. [arXiv:1803.09587](https://arxiv.org/abs/1803.09587)
11. C. Hansen, C. Hansen, L. Maystre, R. Mehrotra, B. Brost, F. Tomasi, and M. Lalmas, "Contextual and Sequential User Embeddings for Large-Scale Music Recommendation," in *Proc. ACM RecSys*, 2020. [doi:10.1145/3383313.3412248](https://doi.org/10.1145/3383313.3412248)

**Bandits**

12. L. Li, W. Chu, J. Langford, and R. E. Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation," in *Proc. WWW*, 2010.
13. O. Chapelle and L. Li, "An Empirical Evaluation of Thompson Sampling," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2011.
14. X. Wang, Y. Wang, D. Hsu, and Y. Wang, "Exploration in Interactive Personalized Music Recommendation: A Reinforcement Learning Approach," *ACM Trans. Multimedia Computing, Communications, and Applications*, vol. 11, no. 1, 2014. [arXiv:1311.6355](https://arxiv.org/abs/1311.6355)
15. J. McInerney, B. Lacker, S. Hansen, K. Higley, H. Bouchard, A. Gruson, and R. Mehrotra, "Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits," in *Proc. ACM RecSys*, 2018.
16. A. Garivier and E. Moulines, "On Upper-Confidence Bound Policies for Switching Bandit Problems," in *Proc. ALT*, LNCS 6925, 2011, pp. 174–188. [arXiv:0805.3415](https://arxiv.org/abs/0805.3415)

**Reinforcement learning for recommendation**

17. G. Shani, D. Heckerman, and R. I. Brafman, "An MDP-Based Recommender System," *Journal of Machine Learning Research*, vol. 6, 2005.
18. C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning*, vol. 8, pp. 279–292, 1992.
19. X. Zhao, L. Zhang, Z. Ding, L. Xia, J. Tang, and D. Yin, "Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning," in *Proc. ACM KDD*, 2018. [arXiv:1802.06501](https://arxiv.org/abs/1802.06501)
20. M. Chen, A. Beutel, P. Covington, S. Jain, F. Belletti, and E. H. Chi, "Top-K Off-Policy Correction for a REINFORCE Recommender System," in *Proc. ACM WSDM*, 2019.
21. E. Ie, V. Jain, J. Wang, S. Narvekar, R. Agarwal, R. Wu, H.-T. Cheng, T. Chandra, and C. Boutilier, "SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets," in *Proc. IJCAI*, 2019.
22. M. M. Afsar, T. Crump, and B. Far, "Reinforcement Learning Based Recommender Systems: A Survey," *ACM Computing Surveys*, vol. 55, no. 7, 2022.

**Reinforcement learning for music**

23. E. Liebman, M. Saar-Tsechansky, and P. Stone, "DJ-MC: A Reinforcement-Learning Agent for Music Playlist Recommendation," in *Proc. AAMAS*, 2015. [arXiv:1401.1880](https://arxiv.org/abs/1401.1880)
24. F. Tomasi, J. Cauteruccio, S. Kanoria, K. Ciosek, M. Rinaldi, and Z. Dai, "Automatic Music Playlist Generation via Simulation-based Reinforcement Learning," in *Proc. ACM KDD*, 2023. [arXiv:2310.09123](https://arxiv.org/abs/2310.09123)

**Skip behavior and datasets**

25. B. Brost, R. Mehrotra, and T. Jehan, "The Music Streaming Sessions Dataset," in *Proc. The Web Conference (WWW)*, 2019. [arXiv:1901.09851](https://arxiv.org/abs/1901.09851)
26. Spotify Sequential Skip Prediction Challenge, *WSDM Cup 2019*. [AIcrowd](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge) (problem context).
27. F. Meggetto, C. Revie, J. Levine, and Y. Moshfeghi, "Why People Skip Music? On Predicting Music Skips using Deep Reinforcement Learning," in *Proc. ACM CHIIR*, 2023. [arXiv:2301.03881](https://arxiv.org/abs/2301.03881)

**User simulation**

28. E. Ie, C.-W. Hsu, M. Mladenov, V. Jain, S. Narvekar, J. Wang, R. Wu, and C. Boutilier, "RecSim: A Configurable Simulation Platform for Recommender Systems," arXiv:1909.04847, 2019. [arXiv](https://arxiv.org/abs/1909.04847)
29. D. Rohde, S. Bonner, T. Dunlop, F. Vasile, and A. Karatzoglou, "RecoGym: A Reinforcement Learning Environment for the Problem of Product Recommendation in Online Advertising," arXiv:1808.00720, 2018.

**Diversity and novelty**

30. J. Carbonell and J. Goldstein, "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries," in *Proc. ACM SIGIR*, 1998.
31. C.-N. Ziegler, S. M. McNee, J. A. Konstan, and G. Lausen, "Improving Recommendation Lists Through Topic Diversification," in *Proc. WWW*, 2005.
32. M. Kaminskas and D. Bridge, "Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems," *ACM Trans. Interactive Intelligent Systems*, vol. 7, no. 1, 2016.
33. C. Hansen, R. Mehrotra, C. Hansen, B. Brost, L. Maystre, and M. Lalmas, "Shifting Consumption towards Diverse Content on Music Streaming Platforms," in *Proc. ACM WSDM*, 2021. [doi:10.1145/3437963.3441775](https://doi.org/10.1145/3437963.3441775)

**Hierarchical RL**

34. R. S. Sutton, D. Precup, and S. Singh, "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning," *Artificial Intelligence*, vol. 112, pp. 181–211, 1999.