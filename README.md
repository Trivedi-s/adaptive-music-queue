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

1. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
2. Spotify Sequential Skip Prediction Challenge, *RecSys Challenge 2019* (problem context).