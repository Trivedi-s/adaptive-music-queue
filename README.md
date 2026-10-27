# Adaptive Music Queue: Reinforcement Learning for Session-Aware Song Recommendation

**CS5100: Foundations of Artificial Intelligence, Group 3**

Jiang Aitong · Shloka Shreyans Trivedi · Aryan Rakeshbhai Shah

> **Status:** Proposal stage. This repository has the project plan only. Code, experiments, and results will be added as work progresses.

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

- A catalog of **~500–2,000 tracks** with genre and audio-feature metadata
- Simulated listening sessions of **15–30 recommendations**
- Tracking of short-term feedback and a longer-term preference profile
- Choosing the next track from a limited candidate set
- Reacting to repeated skips, avoiding too much repetition, and sometimes exploring unfamiliar styles
- Several configurable **simulated listener profiles**, including ones whose preferences change during a session
- A lightweight **interactive demo** where a user gives feedback by hand

**Out of scope:** audio streaming integration.

## AI Formulation

The recommender is modeled as a **Markov Decision Process (MDP)**:

- **State:** recent listening feedback, aggregate song features, current preference estimates
- **Action:** choose the next candidate song, or a high-level style direction
- **Environment:** returns a skip, completion, or like signal and updates the session
- **Reward:** balances positive feedback against early skips, repetition, and diversity

### Methods

- **RL policy:** discretized **Q-learning**, implemented from scratch
- **Baselines:**
  - Content-similarity recommender
  - Simple feedback-driven heuristic
- **Stretch goal:** a **two-level policy**. The upper level decides whether to *persist*, *shift style*, or *explore*. The lower level ranks candidate tracks.

> The project uses no pretrained recommenders and no imported RL libraries.

## Evaluation

The agent trains through repeated interactions with simulated listeners, optionally calibrated with accessible historical listening data. Evaluation uses **held-out listener profiles and session scenarios**.

**Metrics,** compared across fixed random seeds:

- Cumulative reward
- Early-skip rate
- Preference-shift recovery time
- Catalog diversity
- Repetition

**Ablations:**

- Removing session history
- Removing exploration

> Better results in simulation will not be presented as proof of real-world user satisfaction.

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