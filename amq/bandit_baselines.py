"""Random and sliding-window epsilon-greedy reference agents.

Call agent(env) to select a catalog row index. For the bandit, call
agent.update(track, result.reward) after env.step(track). Call reset() at the
start of each session. Only observed rewards are used, never hidden listener state.

Example:
    agent = SlidingWindowBandit(catalog, seed=42)
    env.reset(seed=42)
    agent.reset(seed=42)
    while not env.done:
        track = agent(env)
        result = env.step(track)
        agent.update(track, result.reward)

For RandomBaseline the loop is identical except no update() is needed.

"""
from collections import deque
import math
import operator
import random


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class RandomBaseline:
    """Uniformly sample from all unplayed tracks, rather than uniform clusters."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

    def __call__(self, env):
        played = set(env.played)
        available = [i for i in range(env.catalog.n_tracks) if i not in played]
        return self.rng.choice(available) if available else None


class SlidingWindowBandit:
    """Estimate cluster rewards from the last window_size interactions.

    The window is global across clusters, so old evidence expires even for an
    unselected cluster. Available clusters with no evidence are tried first.
    Otherwise epsilon explores uniformly over available clusters; exploitation
    maximizes the window mean with random tie-breaking. Within the selected
    cluster an unplayed track is chosen uniformly. This isolates the cluster
    adaptation rule from content-based ranking.

    Defaults epsilon=.1 and window_size=10 are initial choices, not tuned values.
    """

    def __init__(self, catalog, epsilon=0.1, window_size=10, seed=None):
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1.")
        self.catalog = catalog
        self.epsilon = epsilon
        self.history = deque(maxlen=_positive_integer(window_size, "window_size"))
        self.rng = random.Random(seed)

    def reset(self, seed=None):
        self.history.clear()
        if seed is not None:
            self.rng.seed(seed)

    def update(self, track, reward):
        track = operator.index(track)
        reward = float(reward)
        if not 0 <= track < self.catalog.n_tracks:
            raise ValueError("track is outside the catalog.")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite.")
        self.history.append((int(self.catalog.cluster[track]), reward))

    def __call__(self, env):
        if env.catalog is not self.catalog:
            raise ValueError("Bandit and environment must use the same catalog.")
        played = set(env.played)
        candidates = {
            z: [int(i) for i in members if i not in played]
            for z, members in enumerate(self.catalog.members)
        }
        available = [z for z, tracks in candidates.items() if tracks]
        if not available:
            return None

        rewards = {z: [] for z in available}
        for z, reward in self.history:
            if z in rewards:
                rewards[z].append(reward)
        unseen = [z for z in available if not rewards[z]]
        if unseen:
            cluster = self.rng.choice(unseen)
        elif self.rng.random() < self.epsilon:
            cluster = self.rng.choice(available)
        else:
            means = {z: sum(rewards[z]) / len(rewards[z]) for z in available}
            best = max(means.values())
            cluster = self.rng.choice([z for z in available if means[z] == best])
        return self.rng.choice(candidates[cluster])
