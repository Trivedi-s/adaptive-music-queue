"""Run sessions, train TD agents, and evaluate any agent on held-out listeners.

Every agent follows one protocol:
    agent.reset(seed)                      start a session
    track = agent.act(env)                 choose a catalog row to play
    agent.observe(track, result, learn)    see the StepResult

Seeds. Training sessions use listener seeds 0, 1, 2, ...; evaluation uses EVAL_SEED_BASE + k,
so agents are always scored on listeners they never trained on. Because the simulator
pre-samples each listener's hidden path, every agent evaluated on the same seeds faces the
same listeners and the same preference shifts (paired comparison).

Reference policies (README "Baselines"):
    RandomAgent         a random unplayed track (amq/bandit_baselines.py)
    StayAgent           content similarity: always STAY
    SkipStreakAgent     STAY, but EXPLORE after k consecutive skips (raw skip counting)
    BanditAgent         sliding-window epsilon-greedy over clusters (amq/bandit_baselines.py)
    BeliefGreedyAgent   always FOLLOW_BELIEF with the HMM belief (belief, no learning)
    OracleAgent         FOLLOW_BELIEF given the true hidden style (upper bound, not deployable)
"""
import numpy as np

from .bandit_baselines import RandomBaseline, SlidingWindowBandit
from .belief import BeliefFilter
from .selector import Macro, Selector
from .simulator import ListenerEnv
from .state import skip_streak

EVAL_SEED_BASE = 1_000_000


# ---- running sessions ---------------------------------------------------------

def run_episode(env, agent, seed, learn=False):
    """Play one session. Returns (total reward, number of skips)."""
    env.reset(seed=seed)
    agent.reset(seed=seed)
    total, skips = 0.0, 0
    while not env.done:
        track = agent.act(env)
        result = env.step(track)
        agent.observe(track, result, learn=learn)
        total += result.reward
        skips += result.feedback.is_skip
    return total, skips


def evaluate(agent, envs, n_seeds=100, seed_base=EVAL_SEED_BASE):
    """Mean session return per profile on held-out listeners, with exploration off."""
    saved = getattr(agent, "epsilon", None)
    if saved is not None:
        agent.epsilon = 0.0
    try:
        out = {}
        for name, env in envs.items():
            returns = [run_episode(env, agent, seed_base + k)[0] for k in range(n_seeds)]
            out[name] = float(np.mean(returns))
        return out
    finally:
        if saved is not None:
            agent.epsilon = saved


def make_envs(catalog, profiles, session_length=25):
    return {name: ListenerEnv(catalog, name, session_length=session_length) for name in profiles}


def epsilon_schedule(episode, n_episodes, start=1.0, end=0.05, decay_fraction=0.6):
    """Linear decay from `start` to `end` over the first `decay_fraction` of training."""
    frac = min(episode / max(decay_fraction * n_episodes, 1), 1.0)
    return start + frac * (end - start)


def train(agent, envs, n_episodes, eval_every=1000, eval_seeds=100, seed=0, log=None):
    """Train on a uniform mix of the given profiles. Returns the learning curve
    [(episodes trained, {profile: mean held-out return})], starting at 0 episodes."""
    rng = np.random.default_rng(seed)
    names = list(envs)
    curve = [(0, evaluate(agent, envs, eval_seeds))]
    for episode in range(n_episodes):
        agent.epsilon = epsilon_schedule(episode, n_episodes)
        env = envs[names[rng.integers(len(names))]]
        run_episode(env, agent, seed=seed * n_episodes + episode, learn=True)
        if (episode + 1) % eval_every == 0:
            curve.append((episode + 1, evaluate(agent, envs, eval_seeds)))
            if log:
                log(curve[-1])
    return curve


# ---- reference policies -------------------------------------------------------

class _SelectorAgent:
    """Base for fixed rules built on the shared selector (no learning)."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.selector = Selector(catalog)

    def reset(self, seed=None):
        self.history = []

    def observe(self, track, result, learn=False):
        self.history.append((int(track), result.feedback))


class StayAgent(_SelectorAgent):
    def act(self, env=None):
        return self.selector.select(Macro.STAY, self.history).track


class SkipStreakAgent(_SelectorAgent):
    def __init__(self, catalog, k=2):
        super().__init__(catalog)
        self.k = k

    def act(self, env=None):
        macro = Macro.EXPLORE if skip_streak(self.history) >= self.k else Macro.STAY
        return self.selector.select(macro, self.history).track


class BeliefGreedyAgent(_SelectorAgent):
    def __init__(self, catalog, model=None):
        super().__init__(catalog)
        self.filter = BeliefFilter(catalog, model)

    def reset(self, seed=None):
        super().reset(seed)
        self.filter.reset()

    def act(self, env=None):
        return self.selector.select(Macro.FOLLOW_BELIEF, self.history, self.filter.belief).track

    def observe(self, track, result, learn=False):
        super().observe(track, result, learn)
        self.filter.update(track, result.feedback)


class OracleAgent(_SelectorAgent):
    """Reads the hidden style from env.listener: an upper bound, never a real agent."""

    def act(self, env):
        belief = np.zeros(self.catalog.n_clusters)
        belief[env.listener.z_path[env.t]] = 1.0
        return self.selector.select(Macro.FOLLOW_BELIEF, self.history, belief).track


class RandomAgent:
    def __init__(self, catalog):
        self.inner = RandomBaseline()

    def reset(self, seed=None):
        self.inner.reset(seed)

    def act(self, env):
        return self.inner(env)

    def observe(self, track, result, learn=False):
        pass


class BanditAgent:
    def __init__(self, catalog, epsilon=0.1, window_size=10):
        self.inner = SlidingWindowBandit(catalog, epsilon=epsilon, window_size=window_size)

    def reset(self, seed=None):
        self.inner.reset(seed)

    def act(self, env):
        return self.inner(env)

    def observe(self, track, result, learn=False):
        self.inner.update(track, result.reward)


REFERENCE_AGENTS = {
    "random": RandomAgent,
    "content similarity (STAY)": StayAgent,
    "skip-streak k=2": SkipStreakAgent,
    "sliding-window bandit": BanditAgent,
    "belief greedy (no learning)": BeliefGreedyAgent,
    "oracle": OracleAgent,
}
