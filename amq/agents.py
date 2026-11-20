"""Tabular TD agents: Q-learning and SARSA over macro-actions (README "Methods").

Each step the agent
  1. summarizes its belief and recent feedback as a discrete state s (amq/state.py),
  2. picks a macro-action a with epsilon-greedy on Q[s],
  3. lets the low-level selector turn a into a track (amq/selector.py),
  4. after the feedback, updates the HMM belief (amq/belief.py) and Q[s, a]:

       Q-learning (off-policy)  target = r + gamma * max_a' Q[s', a']
       SARSA      (on-policy)   target = r + gamma * Q[s', a'],  a' the action it will take next

     At the end of the session the target is just r.

Agent protocol shared with the evaluation code (amq/evaluate.py):
    agent.reset(seed)                      start a session
    track = agent.act(env)                 choose a catalog row to play
    agent.observe(track, result, learn)    see the StepResult; learn only if `learn`

The TD agent only uses `env.session_length` from the environment. Everything else comes
from its own history of (track, feedback) pairs, so it never sees the hidden state.

Ablation: with use_belief=False the agent keeps no belief. The selector then acts on a
flat belief and the state's in_map/confidence features are constant, so the agent sees
only the skip streak, last feedback and session phase: "learned skip counting".
"""
import numpy as np

from .belief import BeliefFilter
from .selector import N_MACROS, Macro, Selector
from .state import N_STATES, State, StateEncoder


class TDAgent:
    def __init__(self, catalog, method="q_learning", alpha=0.1, gamma=0.9, epsilon=0.1,
                 session_length=25, model=None, use_belief=True, seed=None):
        if method not in ("q_learning", "sarsa"):
            raise ValueError("method must be 'q_learning' or 'sarsa'")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        self.catalog = catalog
        self.method = method
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.use_belief = use_belief
        self.selector = Selector(catalog)
        self.filter = BeliefFilter(catalog, model)
        self.encoder = StateEncoder(catalog, session_length, self.selector.home_cluster)
        self.Q = np.zeros((N_STATES, N_MACROS))
        self.visits = np.zeros((N_STATES, N_MACROS), dtype=int)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---- session ------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.history = []
        self.filter.reset()
        self._state = self._action = self._next_action = None
        self.last_choice = None

    @property
    def belief(self):
        """Belief passed to the selector and the state encoder (flat without a belief)."""
        return self.filter.belief if self.use_belief else np.full(self.catalog.n_clusters,
                                                                  1 / self.catalog.n_clusters)

    def state(self):
        return self.encoder.encode(self.history, self.belief)

    # ---- acting -------------------------------------------------------------

    def greedy(self, s):
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    def policy(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_MACROS))
        return self.greedy(s)

    def act(self, env=None):
        s = self.state().index
        a = self._next_action if self._next_action is not None else self.policy(s)
        self._next_action = None
        self._state, self._action = s, a
        self.last_choice = self.selector.select(Macro(a), self.history,
                                                self.belief if self.use_belief else None)
        return self.last_choice.track

    # ---- learning -----------------------------------------------------------

    def observe(self, track, result, learn=True):
        self.history.append((int(track), result.feedback))
        if self.use_belief:
            self.filter.update(track, result.feedback)
        if not learn:
            return
        s, a = self._state, self._action
        if result.done:
            target = result.reward
        else:
            s2 = self.state().index
            if self.method == "sarsa":
                a2 = self.policy(s2)
                self._next_action = a2
                target = result.reward + self.gamma * self.Q[s2, a2]
            else:
                target = result.reward + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.visits[s, a] += 1

    # ---- inspection ---------------------------------------------------------

    def policy_table(self, min_visits=20):
        """[(State, best macro, Q row, visits)] for states visited at least `min_visits` times."""
        rows = []
        for s in np.flatnonzero(self.visits.sum(axis=1) >= min_visits):
            rows.append((State.from_index(s), Macro(int(np.argmax(self.Q[s]))),
                         self.Q[s].copy(), int(self.visits[s].sum())))
        return rows
