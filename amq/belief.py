"""HMM forward filter: the agent's belief over the listener's hidden style.

(README "AI Formulation", Belief update.) The agent cannot see which style cluster z the
listener wants. It keeps a belief b(z) = P(z | feedback so far) and updates it after every
track with the forward algorithm, using its *own* approximate model of a listener:

  transition   stay with prob 1 - rho, else switch to one of the other K-1 clusters
               uniformly (the agent does not know the listener's long-term taste w)
  emission     P(feedback | x, z) = d * P_dislike(feedback)
                                  + (1 - d) * OrderedLogit(feedback | beta * sim(x, mu_z) + bias)

The emission has the same *form* as the simulator's (`feedback_probs`), but the agent's
parameters (rho, beta, bias, d) are guesses that need not match any real listener. The
simulator's popularity, fatigue and novelty terms are left out: the agent cannot know them,
and the selector already avoids repeats.

The dislike term d is what lets the filter tell a single rejected track from a change of
mood: a skip is always partly explained by "they just don't like this song", so one skip
moves the belief only a little, while a run of skips in one style moves it a lot.

Timing. The simulator's z_t judges track t and switches happen between tracks. So after
the feedback on track t the filter does

  update    posterior(z)  ∝ belief(z) · P(feedback_t | x_t, z)      belief over z_t
  predict   belief(z')    = Σ_z posterior(z) · T(z → z')             belief over z_{t+1}

and `belief` is always the prediction for the *next* track, which is what the agent acts on.

Usage:
    bf = BeliefFilter(catalog)          # or BeliefFilter(catalog, ListenerModel(rho=0.1))
    bf.reset()
    choice = selector.select(macro, env.history, bf.belief)
    result = env.step(choice.track)
    bf.update(choice.track, result.feedback)
"""
from dataclasses import dataclass

import numpy as np

from .simulator import CUTPOINTS, DISLIKE_PROBS, feedback_probs


@dataclass(frozen=True)
class ListenerModel:
    """The agent's assumptions about listeners. Defaults are deliberately generic."""
    rho: float = 0.05              # assumed switch probability per track
    beta: float = 6.0              # assumed weight of style fit
    bias: float = -3.0             # assumed base logit
    dislike_rate: float = 0.10     # assumed share of tracks skipped whatever the mood
    cutpoints: tuple = CUTPOINTS
    dislike_probs: tuple = tuple(DISLIKE_PROBS)


class BeliefFilter:
    def __init__(self, catalog, model=None):
        self.catalog = catalog
        self.model = model or ListenerModel()
        K = catalog.n_clusters
        m = self.model
        if not 0.0 <= m.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if not 0.0 <= m.dislike_rate < 1.0:
            raise ValueError("dislike_rate must be in [0, 1)")
        # T[z, z'] = P(z_{t+1} = z' | z_t = z)
        self.transition = np.full((K, K), m.rho / (K - 1)) if K > 1 else np.ones((1, 1))
        np.fill_diagonal(self.transition, 1.0 - m.rho if K > 1 else 1.0)
        self._dislike = np.asarray(m.dislike_probs, dtype=float)
        self.reset()

    def reset(self, prior=None):
        """Start a new session. The default prior is uniform over clusters."""
        K = self.catalog.n_clusters
        self.belief = np.full(K, 1.0 / K) if prior is None else _normalize(np.asarray(prior, float))
        self.posterior = None      # belief over the style that judged the last track
        self.n_updates = 0
        return self.belief

    def emission(self, track):
        """(K, 5) array: P(feedback | track, z) for every cluster z under the agent's model."""
        m = self.model
        z = np.arange(self.catalog.n_clusters)
        logits = m.beta * self.catalog.similarity(int(track), z) + m.bias
        return m.dislike_rate * self._dislike + (1 - m.dislike_rate) * feedback_probs(logits, m.cutpoints)

    def update(self, track, feedback):
        """Fold in the feedback on `track` and return the belief for the next track."""
        likelihood = self.emission(track)[:, int(feedback)]
        self.posterior = _normalize(self.belief * likelihood)
        self.belief = self.posterior @ self.transition
        self.n_updates += 1
        return self.belief

    def run(self, history):
        """Reset and replay a whole history [(track, Feedback)]. Returns the final belief."""
        self.reset()
        for track, feedback in history:
            self.update(track, feedback)
        return self.belief

    @property
    def map_cluster(self):
        """Most likely style for the next track."""
        return int(np.argmax(self.belief))

    @property
    def confidence(self):
        """max_z belief(z): 1/K when clueless, 1 when certain."""
        return float(self.belief.max())


def _normalize(p):
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("belief has no probability mass")
    return p / total
