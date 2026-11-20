"""Discretize the agent's belief and recent feedback into a small tabular state.

(README "MDP Components", State.) The RL agent cannot use the continuous belief vector as
a table index, so each step is summarized by five small features:

  feature        values                                 meaning
  in_map         0 = no, 1 = yes                        is the last track's cluster the belief's top cluster?
  confidence     0 = low, 1 = medium, 2 = high          max belief < 1/3, < 2/3, or above
  skip_streak    0, 1, 2, 3 (= 3 or more)               consecutive skips just now
  last_feedback  0 = negative, 1 = neutral, 2 = positive
  phase          0 = early, 1 = mid, 2 = late           first, middle or last third of the session

`last_feedback` encodes the *strength* of the last response, so that it adds to the skip
streak instead of repeating it: early skip = negative; late skip or complete = neutral;
replay or like = positive. Together with the streak this tells early from late skips and
complete from like. Before the first track it is neutral.

That gives 2 * 3 * 4 * 3 * 3 = 216 states. Some combinations cannot happen (for example a
skip streak of 0 with negative feedback), so the agent only visits a subset of them.

The confidence cut points 1/3 and 2/3 are close to the 33rd and 67th percentiles of the
belief confidence seen in simulated sessions, so each level is visited about equally.

Before the first track, the "last track's cluster" is the selector's home cluster.
"""
from dataclasses import dataclass

import numpy as np

from .simulator import Feedback

SHAPE = (2, 3, 4, 3, 3)          # in_map, confidence, skip_streak, last_feedback, phase
N_STATES = int(np.prod(SHAPE))
CONFIDENCE_CUTS = (1 / 3, 2 / 3)
MAX_STREAK = 3

_FEEDBACK_LEVEL = {
    Feedback.EARLY_SKIP: 0,
    Feedback.LATE_SKIP: 1,
    Feedback.COMPLETE: 1,
    Feedback.REPLAY: 2,
    Feedback.LIKE: 2,
}


@dataclass(frozen=True)
class State:
    in_map: int
    confidence: int
    skip_streak: int
    last_feedback: int
    phase: int

    @property
    def index(self):
        return int(np.ravel_multi_index(
            (self.in_map, self.confidence, self.skip_streak, self.last_feedback, self.phase), SHAPE))

    @classmethod
    def from_index(cls, index):
        return cls(*(int(v) for v in np.unravel_index(index, SHAPE)))

    def __str__(self):
        return (f"{'in' if self.in_map else 'off'}-map, "
                f"{('low', 'med', 'high')[self.confidence]} conf, "
                f"streak {self.skip_streak}{'+' if self.skip_streak == MAX_STREAK else ''}, "
                f"last {('neg', 'neutral', 'pos')[self.last_feedback]}, "
                f"{('early', 'mid', 'late')[self.phase]}")


def skip_streak(history):
    n = 0
    for _, feedback in reversed(history):
        if not feedback.is_skip:
            break
        n += 1
    return n


class StateEncoder:
    def __init__(self, catalog, session_length, home_cluster):
        self.catalog = catalog
        self.session_length = session_length
        self.home_cluster = home_cluster

    def encode(self, history, belief):
        """State before choosing the next track, from the history so far and the belief."""
        belief = np.asarray(belief)
        current = self.catalog.cluster[history[-1][0]] if history else self.home_cluster
        top = belief.max()
        t = len(history)
        return State(
            in_map=int(belief[current] >= top),          # ties count as "in"
            confidence=int(np.searchsorted(CONFIDENCE_CUTS, top, side="right")),
            skip_streak=min(skip_streak(history), MAX_STREAK),
            last_feedback=_FEEDBACK_LEVEL[history[-1][1]] if history else 1,
            phase=min(3 * t // self.session_length, 2),
        )
