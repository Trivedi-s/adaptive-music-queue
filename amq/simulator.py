"""HMM listener simulator (README "Environment: HMM Listener Simulator").

Each simulated listener is a hidden Markov model:

  hidden state   z_t in {0..K-1}: the style cluster the listener wants right now
  profile        w: distribution over clusters (long-term taste), drawn per listener
  transition     z_{t+1} = z_t with prob 1 - rho, else a *different* cluster drawn from w
                 (Drifter: only from the nearest clusters; Abrupt shifter: forced switch
                 at `shift_step`)
  emission       logit = beta * sim(x, mu_z) + lam * popularity(x) - eta * fatigue(x) + bias
                 P(listen | x, z) = sigmoid(logit)
  feedback       ordered logit: s = logit + Logistic noise, cut at `cutpoints` into
                 early skip < late skip < complete < replay < like.  Because the
                 complete/late-skip cut is at 0, P(complete or better) = sigmoid(logit).
  dislikes       a random subset of tracks (dislike_rate) is skipped whatever the mood

The hidden path z_0..z_{L-1} does not depend on what the agent plays, so it is sampled
up front in `reset()`. With the same seed every agent faces the same listener and the same
preference shifts, which makes agent comparisons paired.

Agents may use only what the listener reveals: the feedback and reward of each step, and
the session history (`env.history`). `env.listener` and `StepResult.info` hold the ground
truth (hidden z, dislikes) and are meant for the Oracle baseline and the evaluation metrics.

Usage:
    from amq import Catalog, ListenerEnv, PROFILES
    env = ListenerEnv(Catalog.load(), PROFILES["abrupt"])
    env.reset(seed=0)
    while not env.done:
        result = env.step(track_index)     # -> StepResult(feedback, reward, done, info)
"""
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Feedback(IntEnum):
    EARLY_SKIP = 0
    LATE_SKIP = 1
    COMPLETE = 2
    REPLAY = 3
    LIKE = 4

    @property
    def is_skip(self):
        return self <= Feedback.LATE_SKIP

    @property
    def is_positive(self):
        return self >= Feedback.COMPLETE


N_FEEDBACK = len(Feedback)

# Cut points on s = logit + noise between consecutive outcomes. The cut at 0 between
# LATE_SKIP and COMPLETE makes P(positive) exactly sigmoid(logit).
CUTPOINTS = (-1.0, 0.0, 2.5, 3.5)

# Feedback distribution for a track on the listener's dislike list, whatever their mood.
DISLIKE_PROBS = np.array([0.8, 0.2, 0.0, 0.0, 0.0])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def feedback_probs(logit, cutpoints=CUTPOINTS):
    """P(feedback | logit) under the ordered-logit model, shape (..., 5).

    Exposed so the agent's HMM filter can use the same emission form with its own
    (approximate) parameters.
    """
    logit = np.asarray(logit, dtype=float)[..., None]
    cdf = sigmoid(np.asarray(cutpoints) - logit)            # P(s < c_k)
    zeros, ones = np.zeros(cdf.shape[:-1] + (1,)), np.ones(cdf.shape[:-1] + (1,))
    return np.diff(np.concatenate([zeros, cdf, ones], axis=-1), axis=-1)


@dataclass(frozen=True)
class ListenerProfile:
    name: str
    rho: object = 0.0              # switch rate per step: a float, or (lo, hi) drawn per listener
    switch: str = "profile"        # "profile": new style drawn from w; "neighbor": from nearest clusters
    n_neighbors: int = 3           # how many nearest clusters a "neighbor" switch can reach
    shift_step: object = None      # if set, force a style switch before track `shift_step` (0-based)
    beta: float = 6.0              # weight of style fit sim(x, mu_z)
    bias: float = -3.0             # b: sets the base positive rate
    lam: float = 1.0               # weight of (popularity - catalog mean); < 0 prefers obscure tracks
    eta: float = 1.0               # weight of fatigue: repeats of the track + same artist recently
    eta_cluster: float = 0.0       # novelty seeking: penalty x share of recent tracks in the same cluster
    dislike_rate: float = 0.10     # share of the catalog the listener always skips
    taste_concentration: float = 0.5   # Dirichlet alpha for w; smaller = more peaked taste
    recent_window: int = 5         # tracks that count as "recent" for fatigue
    cutpoints: tuple = CUTPOINTS


PROFILES = {
    "stable": ListenerProfile("stable"),
    "drifter": ListenerProfile("drifter", rho=(0.05, 0.10), switch="neighbor"),
    "abrupt": ListenerProfile("abrupt", shift_step=10),
    "picky": ListenerProfile("picky", dislike_rate=0.30),
    "explorer": ListenerProfile("explorer", rho=0.10, lam=-1.0, eta_cluster=1.0,
                                taste_concentration=2.0),
}


@dataclass(frozen=True)
class RewardConfig:
    """README "Reward": like +2, replay +1.5, complete +1, late skip -0.5, early skip -1,
    plus -0.5 when the artist was already played in the last `artist_window` tracks."""
    feedback: dict = field(default_factory=lambda: {
        Feedback.EARLY_SKIP: -1.0,
        Feedback.LATE_SKIP: -0.5,
        Feedback.COMPLETE: 1.0,
        Feedback.REPLAY: 1.5,
        Feedback.LIKE: 2.0,
    })
    artist_repeat: float = -0.5
    artist_window: int = 5


@dataclass
class Listener:
    """Ground truth for one sampled listener. Not visible to agents (Oracle/metrics only)."""
    profile: ListenerProfile
    w: np.ndarray            # (K,) long-term taste over clusters
    rho: float
    z_path: np.ndarray       # (L,) hidden style that judges each track of the session
    dislikes: np.ndarray     # (N,) bool, tracks skipped regardless of mood


@dataclass
class StepResult:
    feedback: Feedback
    reward: float
    done: bool
    info: dict               # ground truth for metrics: z, p_positive, disliked, shifted, artist_repeat


class ListenerEnv:
    def __init__(self, catalog, profile, session_length=25, reward=None):
        if isinstance(profile, str):
            profile = PROFILES[profile]
        if profile.shift_step is not None and not 0 < profile.shift_step < session_length:
            raise ValueError("shift_step must fall inside the session")
        self.catalog = catalog
        self.profile = profile
        self.session_length = session_length
        self.reward_config = reward or RewardConfig()
        self._pop_mean = catalog.popularity.mean()
        self.listener = None

    # ---- session lifecycle -------------------------------------------------

    def reset(self, seed=None):
        """Sample a new listener and start a session. Same seed -> same listener and hidden path."""
        taste_seq, path_seq, feedback_seq = np.random.SeedSequence(seed).spawn(3)
        taste_rng, path_rng = np.random.default_rng(taste_seq), np.random.default_rng(path_seq)
        self._rng = np.random.default_rng(feedback_seq)

        p, cat = self.profile, self.catalog
        w = taste_rng.dirichlet(np.full(cat.n_clusters, p.taste_concentration))
        rho = taste_rng.uniform(*p.rho) if isinstance(p.rho, tuple) else float(p.rho)
        dislikes = taste_rng.random(cat.n_tracks) < p.dislike_rate
        self.listener = Listener(p, w, rho, self._sample_path(w, rho, path_rng), dislikes)

        self.t = 0
        self.history = []                                   # [(track, Feedback), ...]
        self.play_count = np.zeros(cat.n_tracks, dtype=int)
        return self.listener

    def _sample_path(self, w, rho, rng):
        p = self.profile
        z = np.empty(self.session_length, dtype=int)
        z[0] = rng.choice(len(w), p=w)
        for t in range(1, self.session_length):
            forced = p.shift_step is not None and t == p.shift_step
            z[t] = self._switch_target(z[t - 1], w, rng) if forced or rng.random() < rho else z[t - 1]
        return z

    def _switch_target(self, z, w, rng):
        if self.profile.switch == "neighbor":
            options = np.argsort(self.catalog.centroid_dist[z])[1:1 + self.profile.n_neighbors]
        else:
            options = np.delete(np.arange(len(w)), z)
        weights = w[options] + 1e-12
        return rng.choice(options, p=weights / weights.sum())

    @property
    def done(self):
        return self.listener is not None and self.t >= self.session_length

    @property
    def recent_tracks(self):
        return [track for track, _ in self.history[-self.profile.recent_window:]]

    @property
    def played(self):
        return np.flatnonzero(self.play_count)

    # ---- listener model ----------------------------------------------------

    def logit(self, tracks, z):
        """Emission logit of the listener for `tracks` if their current style were z.

        Includes this session's fatigue, but not the dislike list (handled in `step`)."""
        p, cat = self.profile, self.catalog
        tracks = np.asarray(tracks)
        recent = np.array(self.recent_tracks, dtype=int)
        artist_recent = (cat.artist[recent][None, :] == cat.artist[tracks].reshape(-1, 1)).sum(1)
        fatigue = self.play_count[tracks].reshape(-1) + artist_recent
        cluster_share = (
            (cat.cluster[recent][None, :] == cat.cluster[tracks].reshape(-1, 1)).mean(1)
            if len(recent) else np.zeros(tracks.size)
        )
        out = (p.beta * cat.similarity(tracks, z)
               + p.lam * (cat.popularity[tracks] - self._pop_mean)
               - p.eta * fatigue.reshape(tracks.shape)
               - p.eta_cluster * cluster_share.reshape(tracks.shape)
               + p.bias)
        return out

    def step(self, track):
        if self.listener is None:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise RuntimeError("session is over; call reset()")
        track = int(track)
        if not 0 <= track < self.catalog.n_tracks:
            raise IndexError(f"track index {track} out of range")

        z = int(self.listener.z_path[self.t])
        disliked = bool(self.listener.dislikes[track])
        if disliked:
            probs = DISLIKE_PROBS
        else:
            probs = feedback_probs(self.logit(track, z), self.profile.cutpoints)
        feedback = Feedback(self._rng.choice(N_FEEDBACK, p=probs))

        rc = self.reward_config
        recent_artists = self.catalog.artist[[tr for tr, _ in self.history[-rc.artist_window:]]]
        artist_repeat = bool((recent_artists == self.catalog.artist[track]).any())
        reward = rc.feedback[feedback] + (rc.artist_repeat if artist_repeat else 0.0)

        info = {
            "z": z,
            "shifted": self.t > 0 and z != self.listener.z_path[self.t - 1],
            "p_positive": float(probs[Feedback.COMPLETE:].sum()),
            "disliked": disliked,
            "artist_repeat": artist_repeat,
        }
        self.history.append((track, feedback))
        self.play_count[track] += 1
        self.t += 1
        return StepResult(feedback, reward, self.done, info)
