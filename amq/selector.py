"""Macro-actions and the low-level track selector (README "MDP Components", Actions).

The RL agent chooses one of four macro-actions. Each one names a *target cluster*; the
low-level selector then picks a concrete track from that cluster. The selector is
deterministic, so all learning happens at the macro level.

  STAY           the current cluster (cluster of the last played track)
  BRIDGE         one of the current cluster's `n_bridge` nearest clusters
  FOLLOW_BELIEF  the cluster with the highest belief
  EXPLORE        the least-visited cluster this session, other than the current one

Ties are broken so that the choice is always well defined and never loops:

  BRIDGE         highest belief, then fewest visits, then nearest
  FOLLOW_BELIEF  highest belief, then nearest to the current cluster (so a flat
                 belief means "stay")
  EXPLORE        fewest visits, then highest belief, then nearest

Without a belief (None), a uniform belief is assumed, so baselines can use the same
actions. Under a uniform belief, repeated BRIDGE moves walk outward to unvisited
neighbors instead of bouncing between the same two clusters, and repeated EXPLORE
moves eventually visit every cluster.

Track choice inside the target cluster:
  1. only tracks not yet played this session;
  2. skip artists heard in the last `artist_window` tracks, unless nothing else is left;
  3. rank by distance to an anchor: the mean of the tracks the listener responded to
     positively *in this cluster* this session, or the cluster centroid if there are
     none. A typical track also gives the belief filter the clearest evidence;
  4. if the cluster has no unplayed tracks, use the nearest cluster that has some.

The selector reads only what an agent may see: the session history [(track, Feedback)]
and the agent's own belief. It never touches the environment's ground truth.

Before the first track there is no current cluster. The selector then acts as if the
current cluster were the one closest to the catalog's average sound (`home_cluster`),
a neutral, deterministic opener.
"""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Macro(IntEnum):
    STAY = 0
    BRIDGE = 1
    FOLLOW_BELIEF = 2
    EXPLORE = 3


N_MACROS = len(Macro)


@dataclass(frozen=True)
class Choice:
    track: int        # catalog row index to pass to env.step()
    cluster: int      # cluster the track was taken from
    target: int       # cluster the macro-action asked for (differs only if it was exhausted)
    macro: Macro


class Selector:
    def __init__(self, catalog, artist_window=5, n_bridge=3):
        self.catalog = catalog
        self.artist_window = artist_window
        self.n_bridge = n_bridge
        center = catalog.features.mean(axis=0)
        self.home_cluster = int(np.argmin(np.linalg.norm(catalog.centroids - center, axis=1)))
        # neighbors[z] = other clusters ordered from nearest to farthest
        self.neighbors = [
            [int(c) for c in np.argsort(catalog.centroid_dist[z], kind="stable") if c != z]
            for z in range(catalog.n_clusters)
        ]

    # ---- what the session history tells us -----------------------------------

    def current_cluster(self, history):
        """Cluster of the last played track, or None before the first track."""
        return int(self.catalog.cluster[history[-1][0]]) if history else None

    def visits(self, history):
        """(K,) number of tracks played from each cluster this session."""
        tracks = np.array([t for t, _ in history], dtype=int)
        return np.bincount(self.catalog.cluster[tracks], minlength=self.catalog.n_clusters)

    # ---- macro-action -> target cluster ---------------------------------------

    def target_cluster(self, macro, history, belief=None):
        macro = Macro(macro)
        K = self.catalog.n_clusters
        belief = np.full(K, 1.0 / K) if belief is None else np.asarray(belief, dtype=float)
        if belief.shape != (K,):
            raise ValueError(f"belief must have shape ({K},)")
        current = self.current_cluster(history)
        here = self.home_cluster if current is None else current
        visits = self.visits(history)
        dist = self.catalog.centroid_dist[here]
        b = np.round(belief, 9)   # so float noise doesn't break exact ties

        if macro == Macro.STAY:
            return here
        if macro == Macro.BRIDGE:
            options = self.neighbors[here][:self.n_bridge]
            return min(options, key=lambda z: (-b[z], visits[z], dist[z]))
        if macro == Macro.FOLLOW_BELIEF:
            return min(range(K), key=lambda z: (-b[z], dist[z]))
        # EXPLORE
        return min(self.neighbors[here], key=lambda z: (visits[z], -b[z], dist[z]))

    # ---- target cluster -> track ----------------------------------------------

    def rank(self, cluster, history):
        """Unplayed tracks of `cluster`, best first. Recent artists go last."""
        cat = self.catalog
        played = {t for t, _ in history}
        tracks = np.array([t for t in cat.members[cluster] if t not in played], dtype=int)
        if tracks.size == 0:
            return tracks

        liked = [t for t, fb in history if fb.is_positive and cat.cluster[t] == cluster]
        anchor = cat.features[liked].mean(axis=0) if liked else cat.centroids[cluster]
        dist = np.linalg.norm(cat.features[tracks] - anchor, axis=1)

        recent_artists = {cat.artist[t] for t, _ in history[-self.artist_window:]}
        repeat = np.array([cat.artist[t] in recent_artists for t in tracks])
        order = np.lexsort((tracks, dist, repeat))   # non-repeats first, then nearest, then id
        return tracks[order]

    def select(self, macro, history, belief=None):
        """Pick the next track for a macro-action. Returns a Choice."""
        macro = Macro(macro)
        target = self.target_cluster(macro, history, belief)
        for cluster in [target] + self.neighbors[target]:
            ranked = self.rank(cluster, history)
            if ranked.size:
                return Choice(int(ranked[0]), cluster, target, macro)
        raise RuntimeError("every track in the catalog has been played")

    def candidates(self, macro, history, belief=None, n=20):
        """The top-n tracks the selector would consider (for the demo and debugging)."""
        target = self.target_cluster(macro, history, belief)
        return self.rank(target, history)[:n]
