"""Unit tests for the macro-actions and the low-level selector.

Run from the repo root:  python -m unittest discover tests
"""
import unittest

import numpy as np

from amq import Catalog, Feedback, ListenerEnv, Macro, PROFILES, Selector

CAT = Catalog.load()
SEL = Selector(CAT)
K = CAT.n_clusters


def one_hot(z):
    b = np.zeros(K)
    b[z] = 1.0
    return b


def history_in(cluster, feedback, n):
    """n distinct tracks from `cluster` that all got `feedback`."""
    return [(int(t), feedback) for t in CAT.members[cluster][:n]]


class TestTargetCluster(unittest.TestCase):
    def test_cold_start_uses_home_cluster(self):
        center = CAT.features.mean(axis=0)
        expected = np.argmin(np.linalg.norm(CAT.centroids - center, axis=1))
        self.assertEqual(SEL.home_cluster, expected)
        for macro in (Macro.STAY, Macro.FOLLOW_BELIEF):
            self.assertEqual(SEL.target_cluster(macro, []), SEL.home_cluster)

    def test_stay_targets_cluster_of_last_track(self):
        for z in range(K):
            history = history_in((z + 1) % K, Feedback.COMPLETE, 2) + history_in(z, Feedback.EARLY_SKIP, 1)
            self.assertEqual(SEL.target_cluster(Macro.STAY, history), z)

    def test_follow_belief_targets_argmax(self):
        history = history_in(0, Feedback.COMPLETE, 1)
        for z in range(K):
            self.assertEqual(SEL.target_cluster(Macro.FOLLOW_BELIEF, history, one_hot(z)), z)

    def test_follow_flat_belief_means_stay(self):
        for z in range(K):
            history = history_in(z, Feedback.LATE_SKIP, 1)
            self.assertEqual(SEL.target_cluster(Macro.FOLLOW_BELIEF, history), z)
            self.assertEqual(SEL.target_cluster(Macro.FOLLOW_BELIEF, history, np.full(K, 1 / K)), z)

    def test_bridge_goes_to_a_near_neighbor_preferring_belief(self):
        for z in range(K):
            history = history_in(z, Feedback.COMPLETE, 1)
            near = SEL.neighbors[z][:SEL.n_bridge]
            self.assertEqual(SEL.target_cluster(Macro.BRIDGE, history), near[0])
            # belief on the farthest of the near neighbors pulls BRIDGE there
            self.assertEqual(SEL.target_cluster(Macro.BRIDGE, history, one_hot(near[-1])), near[-1])
            # belief on a far cluster cannot drag BRIDGE outside the neighborhood
            far = SEL.neighbors[z][-1]
            self.assertIn(SEL.target_cluster(Macro.BRIDGE, history, one_hot(far)), near)

    def test_repeated_bridge_does_not_ping_pong(self):
        history = []
        seen = []
        for _ in range(6):
            choice = SEL.select(Macro.BRIDGE, history)
            history.append((choice.track, Feedback.EARLY_SKIP))
            seen.append(choice.cluster)
        self.assertGreater(len(set(seen)), 2, seen)

    def test_explore_never_stays_and_sweeps_every_cluster(self):
        history = []
        for step in range(K):
            current = SEL.current_cluster(history)
            choice = SEL.select(Macro.EXPLORE, history)
            if current is not None:
                self.assertNotEqual(choice.cluster, current)
            history.append((choice.track, Feedback.EARLY_SKIP))
        self.assertEqual(set(SEL.visits(history).nonzero()[0]), set(range(K)))

    def test_explore_prefers_unvisited_then_plausible(self):
        history = history_in(0, Feedback.COMPLETE, 3) + history_in(1, Feedback.COMPLETE, 1)
        unvisited = [z for z in range(K) if z not in (0, 1)]
        self.assertIn(SEL.target_cluster(Macro.EXPLORE, history), unvisited)
        self.assertEqual(SEL.target_cluster(Macro.EXPLORE, history, one_hot(unvisited[-1])), unvisited[-1])

    def test_bad_belief_shape_raises(self):
        with self.assertRaises(ValueError):
            SEL.target_cluster(Macro.FOLLOW_BELIEF, [], np.ones(K + 1))


class TestTrackChoice(unittest.TestCase):
    def test_never_repeats_a_track_and_stays_in_target(self):
        rng = np.random.default_rng(0)
        history = []
        for _ in range(30):
            macro = Macro(int(rng.integers(len(Macro))))
            belief = rng.dirichlet(np.ones(K))
            choice = SEL.select(macro, history, belief)
            self.assertNotIn(choice.track, [t for t, _ in history])
            self.assertEqual(CAT.cluster[choice.track], choice.cluster)
            self.assertEqual(choice.cluster, choice.target)
            history.append((choice.track, Feedback(int(rng.integers(len(Feedback))))))

    def test_avoids_recent_artists_when_possible(self):
        history = []
        for _ in range(25):
            choice = SEL.select(Macro.STAY, history)
            recent = {CAT.artist[t] for t, _ in history[-SEL.artist_window:]}
            self.assertNotIn(CAT.artist[choice.track], recent)
            history.append((choice.track, Feedback.COMPLETE))

    def test_falls_back_to_recent_artist_only_when_nothing_else(self):
        # A cluster whose unplayed tracks are all by recently played artists still yields a track.
        z = 0
        artist = CAT.artist[CAT.members[z][0]]
        same = [int(t) for t in CAT.members[z] if CAT.artist[t] == artist]
        others = [(int(t), Feedback.COMPLETE) for t in CAT.members[z] if CAT.artist[t] != artist]
        history = others + [(same[0], Feedback.COMPLETE)]
        ranked = SEL.rank(z, history)
        self.assertEqual(set(ranked.tolist()), set(same[1:]))

    def test_first_pick_is_the_most_typical_track(self):
        for z in range(K):
            members = CAT.members[z]
            typical = members[np.argmin(np.linalg.norm(CAT.features[members] - CAT.centroids[z], axis=1))]
            self.assertEqual(SEL.rank(z, [])[0], typical)

    def test_anchor_moves_to_tracks_the_listener_liked(self):
        z = 5
        members = CAT.members[z]
        # the least typical track in the cluster, liked by the listener
        liked = int(members[np.argmax(np.linalg.norm(CAT.features[members] - CAT.centroids[z], axis=1))])
        history = [(liked, Feedback.LIKE)]
        ranked = SEL.rank(z, history)
        candidates = ranked[[CAT.artist[t] != CAT.artist[liked] for t in ranked]]
        nearest = candidates[np.argmin(np.linalg.norm(CAT.features[candidates] - CAT.features[liked], axis=1))]
        self.assertEqual(ranked[0], nearest)
        # a skipped track does not move the anchor: the ranking stays centroid-based
        # (apart from dropping that track and its artist to the end)
        centroid_order = [t for t in SEL.rank(z, []) if CAT.artist[t] != CAT.artist[liked]]
        self.assertEqual(SEL.rank(z, [(liked, Feedback.EARLY_SKIP)])[0], centroid_order[0])

    def test_exhausted_cluster_falls_back_to_nearest(self):
        z = 4
        history = [(int(t), Feedback.EARLY_SKIP) for t in CAT.members[z]]
        choice = SEL.select(Macro.STAY, history)
        self.assertEqual(choice.target, z)
        self.assertEqual(choice.cluster, SEL.neighbors[z][0])

    def test_candidates_are_the_top_of_the_ranking(self):
        history = history_in(2, Feedback.COMPLETE, 3)
        cands = SEL.candidates(Macro.STAY, history, n=20)
        self.assertEqual(len(cands), 20)
        self.assertEqual(cands[0], SEL.select(Macro.STAY, history).track)


class TestWithSimulator(unittest.TestCase):
    def run_policy(self, profile, policy, seeds=40):
        env = ListenerEnv(CAT, profile)
        returns = []
        for seed in range(seeds):
            env.reset(seed=seed)
            total = 0.0
            while not env.done:
                result = env.step(policy(env).track)
                self.assertFalse(result.info["artist_repeat"])
                total += result.reward
            returns.append(total)
        return np.mean(returns)

    def test_follow_true_state_beats_random_by_a_wide_margin(self):
        for name in PROFILES:
            oracle = self.run_policy(name, lambda env: SEL.select(
                Macro.FOLLOW_BELIEF, env.history, one_hot(env.listener.z_path[env.t])))
            stay = self.run_policy(name, lambda env: SEL.select(Macro.STAY, env.history))
            self.assertGreater(oracle, stay + 10, name)
            self.assertGreater(oracle, 5, name)


if __name__ == "__main__":
    unittest.main()
