"""Unit tests for the catalog and the HMM listener simulator.

Run from the repo root:  python -m unittest discover tests
"""
import dataclasses
import unittest

import numpy as np

from amq import PROFILES, Catalog, Feedback, ListenerEnv, RewardConfig, feedback_probs
from amq.simulator import sigmoid

CAT = Catalog.load()


def fresh_track(env, z, rng):
    """An unplayed track in cluster z whose artist was not played recently."""
    played = set(env.played.tolist())
    recent_artists = set(CAT.artist[env.recent_tracks].tolist())
    pool = [i for i in CAT.members[z] if i not in played and CAT.artist[i] not in recent_artists]
    return int(rng.choice(pool))


class TestCatalog(unittest.TestCase):
    def test_shapes_and_partition(self):
        self.assertEqual(CAT.n_tracks, 2000)
        self.assertEqual(CAT.n_clusters, 8)
        self.assertEqual(CAT.features.shape, (2000, 7))
        self.assertEqual(sum(len(m) for m in CAT.members), CAT.n_tracks)
        self.assertTrue(((CAT.features >= 0) & (CAT.features <= 1)).all())
        self.assertTrue(((CAT.popularity >= 0) & (CAT.popularity <= 1)).all())

    def test_nearest_cluster_matches_centroid_distances(self):
        d = CAT.centroid_dist + np.diag(np.full(CAT.n_clusters, np.inf))
        np.testing.assert_array_equal(CAT.nearest_cluster, d.argmin(axis=1))

    def test_tracks_are_closest_to_own_centroid_on_average(self):
        for z, members in enumerate(CAT.members):
            own = CAT.similarity(members, z).mean()
            for other in range(CAT.n_clusters):
                if other != z:
                    self.assertGreater(own, CAT.similarity(members, other).mean())


class TestFeedbackModel(unittest.TestCase):
    def test_probs_are_a_distribution_and_match_sigmoid(self):
        logits = np.linspace(-6, 6, 25)
        probs = feedback_probs(logits)
        self.assertEqual(probs.shape, (25, 5))
        self.assertTrue((probs >= 0).all())
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        np.testing.assert_allclose(probs[:, Feedback.COMPLETE:].sum(axis=1), sigmoid(logits))

    def test_higher_logit_shifts_mass_towards_positive(self):
        lo, hi = feedback_probs(-1.0), feedback_probs(2.0)
        self.assertGreater(lo[Feedback.EARLY_SKIP], hi[Feedback.EARLY_SKIP])
        self.assertGreater(hi[Feedback.LIKE], lo[Feedback.LIKE])


class TestHiddenPath(unittest.TestCase):
    def test_stable_and_picky_never_switch(self):
        for name in ["stable", "picky"]:
            env = ListenerEnv(CAT, name)
            for seed in range(50):
                env.reset(seed=seed)
                self.assertEqual(len(set(env.listener.z_path)), 1)

    def test_abrupt_switches_exactly_at_shift_step(self):
        env = ListenerEnv(CAT, "abrupt", session_length=25)
        for seed in range(50):
            z = env.reset(seed=seed).z_path
            changes = np.flatnonzero(z[1:] != z[:-1]) + 1
            np.testing.assert_array_equal(changes, [PROFILES["abrupt"].shift_step])

    def test_drifter_moves_only_to_neighbouring_styles(self):
        env = ListenerEnv(CAT, "drifter")
        k = PROFILES["drifter"].n_neighbors
        n_switches = 0
        for seed in range(200):
            z = env.reset(seed=seed).z_path
            self.assertTrue(0.05 <= env.listener.rho <= 0.10)
            for a, b in zip(z[:-1], z[1:]):
                if a != b:
                    n_switches += 1
                    self.assertIn(b, np.argsort(CAT.centroid_dist[a])[1:1 + k])
        self.assertGreater(n_switches, 0)

    def test_switch_rate_close_to_rho(self):
        env = ListenerEnv(CAT, "explorer", session_length=30)
        paths = np.array([env.reset(seed=s).z_path for s in range(400)])
        rate = (paths[:, 1:] != paths[:, :-1]).mean()
        self.assertAlmostEqual(rate, PROFILES["explorer"].rho, delta=0.02)


class TestEnvironment(unittest.TestCase):
    def test_same_seed_same_listener_and_feedback(self):
        env_a, env_b = ListenerEnv(CAT, "drifter"), ListenerEnv(CAT, "drifter")
        a, b = env_a.reset(seed=7), env_b.reset(seed=7)
        np.testing.assert_array_equal(a.w, b.w)
        np.testing.assert_array_equal(a.z_path, b.z_path)
        np.testing.assert_array_equal(a.dislikes, b.dislikes)
        tracks = np.random.default_rng(0).choice(CAT.n_tracks, size=25, replace=False)
        fa = [env_a.step(t).feedback for t in tracks]
        fb = [env_b.step(t).feedback for t in tracks]
        self.assertEqual(fa, fb)

    def test_hidden_path_does_not_depend_on_actions(self):
        env = ListenerEnv(CAT, "explorer")
        path = env.reset(seed=3).z_path.copy()
        for t in range(25):
            env.step(t)
        env.reset(seed=3)
        for t in range(25):
            env.step(1999 - t)
        np.testing.assert_array_equal(env.listener.z_path, path)

    def test_session_length_and_done(self):
        env = ListenerEnv(CAT, "stable", session_length=20)
        env.reset(seed=0)
        results = [env.step(i) for i in range(20)]
        self.assertTrue(env.done and results[-1].done)
        self.assertFalse(any(r.done for r in results[:-1]))
        with self.assertRaises(RuntimeError):
            env.step(100)

    def test_invalid_use_raises(self):
        env = ListenerEnv(CAT, "stable")
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(seed=0)
        with self.assertRaises(IndexError):
            env.step(CAT.n_tracks)
        with self.assertRaises(ValueError):
            ListenerEnv(CAT, "abrupt", session_length=8)

    def test_stable_listener_prefers_its_own_style(self):
        env = ListenerEnv(CAT, "stable")
        own, other = [], []
        for seed in range(200):
            z = env.reset(seed=seed).z_path[0]
            rng = np.random.default_rng(seed)
            for t in range(env.session_length):
                target = z if t % 2 == 0 else (z + 1 + rng.integers(CAT.n_clusters - 1)) % CAT.n_clusters
                r = env.step(fresh_track(env, target, rng))
                (own if target == z else other).append(r.feedback.is_positive)
        self.assertGreater(np.mean(own), 0.55)
        self.assertLess(np.mean(other), 0.35)

    def test_disliked_tracks_are_skipped_whatever_the_mood(self):
        env = ListenerEnv(CAT, "picky")
        n = 0
        for seed in range(30):
            listener = env.reset(seed=seed)
            z = listener.z_path[0]
            # disliked tracks that fit the listener's current style perfectly well
            for track in np.flatnonzero(listener.dislikes & (CAT.cluster == z))[:env.session_length]:
                r = env.step(track)
                self.assertTrue(r.info["disliked"])
                self.assertTrue(r.feedback.is_skip)
                n += 1
        self.assertGreater(n, 100)

    def test_dislike_rate(self):
        for name in ["stable", "picky"]:
            env = ListenerEnv(CAT, name)
            rate = np.mean([env.reset(seed=s).dislikes.mean() for s in range(20)])
            self.assertAlmostEqual(rate, PROFILES[name].dislike_rate, delta=0.02)

    def test_fatigue_lowers_logit_for_repeats(self):
        env = ListenerEnv(CAT, "stable")
        z = env.reset(seed=0).z_path[0]
        track = int(CAT.members[z][0])
        before = env.logit(track, z)
        env.step(track)
        self.assertAlmostEqual(before - env.logit(track, z), 2 * PROFILES["stable"].eta)  # track + artist
        same_artist = [i for i in range(CAT.n_tracks) if CAT.artist[i] == CAT.artist[track] and i != track]
        if same_artist:
            other = same_artist[0]
            env2 = ListenerEnv(CAT, "stable")
            env2.reset(seed=0)
            fresh = env2.logit(other, z)
            self.assertAlmostEqual(fresh - env.logit(other, z), PROFILES["stable"].eta)

    def test_explorer_penalises_staying_in_one_cluster(self):
        env = ListenerEnv(CAT, "explorer")
        z = env.reset(seed=0).z_path[0]
        rng = np.random.default_rng(0)
        probe = fresh_track(env, z, rng)
        before = env.logit(probe, z)
        for _ in range(5):
            env.step(fresh_track(env, z, rng))
        self.assertAlmostEqual(before - env.logit(probe, z), PROFILES["explorer"].eta_cluster)

    def test_reward_values_and_artist_repeat_penalty(self):
        rc = RewardConfig()
        self.assertEqual(rc.feedback[Feedback.LIKE], 2.0)
        self.assertEqual(rc.feedback[Feedback.EARLY_SKIP], -1.0)
        env = ListenerEnv(CAT, "stable")
        env.reset(seed=0)
        artist = CAT.artist
        a, b = next((i, j) for i in range(CAT.n_tracks) for j in range(i + 1, CAT.n_tracks)
                    if artist[i] == artist[j])
        r1 = env.step(a)
        self.assertFalse(r1.info["artist_repeat"])
        self.assertEqual(r1.reward, rc.feedback[r1.feedback])
        r2 = env.step(b)
        self.assertTrue(r2.info["artist_repeat"])
        self.assertEqual(r2.reward, rc.feedback[r2.feedback] + rc.artist_repeat)

    def test_oracle_beats_random_on_every_profile(self):
        for name in PROFILES:
            env = ListenerEnv(CAT, name)
            oracle, random = [], []
            for seed in range(60):
                for policy, out in [("oracle", oracle), ("random", random)]:
                    env.reset(seed=seed)
                    rng = np.random.default_rng(seed)
                    total = 0.0
                    while not env.done:
                        if policy == "oracle":
                            track = fresh_track(env, env.listener.z_path[env.t], rng)
                        else:
                            track = int(rng.choice(np.setdiff1d(np.arange(CAT.n_tracks), env.played)))
                        total += env.step(track).reward
                    out.append(total)
            self.assertGreater(np.mean(oracle), np.mean(random) + 5, name)


if __name__ == "__main__":
    unittest.main()
