"""Unit tests for the HMM belief filter.

Run from the repo root:  python -m unittest discover tests
"""
import itertools
import unittest

import numpy as np

from amq import Catalog, Feedback, ListenerEnv, Macro, Selector
from amq.belief import BeliefFilter, ListenerModel

CAT = Catalog.load()
SEL = Selector(CAT)
K = CAT.n_clusters


def typical(z, i=0):
    """The i-th most typical track of cluster z."""
    return int(SEL.rank(z, [])[i])


def streak(history):
    n = 0
    for _, fb in reversed(history):
        if not fb.is_skip:
            break
        n += 1
    return n


class TestFilterMechanics(unittest.TestCase):
    def test_starts_uniform_and_stays_a_distribution(self):
        bf = BeliefFilter(CAT)
        np.testing.assert_allclose(bf.belief, np.full(K, 1 / K))
        rng = np.random.default_rng(0)
        for _ in range(30):
            bf.update(int(rng.integers(CAT.n_tracks)), Feedback(int(rng.integers(5))))
            self.assertAlmostEqual(bf.belief.sum(), 1.0)
            self.assertAlmostEqual(bf.posterior.sum(), 1.0)
            self.assertTrue((bf.belief > 0).all())

    def test_emission_is_a_distribution_per_cluster(self):
        E = BeliefFilter(CAT).emission(typical(3))
        self.assertEqual(E.shape, (K, len(Feedback)))
        np.testing.assert_allclose(E.sum(axis=1), 1.0)

    def test_matches_brute_force_enumeration_of_hidden_paths(self):
        model = ListenerModel(rho=0.15)
        bf = BeliefFilter(CAT, model)
        steps = [(typical(1), Feedback.COMPLETE), (typical(5), Feedback.EARLY_SKIP),
                 (typical(6, 3), Feedback.LIKE)]
        T = bf.transition
        E = [bf.emission(track)[:, int(fb)] for track, fb in steps]
        # P(z_3 = z' | observations) by summing over every path z_0, z_1, z_2
        joint = np.zeros(K)
        for path in itertools.product(range(K), repeat=len(steps)):
            p = 1.0 / K
            for t, z in enumerate(path):
                if t:
                    p *= T[path[t - 1], z]
                p *= E[t][z]
            joint += p * T[path[-1]]
        np.testing.assert_allclose(bf.run(steps), joint / joint.sum())

    def test_predict_step_leaks_rho_to_other_clusters(self):
        rho = 0.1
        bf = BeliefFilter(CAT, ListenerModel(rho=rho))
        bf.update(typical(2), Feedback.LIKE)
        post = bf.posterior
        expected = post * (1 - rho) + (1 - post) * rho / (K - 1)
        np.testing.assert_allclose(bf.belief, expected)

    def test_rho_zero_never_leaks(self):
        bf = BeliefFilter(CAT, ListenerModel(rho=0.0))
        bf.update(typical(4), Feedback.LIKE)
        np.testing.assert_allclose(bf.belief, bf.posterior)

    def test_run_replays_history(self):
        history = [(typical(z), Feedback(z % 5)) for z in range(K)]
        a = BeliefFilter(CAT)
        for track, fb in history:
            a.update(track, fb)
        np.testing.assert_allclose(BeliefFilter(CAT).run(history), a.belief)

    def test_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            BeliefFilter(CAT, ListenerModel(rho=1.5))
        with self.assertRaises(ValueError):
            BeliefFilter(CAT, ListenerModel(dislike_rate=1.0))


class TestInference(unittest.TestCase):
    def test_likes_concentrate_belief_on_their_cluster(self):
        for z in range(K):
            bf = BeliefFilter(CAT)
            for i in range(4):
                bf.update(typical(z, i), Feedback.LIKE)
            self.assertEqual(bf.map_cluster, z)
            self.assertGreater(bf.confidence, 0.5)

    def test_skip_lowers_and_like_raises_belief_in_that_cluster(self):
        for z in range(K):
            for fb, sign in [(Feedback.EARLY_SKIP, -1), (Feedback.LIKE, +1)]:
                bf = BeliefFilter(CAT)
                before = bf.belief[z]
                bf.update(typical(z), fb)
                self.assertEqual(np.sign(bf.belief[z] - before), sign)

    def test_one_skip_is_a_rejection_but_a_run_of_skips_is_a_shift(self):
        """The core ambiguity: a confident belief survives one skip but not a run."""
        survived, moved = 0, 0
        for z in range(K):
            bf = BeliefFilter(CAT)
            for i in range(4):
                bf.update(typical(z, i), Feedback.LIKE)
            bf.update(typical(z, 4), Feedback.EARLY_SKIP)
            survived += bf.map_cluster == z
            for i in range(5, 8):
                bf.update(typical(z, i), Feedback.EARLY_SKIP)
            moved += bf.map_cluster != z
        self.assertEqual(survived, K)
        self.assertEqual(moved, K)

    def test_dislike_rate_makes_single_skips_weaker_evidence(self):
        z = 6
        drops = []
        for d in (0.0, 0.1, 0.3):
            bf = BeliefFilter(CAT, ListenerModel(dislike_rate=d))
            for i in range(3):
                bf.update(typical(z, i), Feedback.LIKE)
            before = bf.belief[z]
            bf.update(typical(z, 3), Feedback.EARLY_SKIP)
            drops.append(before - bf.belief[z])
        self.assertGreater(drops[0], drops[1])
        self.assertGreater(drops[1], drops[2])


class TestWithSimulator(unittest.TestCase):
    def rollout(self, profile, policy, seeds=40, rho=0.05):
        env = ListenerEnv(CAT, profile)
        bf = BeliefFilter(CAT, ListenerModel(rho=rho))
        returns, on_truth = [], []
        for seed in range(seeds):
            env.reset(seed=seed)
            bf.reset()
            total = 0.0
            while not env.done:
                on_truth.append((env.t, bf.belief[env.listener.z_path[env.t]]))
                choice = policy(env, bf)
                result = env.step(choice.track)
                bf.update(choice.track, result.feedback)
                total += result.reward
            returns.append(total)
        return np.mean(returns), on_truth

    def test_belief_tracks_the_hidden_style(self):
        _, on_truth = self.rollout("stable", lambda env, bf: SEL.select(
            Macro.FOLLOW_BELIEF, env.history, bf.belief))
        late = [b for t, b in on_truth if t >= 10]
        self.assertGreater(np.mean(late), 3 / K)   # well above the 1/K of a clueless filter

    def test_belief_recovers_after_an_abrupt_shift(self):
        env = ListenerEnv(CAT, "abrupt")
        shift = env.profile.shift_step
        at_shift, later = [], []
        bf = BeliefFilter(CAT)
        for seed in range(40):
            env.reset(seed=seed)
            bf.reset()
            new_z = env.listener.z_path[shift]
            while not env.done:
                if env.t == shift:
                    at_shift.append(bf.belief[new_z])
                if env.t == shift + 6:
                    later.append(bf.belief[new_z])
                choice = SEL.select(Macro.FOLLOW_BELIEF, env.history, bf.belief)
                bf.update(choice.track, env.step(choice.track).feedback)
        self.assertGreater(np.mean(later), np.mean(at_shift) + 0.2)

    def test_following_the_belief_beats_skip_counting(self):
        follow = lambda env, bf: SEL.select(Macro.FOLLOW_BELIEF, env.history, bf.belief)
        count = lambda env, bf: SEL.select(
            Macro.EXPLORE if streak(env.history) >= 2 else Macro.STAY, env.history)
        for profile in ("stable", "picky", "abrupt"):
            self.assertGreater(self.rollout(profile, follow)[0], self.rollout(profile, count)[0] + 2, profile)


if __name__ == "__main__":
    unittest.main()
