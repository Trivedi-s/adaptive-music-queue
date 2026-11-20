"""Behavior checks for random and sliding-window baselines."""
import unittest
from types import SimpleNamespace
from amq.bandit_baselines import RandomBaseline, SlidingWindowBandit


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.catalog = SimpleNamespace(
            n_tracks=4, cluster=[0, 0, 1, 1], members=([0, 1], [2, 3])
        )
        self.env = SimpleNamespace(catalog=self.catalog, played=[])

    def test_random_excludes_played_and_handles_exhaustion(self):
        agent = RandomBaseline(seed=3)
        self.env.played = [0, 1, 2]
        self.assertEqual(agent(self.env), 3)
        self.env.played.append(3)
        self.assertIsNone(agent(self.env))

    def test_seed_reproducibility(self):
        a, b = RandomBaseline(8), RandomBaseline(8)
        self.assertEqual([a(self.env) for _ in range(20)],
                         [b(self.env) for _ in range(20)])

    def test_exploitation_uses_reward_means(self):
        agent = SlidingWindowBandit(self.catalog, epsilon=0, seed=2)
        agent.update(0, -1)
        agent.update(2, 1)
        self.assertEqual(self.catalog.cluster[agent(self.env)], 1)

    def test_global_window_expires_old_reward(self):
        agent = SlidingWindowBandit(self.catalog, epsilon=0, window_size=2)
        agent.update(0, 100)
        agent.update(2, 1)
        agent.update(0, -1)
        self.assertEqual(self.catalog.cluster[agent(self.env)], 1)

    def test_missing_evidence_is_explored(self):
        agent = SlidingWindowBandit(self.catalog, epsilon=0)
        agent.update(0, 2)
        self.assertEqual(self.catalog.cluster[agent(self.env)], 1)

    def test_exhausted_cluster_is_not_selected(self):
        agent = SlidingWindowBandit(self.catalog)
        self.env.played = [0, 1]
        self.assertIn(agent(self.env), [2, 3])
        self.env.played = list(range(4))
        self.assertIsNone(agent(self.env))

    def test_reset_clears_session_evidence(self):
        agent = SlidingWindowBandit(self.catalog)
        agent.update(0, 2)
        agent.reset(seed=7)
        self.assertEqual(len(agent.history), 0)

    def test_full_exploration_can_choose_both_clusters(self):
        agent = SlidingWindowBandit(self.catalog, epsilon=1, seed=9)
        agent.update(0, -1)
        agent.update(2, 2)
        chosen = {self.catalog.cluster[agent(self.env)] for _ in range(40)}
        self.assertEqual(chosen, {0, 1})

    def test_invalid_parameters(self):
        for window in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                SlidingWindowBandit(self.catalog, window_size=window)
        for epsilon in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                SlidingWindowBandit(self.catalog, epsilon=epsilon)


if __name__ == "__main__":
    unittest.main()
