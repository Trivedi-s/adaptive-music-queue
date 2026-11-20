"""Unit tests for the state encoder, the TD agents, and the evaluation helpers.

Run from the repo root:  python -m unittest discover tests
"""
import unittest
from types import SimpleNamespace

import numpy as np

from amq import PROFILES, Catalog, Feedback, Macro
from amq.agents import TDAgent
from amq.evaluate import (EVAL_SEED_BASE, REFERENCE_AGENTS, epsilon_schedule, evaluate,
                          make_envs, run_episode, train)
from amq.selector import N_MACROS
from amq.state import N_STATES, SHAPE, State, StateEncoder

CAT = Catalog.load()
HOME = 3
ENC = StateEncoder(CAT, session_length=24, home_cluster=HOME)
K = CAT.n_clusters


def one_hot(z, p=1.0):
    b = np.full(K, (1 - p) / (K - 1))
    b[z] = p
    return b


def track_in(z, i=0):
    return int(CAT.members[z][i])


class TestState(unittest.TestCase):
    def test_index_round_trips_and_covers_216_states(self):
        self.assertEqual(N_STATES, 216)
        seen = set()
        for idx in range(N_STATES):
            s = State.from_index(idx)
            self.assertEqual(s.index, idx)
            seen.add(s)
        self.assertEqual(len(seen), N_STATES)

    def test_cold_start_state(self):
        s = ENC.encode([], np.full(K, 1 / K))
        self.assertEqual((s.in_map, s.confidence, s.skip_streak, s.last_feedback, s.phase),
                         (1, 0, 0, 1, 0))

    def test_in_map_and_confidence(self):
        history = [(track_in(5), Feedback.COMPLETE)]
        self.assertEqual(ENC.encode(history, one_hot(5, 0.9)).in_map, 1)
        self.assertEqual(ENC.encode(history, one_hot(2, 0.9)).in_map, 0)
        for p, level in [(0.2, 0), (0.5, 1), (0.9, 2)]:
            self.assertEqual(ENC.encode(history, one_hot(5, p)).confidence, level)

    def test_skip_streak_caps_and_resets(self):
        b = one_hot(0, 0.5)
        skips = [(track_in(0, i), Feedback.EARLY_SKIP) for i in range(5)]
        self.assertEqual(ENC.encode(skips[:2], b).skip_streak, 2)
        self.assertEqual(ENC.encode(skips, b).skip_streak, 3)
        self.assertEqual(ENC.encode(skips + [(track_in(0, 9), Feedback.LIKE)], b).skip_streak, 0)

    def test_last_feedback_levels(self):
        b = one_hot(0, 0.5)
        for fb, level in [(Feedback.EARLY_SKIP, 0), (Feedback.LATE_SKIP, 1), (Feedback.COMPLETE, 1),
                          (Feedback.REPLAY, 2), (Feedback.LIKE, 2)]:
            self.assertEqual(ENC.encode([(track_in(0), fb)], b).last_feedback, level)

    def test_phase_splits_session_in_thirds(self):
        b = one_hot(0, 0.5)
        phases = [ENC.encode([(track_in(0, i), Feedback.COMPLETE) for i in range(t)], b).phase
                  for t in range(24)]
        self.assertEqual(phases, [0] * 8 + [1] * 8 + [2] * 8)


def fake_result(feedback, reward, done=False):
    return SimpleNamespace(feedback=feedback, reward=reward, done=done)


class TestTDAgent(unittest.TestCase):
    def test_q_learning_update_matches_formula(self):
        agent = TDAgent(CAT, alpha=0.5, gamma=0.9, epsilon=0.0, seed=0)
        agent.Q[:] = 0.0
        track = agent.act()
        s, a = agent._state, agent._action
        agent.Q[:, 2] = 1.0           # make max_a' Q[s', a'] = 1 everywhere
        agent.Q[s, a] = 0.0
        agent.observe(track, fake_result(Feedback.LIKE, 2.0))
        self.assertAlmostEqual(agent.Q[s, a], 0.5 * (2.0 + 0.9 * 1.0))

    def test_sarsa_uses_the_action_it_takes_next(self):
        agent = TDAgent(CAT, method="sarsa", alpha=1.0, gamma=1.0, epsilon=0.0, seed=0)
        agent.Q[:] = 0.0
        agent.Q[:, Macro.EXPLORE] = 3.0   # greedy next action = EXPLORE, Q = 3
        agent.Q[:, Macro.STAY] = 5.0      # ...unless STAY is better
        track = agent.act()
        s, a = agent._state, agent._action
        agent.observe(track, fake_result(Feedback.COMPLETE, 1.0))
        self.assertEqual(agent._next_action, Macro.STAY)
        self.assertAlmostEqual(agent.Q[s, a], 1.0 + 5.0)
        agent.act()
        self.assertEqual(agent._action, Macro.STAY)   # it really takes the action it bootstrapped on

    def test_terminal_step_does_not_bootstrap(self):
        agent = TDAgent(CAT, alpha=1.0, epsilon=0.0, seed=0)
        agent.Q[:] = 10.0
        track = agent.act()
        s, a = agent._state, agent._action
        agent.observe(track, fake_result(Feedback.EARLY_SKIP, -1.0, done=True))
        self.assertAlmostEqual(agent.Q[s, a], -1.0)

    def test_no_learning_when_learn_is_false(self):
        agent = TDAgent(CAT, seed=0)
        track = agent.act()
        agent.observe(track, fake_result(Feedback.LIKE, 2.0), learn=False)
        self.assertEqual(agent.Q.sum(), 0.0)
        self.assertEqual(agent.visits.sum(), 0)

    def test_no_belief_agent_sees_constant_belief_features(self):
        env = make_envs(CAT, ["abrupt"])["abrupt"]
        agent = TDAgent(CAT, use_belief=False, epsilon=0.3, seed=0)
        env.reset(seed=0)
        agent.reset(seed=0)
        states = set()
        while not env.done:
            s = agent.state()
            states.add((s.in_map, s.confidence))
            track = agent.act(env)
            agent.observe(track, env.step(track))
        self.assertEqual(states, {(1, 0)})

    def test_greedy_breaks_ties_randomly(self):
        agent = TDAgent(CAT, seed=0)
        picks = {agent.greedy(0) for _ in range(200)}
        self.assertEqual(picks, set(range(N_MACROS)))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            TDAgent(CAT, method="dqn")
        with self.assertRaises(ValueError):
            TDAgent(CAT, gamma=1.5)


class TestEvaluation(unittest.TestCase):
    def test_epsilon_schedule(self):
        self.assertAlmostEqual(epsilon_schedule(0, 100), 1.0)
        self.assertAlmostEqual(epsilon_schedule(30, 100), 0.525)
        self.assertAlmostEqual(epsilon_schedule(60, 100), 0.05)
        self.assertAlmostEqual(epsilon_schedule(99, 100), 0.05)

    def test_every_reference_agent_plays_valid_sessions(self):
        env = make_envs(CAT, ["drifter"])["drifter"]
        for name, cls in REFERENCE_AGENTS.items():
            total, skips = run_episode(env, cls(CAT), seed=EVAL_SEED_BASE)
            self.assertTrue(np.isfinite(total), name)
            self.assertTrue(0 <= skips <= env.session_length, name)
            self.assertEqual(len(set(env.played.tolist())), env.session_length, name)

    def test_evaluation_restores_epsilon_and_is_deterministic(self):
        envs = make_envs(CAT, ["stable", "picky"])
        agent = TDAgent(CAT, epsilon=0.7, seed=0)
        a = evaluate(agent, envs, n_seeds=5)
        self.assertEqual(agent.epsilon, 0.7)
        self.assertEqual(a, evaluate(agent, envs, n_seeds=5))

    def test_training_improves_over_untrained_agent(self):
        envs = make_envs(CAT, PROFILES)
        agent = TDAgent(CAT, alpha=0.05, seed=0)
        curve = train(agent, envs, n_episodes=1500, eval_every=1500, eval_seeds=30)
        before, after = (np.mean(list(scores.values())) for _, scores in (curve[0], curve[-1]))
        self.assertGreater(after, before + 3)
        self.assertEqual([ep for ep, _ in curve], [0, 1500])


if __name__ == "__main__":
    unittest.main()
