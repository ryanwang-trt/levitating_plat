"""Full-episode integration tests.

Drive the env end-to-end the way training does — through gym's API contract —
and assert the episode stays well-formed the whole way: finite obs/reward,
proper termination, and a clean reset->step->done loop.
"""
import numpy as np
import pytest

from env import DroneLevitationEnv

OBS_DIM = 6


@pytest.fixture
def env():
    e = DroneLevitationEnv()
    yield e
    e.close()


def test_random_policy_episode_stays_well_formed(env):
    obs, _ = env.reset(seed=0)
    env.action_space.seed(0)
    steps = 0
    done = False
    while not done and steps < 1000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # Invariants that must hold on every single step of a full episode.
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs)), f"non-finite obs at step {steps}"
        assert np.isfinite(reward), f"non-finite reward at step {steps}"
        done = terminated or truncated
        steps += 1
    assert done, "episode never terminated or truncated within 1000 steps"


def test_episode_terminates_then_resets_cleanly(env):
    # Run an episode to completion, then confirm reset() gives a fresh start.
    env.reset(seed=1)
    for _ in range(500):
        _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
        if terminated or truncated:
            break
    # After a finished episode, a reset should produce a valid fresh obs.
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))
    assert env.step_count == 0


def test_two_consecutive_episodes_independent(env):
    # step_count must reset between episodes (no leakage across resets).
    env.reset()
    for _ in range(10):
        env.step(np.full(4, 0.5, dtype=np.float32))
    count_after_first = env.step_count
    env.reset()
    assert count_after_first == 10
    assert env.step_count == 0


def test_cumulative_reward_is_finite_over_episode(env):
    env.reset(seed=2)
    total = 0.0
    for _ in range(200):
        _, reward, terminated, truncated, _ = env.step(
            np.full(4, 0.5, dtype=np.float32)
        )
        total += reward
        if terminated or truncated:
            break
    assert np.isfinite(total)
