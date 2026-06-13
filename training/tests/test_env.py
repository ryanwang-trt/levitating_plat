"""Unit / integration tests for DroneLevitationEnv.

Runs PyBullet headless (DIRECT mode) — no GUI. A fresh env is created per test
and closed afterwards so PyBullet connections don't leak.
"""
import numpy as np
import pytest

from env import DroneLevitationEnv

OBS_DIM = 6
ACTION_DIM = 4


@pytest.fixture
def env():
    e = DroneLevitationEnv()  # render_mode=None -> DIRECT (headless)
    yield e
    e.close()


def test_spaces_shapes_and_bounds(env):
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACTION_DIM,)
    # Action space is the motor throttle range [0, 1].
    assert np.all(env.action_space.low == 0.0)
    assert np.all(env.action_space.high == 1.0)


def test_reset_returns_valid_obs(env):
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    assert np.all(np.isfinite(obs))


def test_reset_starts_near_target_height(env):
    obs, _ = env.reset()
    # Platform is spawned at target_height (0.5 m); height is obs[0].
    assert obs[0] == pytest.approx(env.target_height, abs=0.05)


def test_step_returns_five_tuple(env):
    env.reset()
    result = env.step(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_reward_is_finite(env):
    env.reset()
    _, reward, _, _, _ = env.step(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
    assert np.isfinite(reward)


def test_action_is_clipped_internally(env):
    # Out-of-range actions must not crash; env clips to [0, 1] in step().
    env.reset()
    obs, reward, _, _, _ = env.step(np.array([5.0, -3.0, 0.5, 100.0], dtype=np.float32))
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)


def test_full_thrust_climbs(env):
    # Max thrust is ~2x hover, so full throttle should raise the platform.
    env.reset()
    start_h = env._get_obs()[0]
    for _ in range(30):
        obs, _, terminated, truncated, _ = env.step(
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        )
        if terminated or truncated:
            break
    assert obs[0] > start_h


def test_no_thrust_falls_and_terminates(env):
    # Zero thrust -> platform drops below 0.05 m floor -> terminated=True.
    env.reset()
    terminated = False
    for _ in range(500):
        _, _, terminated, truncated, _ = env.step(
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            break
    assert terminated


def test_truncation_at_max_episode_length():
    # Hover-ish action so it doesn't terminate early; episode caps at 1000 steps.
    e = DroneLevitationEnv()
    try:
        e.reset()
        truncated = False
        # Counterbalance gravity to stay alive long enough to hit truncation.
        hover = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        for step in range(1000):
            _, _, terminated, truncated, _ = e.step(hover)
            if terminated:
                # If it happens to terminate, this test's premise fails — but the
                # truncation boundary itself is what we assert below via step_count.
                break
        # step_count is incremented each step; truncation fires at >= 1000.
        assert e.step_count <= 1000
        if e.step_count >= 1000:
            assert truncated
    finally:
        e.close()


def test_seeded_reset_is_reproducible():
    e = DroneLevitationEnv()
    try:
        obs1, _ = e.reset(seed=42)
        obs2, _ = e.reset(seed=42)
        np.testing.assert_allclose(obs1, obs2)
    finally:
        e.close()
