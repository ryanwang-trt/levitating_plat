"""Unit tests for reward.compute_reward (Free Float, Method B).

These pin down the reward *contract* so the upcoming 1M-step run is training
against the reward we think it is — and so the "Free Float default behavior"
doesn't silently change.
"""
import numpy as np
import pytest

from reward import compute_reward

# obs = [height, vz, roll, pitch, roll_rate, pitch_rate]
PERFECT_OBS = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # level, still, at target height
ZERO_ACTION = [0.0, 0.0, 0.0, 0.0]


def test_perfect_state_gives_alive_bonus_only():
    # Level + still + no throttle -> only the alive bonus remains (default 1.0).
    r = compute_reward(PERFECT_OBS, ZERO_ACTION)
    assert r == pytest.approx(1.0)


def test_empty_and_none_config_do_not_crash():
    # A missing/empty reward config must fall back to built-in defaults.
    assert compute_reward(PERFECT_OBS, ZERO_ACTION, None) == pytest.approx(1.0)
    assert compute_reward(PERFECT_OBS, ZERO_ACTION, {}) == pytest.approx(1.0)


def test_tilt_is_penalized_by_weight():
    # Asymmetric value chosen so alive_bonus and the tilt penalty do NOT cancel
    # to a round 0 — if either term were wrong, the net result would change.
    # roll=0.1, pitch=0.0 -> penalty = w_tilt * 0.1 = 5.0 * 0.1 = 0.5
    # reward = alive(1.0) - 0.5 = 0.5
    obs = [0.5, 0.0, 0.1, 0.0, 0.0, 0.0]
    r = compute_reward(obs, ZERO_ACTION)
    assert r == pytest.approx(0.5)


def test_tilt_penalty_uses_absolute_value():
    # Sign of tilt must not matter (|roll| + |pitch|).
    pos = compute_reward([0.5, 0.0, 0.2, -0.1, 0.0, 0.0], ZERO_ACTION)
    neg = compute_reward([0.5, 0.0, -0.2, 0.1, 0.0, 0.0], ZERO_ACTION)
    assert pos == pytest.approx(neg)


def test_vz_is_penalized():
    # |vz| = 0.5 -> penalty = w_vz * 0.5 = 2.0 * 0.5 = 1.0
    r = compute_reward([0.5, 0.5, 0.0, 0.0, 0.0, 0.0], ZERO_ACTION)
    assert r == pytest.approx(1.0 - 1.0)


def test_angular_rate_is_penalized():
    # |roll_rate| + |pitch_rate| = 0.3 + 0.2 -> 1.0 * 0.5 = 0.5
    r = compute_reward([0.5, 0.0, 0.0, 0.0, 0.3, -0.2], ZERO_ACTION)
    assert r == pytest.approx(1.0 - 0.5)


def test_energy_penalty_on_throttle():
    # sum(action**2) = 4 * 0.5**2 = 1.0 -> penalty = w_energy * 1.0 = 0.1
    r = compute_reward(PERFECT_OBS, [0.5, 0.5, 0.5, 0.5])
    assert r == pytest.approx(1.0 - 0.1)


def test_custom_weights_are_respected():
    cfg = {"alive_bonus": 2.0, "w_tilt": 10.0}
    # alive 2.0 - 10.0 * (0.1 + 0.0) = 2.0 - 1.0 = 1.0
    r = compute_reward([0.5, 0.0, 0.1, 0.0, 0.0, 0.0], ZERO_ACTION, cfg)
    assert r == pytest.approx(1.0)


def test_height_hold_disabled_by_default():
    # Default Free Float: being far from target height costs nothing.
    far = compute_reward([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], ZERO_ACTION)
    at = compute_reward([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], ZERO_ACTION)
    assert far == pytest.approx(at)


def test_height_hold_enabled_penalizes_height_error():
    cfg = {"height_hold": {"enabled": True, "weight": 10.0}}
    # height 0.5, target 0.5 (from target_height arg) -> no penalty
    r_at = compute_reward(PERFECT_OBS, ZERO_ACTION, cfg, target_height=0.5)
    # height 0.3, target 0.5 -> penalty 10.0 * 0.2 = 2.0
    r_off = compute_reward([0.3, 0.0, 0.0, 0.0, 0.0, 0.0], ZERO_ACTION, cfg, target_height=0.5)
    assert r_at == pytest.approx(1.0)
    assert r_off == pytest.approx(1.0 - 2.0)


def test_height_hold_target_falls_back_to_config_value():
    # No target_height arg -> uses height_hold.target_height (here 1.0).
    cfg = {"height_hold": {"enabled": True, "weight": 10.0, "target_height": 1.0}}
    r = compute_reward([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], ZERO_ACTION, cfg)
    assert r == pytest.approx(1.0)


def test_returns_python_float():
    r = compute_reward(PERFECT_OBS, ZERO_ACTION)
    assert isinstance(r, float)


def test_accepts_numpy_inputs():
    # The env passes numpy arrays; make sure that path works too.
    obs = np.array(PERFECT_OBS, dtype=np.float32)
    action = np.array(ZERO_ACTION, dtype=np.float32)
    assert compute_reward(obs, action) == pytest.approx(1.0)
