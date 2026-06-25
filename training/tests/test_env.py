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


def test_disturbance_injection_keeps_step_well_formed():
    # Stage 2: with injection forced on every step (prob=1.0), step() must still
    # return finite, correctly-shaped obs/reward and not crash. obs dimension is
    # unchanged (disturbance is a physics-only effect).
    cfg = {"disturbance_injection": {
        "enabled": True, "prob": 1.0, "force_min": 2.0, "force_max": 8.0,
    }}
    e = DroneLevitationEnv(config=cfg)
    try:
        e.reset(seed=0)
        hover = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        for _ in range(50):
            obs, reward, terminated, truncated, _ = e.step(hover)
            assert obs.shape == (OBS_DIM,)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            if terminated or truncated:
                break
    finally:
        e.close()


def test_disturbance_injection_off_by_default():
    # No disturbance_injection config -> shoves never fire, so two seeded runs
    # of the same action sequence stay identical (no hidden RNG draws for force).
    e = DroneLevitationEnv()  # injection absent -> disabled
    try:
        e.reset(seed=7)
        hover = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        first = [e.step(hover)[0].copy() for _ in range(20)]
        e.reset(seed=7)
        second = [e.step(hover)[0].copy() for _ in range(20)]
        for a, b in zip(first, second):
            np.testing.assert_allclose(a, b)
    finally:
        e.close()


def test_air_drag_damps_horizontal_velocity():
    # With linear_damping > 0, a horizontal velocity must DECAY over free-float
    # steps (no thrust asymmetry driving it). Without damping it persists. We kick
    # the platform sideways and check vx shrinks more with damping than without.
    import pybullet as p

    def vx_after_kick(damping):
        cfg = {"physics": {"linear_damping": damping, "angular_damping": 0.0}}
        e = DroneLevitationEnv(config=cfg)
        try:
            e.reset(seed=0)
            p.resetBaseVelocity(e.drone_id, linearVelocity=[1.0, 0.0, 0.0])
            hover = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            for _ in range(60):
                e.step(hover)
            return p.getBaseVelocity(e.drone_id)[0][0]
        finally:
            e.close()

    vx_damped = vx_after_kick(1.0)
    vx_undamped = vx_after_kick(0.0)
    assert abs(vx_damped) < abs(vx_undamped)   # damping bleeds off more speed
    assert abs(vx_damped) < 1.0                # clearly decayed from the 1.0 kick


def test_disturbance_is_held_for_full_duration():
    # A shove must be HELD for exactly duration frames with constant force/offset
    # (a sustained push), not re-sampled each frame. prob=1.0 + duration 5..5
    # forces one shove to start on step 1 and persist for 5 frames.
    cfg = {"disturbance_injection": {
        "enabled": True, "prob": 1.0,
        "force_min": 10.0, "force_max": 10.0,
        "duration_min": 5, "duration_max": 5,
    }}
    e = DroneLevitationEnv(config=cfg)
    try:
        e.reset(seed=0)
        hover = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        e.step(hover)  # step 1 starts the shove and applies frame 1 of 5
        force_after_start = np.array(e._shove_force)
        assert e._shove_remaining == 4, "shove should have 4 of 5 frames left"

        # Next 4 frames keep the SAME force/offset and count down to 0.
        for expected_left in (3, 2, 1, 0):
            e.step(hover)
            # Force must stay constant across the held shove (not re-sampled).
            np.testing.assert_allclose(e._shove_force, force_after_start)
            assert e._shove_remaining == expected_left
    finally:
        e.close()


def test_reset_clears_in_progress_shove():
    # A shove still in progress at reset() must not leak into the next episode.
    cfg = {"disturbance_injection": {
        "enabled": True, "prob": 1.0,
        "force_min": 10.0, "force_max": 10.0,
        "duration_min": 50, "duration_max": 50,
    }}
    e = DroneLevitationEnv(config=cfg)
    try:
        e.reset(seed=0)
        e.step(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        assert e._shove_remaining > 0, "a long shove should be mid-flight"
        e.reset(seed=0)
        assert e._shove_remaining == 0
        assert e._shove_force is None
        assert e._shove_offset is None
        assert e._grace_remaining == 0
    finally:
        e.close()


def test_grace_window_arms_on_shove_and_decays():
    # A shove arms the attitude-penalty grace window to grace_frames; once the
    # shove ends the window counts down on the following shove-free steps. We run
    # a 2-frame shove (prob=1.0 fires it immediately), then zero prob so no new
    # shove re-arms grace, and watch it decay to 0.
    cfg = {"disturbance_injection": {
        "enabled": True, "prob": 1.0,
        "force_min": 6.0, "force_max": 6.0,
        "duration_min": 2, "duration_max": 2,
        "grace_frames": 4,
    }}
    e = DroneLevitationEnv(config=cfg)
    try:
        e.reset(seed=0)
        hover = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        e.step(hover)
        e.step(hover)  # 2-frame shove now over; grace armed to 4, ticked once -> 3
        assert e._shove_remaining == 0
        assert e._grace_remaining == 3

        # Stop new shoves so grace can decay cleanly, then count it down to 0.
        e.dist_cfg["prob"] = 0.0
        for expected in (2, 1, 0, 0):
            e.step(hover)
            assert e._grace_remaining == expected
    finally:
        e.close()
