import numpy as np


def compute_reward(obs, action, config=None, target_height=None):
    """Method B (Free Float) reward.

    The platform is rewarded for staying alive, level, and still — but NOT for
    returning to any particular height/position. An optional height-hold term
    (disabled by default) turns this into a hybrid mode without code changes.

    obs    : [height, vz, roll, pitch, roll_rate, pitch_rate]
    action : [m1, m2, m3, m4] in [0, 1]
    config : the "reward" sub-dict from config.yaml. Missing keys fall back to
             the built-in defaults below, so a missing/empty config never crashes.
    """
    cfg = config or {}
    obs = np.asarray(obs, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)

    height, vz, roll, pitch, roll_rate, pitch_rate = obs

    # ---- Core Free Float terms (always active) ----
    alive_bonus = cfg.get("alive_bonus", 1.0)
    w_tilt = cfg.get("w_tilt", 5.0)
    w_vz = cfg.get("w_vz", 2.0)
    w_ang_rate = cfg.get("w_ang_rate", 1.0)
    w_energy = cfg.get("w_energy", 0.1)

    reward = alive_bonus
    reward -= w_tilt * (abs(roll) + abs(pitch))                 # stay level
    reward -= w_vz * abs(vz)                                    # suppress vertical motion
    reward -= w_ang_rate * (abs(roll_rate) + abs(pitch_rate))  # suppress rotation
    reward -= w_energy * float(np.sum(action ** 2))            # penalize throttle effort

    # ---- Optional height-hold term (off by default -> pure Free Float) ----
    height_hold = cfg.get("height_hold", {}) or {}
    if height_hold.get("enabled", False):
        if target_height is not None:
            target = target_height
        else:
            target = height_hold.get("target_height", 0.5)
        hh_weight = height_hold.get("weight", 10.0)
        reward -= hh_weight * abs(height - target)

    return float(reward)
