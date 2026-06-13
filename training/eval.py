# eval.py — 放在 training/ 目录下
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import time
import yaml
from stable_baselines3 import PPO
from env import DroneLevitationEnv
import pybullet as p

SCRIPT_DIR = Path(__file__).parent

with open(SCRIPT_DIR / "config.yaml") as f:
    config = yaml.safe_load(f) or {}

# GUI 模式起 env
env = DroneLevitationEnv(config=config, render_mode="human")
model = PPO.load(str(SCRIPT_DIR / "models/best_model"), device="cpu")

# 把相机拉近、对准平台悬停高度
p.resetDebugVisualizerCamera(
    cameraDistance=3,        # 离目标多远（小=近）
    cameraYaw=50,              # 水平绕轴角度
    cameraPitch=-15,           # 俯仰角
    cameraTargetPosition=[0, 0, 0.5],   # 对准点：平台的目标高度 0.5m
)

DT = 1 / 240                # PyBullet default timestep
GOAL_SECONDS = 4.0          # free-float duration we want to demonstrate
                            # (episode cap is 1000 steps @ 240Hz ≈ 4.17s)
DEMO_VZ = 2.0

passes = 0
for ep in range(5):
    obs, _ = env.reset()
    # Initial disturbance this episode started from (so we can see it recover).
    if DEMO_VZ is not None:
        p.resetBaseVelocity(
            env.drone_id,
            linearVelocity=[0, 0, DEMO_VZ],
            angularVelocity=[0, 0, 0], 
        )
        obs = env._get_obs() # 覆盖后重新读 obs，保证 start 记录准确
    start_vz, start_roll, start_pitch = obs[1], obs[2], obs[3]

    done = False
    ep_rew = 0.0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_rew += reward
        steps += 1
        time.sleep(DT)      # 放慢到肉眼能看(PyBullet 默认 240Hz)
        done = terminated or truncated

    # It floated for the whole episode only if it was truncated (hit the step
    # cap) rather than terminated (crashed / tipped / flew off).
    held = steps * DT
    survived = truncated and not terminated
    ok = survived and held >= GOAL_SECONDS
    passes += ok
    print(f"episode {ep}: {'PASS' if ok else 'FAIL'}  "
          f"held={held:5.2f}s / {GOAL_SECONDS:.0f}s  reward={ep_rew:7.1f}  "
          f"start[vz={start_vz:+.2f} roll={start_roll:+.2f} pitch={start_pitch:+.2f}]  "
          f"final[h={obs[0]:.3f} roll={obs[2]:+.3f} pitch={obs[3]:+.3f}]")

print(f"\n{passes}/5 episodes held a free float for >= {GOAL_SECONDS:.0f}s")
env.close()