# eval.py — 放在 training/ 目录下
# Stage 2 GUI eval: 在确定步数施加确定大小的外力(含水平方向),观察平台
# 「被推 → 扳正 → 保持 level」的过程。obs = [height, vz, roll, pitch, roll_rate, pitch_rate]
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import time
import yaml
import numpy as np
from stable_baselines3 import PPO
from env import DroneLevitationEnv
import pybullet as p

SCRIPT_DIR = Path(__file__).parent

# Which model to eval. Accepts a bare stage name (e.g. "stage2b" -> loads
# models/stage2b/best) or a full path/zip. Defaults to the last curriculum
# stage. Usage:  python eval.py            (defaults to stage2b)
#                python eval.py stage2a     (a specific stage's best)
#                python eval.py models/stage1/final   (any path)
parser = argparse.ArgumentParser(description="GUI eval of a trained policy")
parser.add_argument(
    "model", nargs="?", default="stage2b",
    help="stage name (-> models/<name>/best) or a path to a model .zip "
         "(default: stage2b, the last curriculum stage)",
)
args = parser.parse_args()

# A bare name with no path separator and no .zip is treated as a stage name.
arg = args.model
if "/" not in arg and "\\" not in arg and not arg.endswith(".zip"):
    model_path = SCRIPT_DIR / "models" / arg / "best"
else:
    model_path = Path(arg) if os.path.isabs(arg) else SCRIPT_DIR / arg

with open(SCRIPT_DIR / "config.yaml") as f:
    config = yaml.safe_load(f) or {}

# Turn OFF the env's *random* in-flight injection for eval: we want deterministic,
# repeatable shoves at known steps so the recover trajectory is clean to read.
# (We call env._apply_disturbance() ourselves below.)
config.setdefault("disturbance_injection", {})["enabled"] = False

# GUI 模式起 env
env = DroneLevitationEnv(config=config, render_mode="human")
print(f"loading model: {model_path}")
model = PPO.load(str(model_path), device="cpu")

# 把相机拉近、对准平台悬停高度
p.resetDebugVisualizerCamera(
    cameraDistance=3,        # 离目标多远（小=近）
    cameraYaw=50,            # 水平绕轴角度
    cameraPitch=-15,         # 俯仰角
    cameraTargetPosition=[0, 0, 0.5],   # 对准点：平台的目标高度 0.5m
)

DT = 1 / 240                # PyBullet default timestep

# Deterministic shoves: {start_step: [fx, fy, fz]} in the platform link frame.
# NOTE: these are sized to match the current TRAINING disturbance range so the
# demo stays IN-distribution — the policy is being shown the kind of push it
# actually trained on (config disturbance_injection: 4-8 N held 1-10 frames).
# Each shove here is ~8 N held for SHOVE_DURATION frames, i.e. the upper end of
# what it saw, so a failure means a genuinely weak policy, not an out-of-range
# eval. (Crank these back up once the policy handles harder shoves.)
# ONE shove per episode, early (step 30), so each episode is a single clean
# "knocked off-level -> driven back to level" recovery you can watch start to
# finish with nothing else interrupting. The three episodes use different push
# directions (per EPISODE_SHOVE below); the rest of each 1000-step episode is
# pure recovery + free float.
SHOVE_STEP = 200                    # the single shove fires at this step each episode
# One push direction per episode (ep 0, 1, 2):
EPISODE_SHOVE = [
    [3.0, 3.0, 6.0],               # ep 0: push +x
    [5.0, 0.0, 5.0],              # ep 1: push -y, slight up
    [0.0, 0.0, 14.0],              # ep 2: push diagonally
]
SHOVE_DURATION = 10                 # hold the shove this many steps (~0.042s @ 240Hz)
SHOVE_OFFSET = [0.12, 0.12, 0.0]   # off-center -> also torques the platform
TRACE_WINDOW = 200                  # steps of roll/pitch to print after the shove starts

for ep in range(3):
    obs, _ = env.reset()
    done = False
    steps = 0
    trace_until = -1          # while steps < this, keep printing roll/pitch
    last_shove_step = None
    active_force = None        # force being applied this tick (None = no shove)
    shove_remaining = 0        # steps left in the current sustained shove

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # The single shove for this episode starts here and is HELD for
        # SHOVE_DURATION steps. Direction depends on which episode we're in.
        if steps == SHOVE_STEP:
            active_force = EPISODE_SHOVE[ep]
            shove_remaining = SHOVE_DURATION
            mag = float(np.linalg.norm(active_force))
            print(f"  [ep {ep}] SHOVE @ step {steps}: force={active_force} |F|={mag:.1f}N "
                  f"x{SHOVE_DURATION} steps  (roll={obs[2]:+.3f} pitch={obs[3]:+.3f} before)")
            trace_until = steps + TRACE_WINDOW
            last_shove_step = steps

        # Apply the held force every step of its window, BEFORE stepping, so it
        # lands in this tick's physics (same ordering as the env's own path).
        if shove_remaining > 0:
            env._apply_disturbance(active_force, SHOVE_OFFSET)
            shove_remaining -= 1

        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1

        # Print the roll/pitch trajectory for a window after each shove so the
        # "knocked off-level -> driven back to ~0" recovery is visible.
        if steps <= trace_until and (steps - last_shove_step) % 10 == 0:
            print(f"        +{steps - last_shove_step:3d} steps: "
                  f"roll={obs[2]:+.3f} pitch={obs[3]:+.3f}")

        time.sleep(DT)      # 放慢到肉眼能看(PyBullet 默认 240Hz)
        done = terminated or truncated

    # Success = survived the full episode (truncated at the step cap) without
    # terminating (crash / tip-over). Horizontal drift is irrelevant by design.
    survived = truncated and not terminated
    print(f"episode {ep}: {'PASS' if survived else 'FAIL'}  "
          f"steps={steps} (truncated={truncated} terminated={terminated})  "
          f"final[h={obs[0]:.3f} roll={obs[2]:+.3f} pitch={obs[3]:+.3f}]\n")

env.close()
