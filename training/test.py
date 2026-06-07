from env import DroneLevitationEnv
import numpy as np
import time

env = DroneLevitationEnv(render_mode="human")
obs, info = env.reset()
print("start:", obs)

# Below hover (0.5 = hover), so it slowly sinks
drop_action = np.array([0.45, 0.45, 0.45, 0.45], dtype=np.float32)

for step in range(500):
    obs, reward, terminated, truncated, info = env.step(drop_action)
    print(f"step {step}: height={obs[0]:.3f}  vz={obs[1]:.3f}")
    time.sleep(1/100)
    if terminated or truncated:
        print("episode ended:", "terminated" if terminated else "truncated")
        break

env.close()