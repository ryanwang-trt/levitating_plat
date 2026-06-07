from env import DroneLevitationEnv
import time

env = DroneLevitationEnv(render_mode="human")
obs, info = env.reset()
print("obs:", obs)

env.close()