import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data


class DroneLevitationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        #The observation space is defined as height，vz (vertical speed) from ToF
        #                                    roll （side tilt), pitch(front tilt), roll_rate, pitch_rate
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, -np.pi, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi, np.pi, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        #The action space is defined as [m1, m2, m3, m4], each linking with 4 motor commands
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.target_height = 0.5  # meters

        # PyBullet setup
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)  # headless, faster for training

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # placeholders, will be loaded in reset()
        self.plane_id = None
        self.drone_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros(6, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        obs = np.zeros(6, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        p.disconnect(self.physics_client)
