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

    def _get_obs(self):
        # Position and orientation (quaternion) of the drone
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        # Linear and angular velocity
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)

        # Convert quaternion → euler angles (roll, pitch, yaw)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        height = pos[2]          # z position
        vz = lin_vel[2]          # vertical velocity
        roll_rate = ang_vel[0]   # angular velocity around x
        pitch_rate = ang_vel[1]  # angular velocity around y

        obs = np.array([
            height, vz, roll, pitch, roll_rate, pitch_rate
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create the platform as a simple box (no URDF file needed)
        half_extents = [0.15, 0.15, 0.02]   # 30cm x 30cm x 4cm platform

        # Collision and Visualization
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                    rgbaColor=[0.2, 0.4, 0.8, 1])

        start_pos = [0, 0, self.target_height]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.drone_id = p.createMultiBody(
            baseMass=1.0,                       # 1 kg platform
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=start_pos,
            baseOrientation=start_orientation,
        )

        obs = self._get_obs()
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
