import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

from reward import compute_reward


class DroneLevitationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, config=None):
        super().__init__()

        # Reward config (the "reward" section of config.yaml). Read once here so
        # step() doesn't re-parse it every tick. Empty dict -> compute_reward uses
        # its built-in Free Float defaults.
        config = config or {}
        self.reward_cfg = config.get("reward", {})

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

        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # ---- 0. Clip action into valid [0, 1] range ----
        # PPO's Gaussian policy can sample outside the action_space bounds
        # (especially early in training), which would otherwise produce
        # negative thrust or more than max thrust.
        action = np.clip(action, 0.0, 1.0)

        # ---- 1. Convert action [0~1] × 4 into upward forces at 4 corners ----
        # Hover force: platform mass (1kg) × gravity (9.81) = 9.81N total to hold still
        # Split across 4 motors, and let max thrust be ~2× hover so it can climb
        max_thrust_per_motor = (1.0 * 9.81 / 4) * 2.0   # ~4.9 N per motor at full throttle

        # 4 corners coordinates of the platform (matches half_extents 0.15 x 0.15)
        corners = [
            [ 0.15,  0.15, 0],
            [-0.15,  0.15, 0],
            [-0.15, -0.15, 0],
            [ 0.15, -0.15, 0],
        ]

        for i in range(4):
            thrust = float(action[i]) * max_thrust_per_motor # 0~1 → 0~max Newtons
            force = [0, 0, thrust] # straight up 
            p.applyExternalForce(
                self.drone_id,
                -1,                       # -1 = apply to base body
                forceObj=force,           # amount of force applied
                posObj=corners[i],        # at this corner
                flags=p.LINK_FRAME,       # relative to platform's own orientation
            )

        # ---- 2. Step the physics forward ----
        p.stepSimulation()

        # ---- 3. Read the new state ----
        obs = self._get_obs()

        # ---- 4. Reward (Method B Free Float, see reward.py) ----
        reward = compute_reward(obs, action, self.reward_cfg, self.target_height)

        # ---- 5. Episode termination ----
        height = obs[0]
        roll, pitch = obs[2], obs[3]
        max_tilt = np.deg2rad(80)   # tipped over at 80 deg
        terminated = bool(
            height < 0.05 or height > 2.5            # crashed or flew too high
            or abs(roll) > max_tilt or abs(pitch) > max_tilt  # tipped over
        )

        self.step_count += 1
        truncated = bool(self.step_count >= 1000)          # max episode length

        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        p.disconnect(self.physics_client)
