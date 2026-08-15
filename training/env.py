import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

from reward import compute_reward

# Set to 200 Hz to match the on-hardware control loop (firmware imu_read runs a 200 Hz
# k_timer loop)

# Training and deployment MUST share a timestep
CONTROL_HZ = 200


class DroneLevitationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, config=None):
        super().__init__()

        # Reward config (the "reward" section of config.yaml). Read once here so
        # step() doesn't re-parse it every tick. Empty dict -> compute_reward uses
        # its built-in Free Float defaults.
        config = config or {}
        self.reward_cfg = config.get("reward", {})

        # Domain randomization config (the "domain_randomization" section). Read
        # once here so reset() doesn't re-parse it each episode. When disabled (or
        # absent) reset() starts the platform from the old perfect, still state.
        self.dr_cfg = config.get("domain_randomization", {}) or {}

        # Stage 2: in-flight disturbance injection (the "disturbance_injection"
        # section). Each step has a small probability of triggering a random
        # external shove that is then HELD for a random number of frames (so the
        # training impulse range covers a sustained push, not just a single-frame
        # tap). Read once here so step() doesn't re-parse it every tick. Disabled
        # (or absent) -> no shoves, i.e. plain Free Float physics.
        self.dist_cfg = config.get("disturbance_injection", {}) or {}

        # Physics config (the "physics" section): air drag (linear/angular
        # damping). Read once here so reset() doesn't re-parse it each episode.
        # Absent/0 -> undamped (the old behavior).
        self.phys_cfg = config.get("physics", {}) or {}

        # Post-shove reward grace window. While a shove is active AND for this
        # many frames after it ends, the attitude penalties (tilt + ang_rate) are
        # waived in compute_reward — being knocked off-level is the disturbance's
        # doing, not the policy's, so we don't punish the policy for *being* hit;
        # we only resume penalizing once the grace window expires, which is what
        # rewards actually RECOVERING. alive_bonus + energy stay active throughout
        # so "die early to stop the bleeding" is no longer the optimal move.
        # 0 -> no grace (old behavior). Reset per episode.
        self._grace_frames = int(self.dist_cfg.get("grace_frames", 0))
        self._grace_remaining = 0           # frames of attitude-penalty grace left

        # State for an in-progress sustained shove (set on trigger, decremented
        # each frame until it expires). Reset per episode in reset().
        self._shove_remaining = 0           # frames left in the current shove
        self._shove_force = None            # [fx, fy, fz] held constant this shove
        self._shove_offset = None           # application point, held constant

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
        # resetSimulation() reverts the timestep to PyBullet's 240 Hz default, so
        # re-assert the 200 Hz control rate every episode (see CONTROL_HZ).
        p.setTimeStep(1.0 / CONTROL_HZ)

        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create the platform as a simple box (no URDF file needed)
        half_extents = [0.15, 0.15, 0.02]   # 30cm x 30cm x 4cm platform

        # Collision and Visualization
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                    rgbaColor=[0.2, 0.4, 0.8, 1])

        # ---- Sample initial disturbance (domain randomization) ----
        # Every episode starts off-nominal so the policy must LEARN to recover
        # into a free float rather than just holding a perfect hover. Disabled
        # -> all-zero disturbance, i.e. the old perfect, still start.
        if self.dr_cfg.get("enabled", False):
            tilt = np.deg2rad(self.dr_cfg.get("tilt_range_deg", 15.0))
            vz_r = self.dr_cfg.get("vz_range", 1.0)
            ang_r = self.dr_cfg.get("ang_rate_range", 1.0)

            init_roll = self.np_random.uniform(-tilt, tilt)
            init_pitch = self.np_random.uniform(-tilt, tilt)
            init_vz = self.np_random.uniform(-vz_r, vz_r)
            init_roll_rate = self.np_random.uniform(-ang_r, ang_r)
            init_pitch_rate = self.np_random.uniform(-ang_r, ang_r)
        else:
            init_roll = init_pitch = 0.0
            init_vz = 0.0
            init_roll_rate = init_pitch_rate = 0.0

        start_pos = [0, 0, self.target_height]
        start_orientation = p.getQuaternionFromEuler([init_roll, init_pitch, 0])
        self.drone_id = p.createMultiBody(
            baseMass=1.0,                       # 1 kg platform
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=start_pos,
            baseOrientation=start_orientation,
        )

        # Air drag. PyBullet bodies are undamped by default, so any horizontal
        # velocity a shove imparts NEVER decays — the platform drifts at constant
        # speed forever, and that residual momentum (invisible to the policy:
        # vx/vy aren't in obs and can't be measured on the real hardware)
        # eventually destabilizes it ~150 frames after a push. A real platform of
        # this size moving through air feels drag that bleeds that momentum off;
        # modeling it lets the policy stay stable on IMU-only obs (attitude +
        # rates + vz), exactly what it has on hardware. 0 -> undamped.
        lin_damp = float(self.phys_cfg.get("linear_damping", 0.0))
        ang_damp = float(self.phys_cfg.get("angular_damping", 0.0))
        p.changeDynamics(
            self.drone_id, -1,
            linearDamping=lin_damp,
            angularDamping=ang_damp,
        )

        # Apply the sampled initial velocities (createMultiBody starts at rest).
        p.resetBaseVelocity(
            self.drone_id,
            linearVelocity=[0, 0, init_vz],
            angularVelocity=[init_roll_rate, init_pitch_rate, 0],
        )

        self.step_count = 0

        # Clear any in-progress sustained shove so it doesn't leak across episodes.
        self._shove_remaining = 0
        self._shove_force = None
        self._shove_offset = None
        self._grace_remaining = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _apply_disturbance(self, force, offset):
        """Inject one external shove on the platform.

        force  : [fx, fy, fz] in the platform's LINK frame (Newtons). Both force
                 and application point are in the link frame so posObj is a simple
                 offset from the base (in WORLD_FRAME, posObj would be an absolute
                 world coordinate, not an offset, and the torque arm would be
                 wrong). The trade-off is the force direction rotates slightly
                 with attitude; at the small tilt angles seen here that's
                 negligible, and it reads as "a push relative to the platform".
        offset : [x, y, z] application point relative to the base. Offsetting from
                 the center of mass makes the shove both translate the platform
                 AND torque it off-level — exactly the disturbance it must learn
                 to correct.
        """
        p.applyExternalForce(
            self.drone_id,
            -1,                        # apply to base body
            forceObj=list(force),
            posObj=list(offset),
            flags=p.LINK_FRAME,
        )

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

        # ---- 1b. Stage 2: random in-flight disturbance injection ----
        # With small per-step probability, shove the platform with a random
        # all-directions force (incl. horizontal x/y) applied off-center, so it
        # gets both knocked sideways AND tilted. The shove is then HELD constant
        # for a random number of frames: a sustained push of magnitude F for D
        # frames delivers impulse F * dt * D, so a wide (force x duration) range
        # covers everything from a single-frame tap to a long, hard shove (incl.
        # the sustained pushes used in eval). obs/reward are untouched: the policy
        # only sees the resulting attitude change and must level back out.
        if self.dist_cfg.get("enabled", False):
            # Only roll for a NEW shove when none is in progress, so prob is the
            # probability of *starting* a shove (not re-rolled mid-push). A shove
            # of duration D starting every ~1/prob steps -> avg D*prob duty cycle.
            if self._shove_remaining <= 0 and self.np_random.random() < self.dist_cfg.get("prob", 0.015):
                fmin = self.dist_cfg.get("force_min", 2.0)
                fmax = self.dist_cfg.get("force_max", 8.0)
                dmin = int(self.dist_cfg.get("duration_min", 1))
                dmax = int(self.dist_cfg.get("duration_max", 1))
                # Random direction on the unit sphere * random magnitude.
                direction = self.np_random.normal(size=3)
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm
                magnitude = self.np_random.uniform(fmin, fmax)
                # Force, application point, and duration are sampled ONCE and held
                # for the whole shove (matches a real sustained push). Offset is
                # within the platform footprint so the shove also torques it
                # off-level (half_extents are 0.15, 0.15).
                self._shove_force = direction * magnitude
                self._shove_offset = [
                    self.np_random.uniform(-0.15, 0.15),
                    self.np_random.uniform(-0.15, 0.15),
                    0.0,
                ]
                self._shove_remaining = self.np_random.integers(dmin, dmax + 1)

            # Apply the held force this frame (if a shove is active) and decrement.
            if self._shove_remaining > 0:
                self._apply_disturbance(self._shove_force, self._shove_offset)
                self._shove_remaining -= 1
                # Being shoved (re)arms the attitude-penalty grace window: it stays
                # open through the shove and for _grace_frames after the last push,
                # so the policy isn't punished for the tilt the shove forced on it.
                self._grace_remaining = self._grace_frames

        # ---- 2. Step the physics forward ----
        p.stepSimulation()

        # ---- 3. Read the new state ----
        obs = self._get_obs()

        # ---- 4. Reward (Method B Free Float, see reward.py) ----
        # During the grace window, waive the tilt/ang_rate penalties: the policy
        # shouldn't be charged for being knocked off-level by the disturbance,
        # only for failing to recover once grace ends.
        attitude_grace = self._grace_remaining > 0
        reward = compute_reward(
            obs, action, self.reward_cfg, self.target_height,
            attitude_grace=attitude_grace,
        )
        if self._grace_remaining > 0:
            self._grace_remaining -= 1

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
