import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #for mac

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from env import DroneLevitationEnv

# Resolve everything relative to this script so training does not depend on the
# current working directory.
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def main(timesteps_override=None):
    config = load_config()
    train_cfg = config.get("training", {})

    # Hyperparameters (fall back to defaults if a key is missing).
    total_timesteps = train_cfg.get("total_timesteps", 1_000_000)
    if timesteps_override is not None:
        total_timesteps = timesteps_override
    learning_rate = train_cfg.get("learning_rate", 3e-4)
    n_steps = train_cfg.get("n_steps", 2048)
    batch_size = train_cfg.get("batch_size", 64)
    seed = train_cfg.get("seed", 0)
    save_path = SCRIPT_DIR / train_cfg.get("save_path", "models/policy_v1")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass the full config so the env can read its reward section.
    env = DroneLevitationEnv(config=config)
    # Monitor tracks episode returns/lengths so PPO prints rollout/ep_rew_mean.
    env = Monitor(env)

    # Separate eval env: same config (keeps domain_randomization on), but every
    # evaluation is seeded identically so the disturbance sequence is fixed.
    # A new high score then means the policy genuinely recovers better, not that
    # it drew easier initial conditions.
    eval_env = Monitor(DroneLevitationEnv(config=config))
    # Seed the eval env's RNG once. Gymnasium dropped Env.seed(); seeding goes
    # through reset(seed=...), which initializes self.np_random. EvalCallback's
    # later reset() calls pass no seed, so the RNG continues deterministically
    # from this seeded stream -> a fixed disturbance sequence across evaluations.
    eval_env.reset(seed=seed)

    # EvalCallback evaluates the current policy every eval_freq steps and saves
    # best_model.zip only when the mean eval reward sets a new high. So however
    # training ends, models/best_model.zip is the best policy seen over the run.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path.parent),  # -> models/best_model.zip
        log_path=str(save_path.parent),
        eval_freq=25_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        device="cpu",          # small MLP on M1 -> CPU is fine, avoids MPS overhead
        verbose=1,
        tensorboard_log=str(SCRIPT_DIR / "tb_logs"),
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(str(save_path))
    print(f"saved model to {save_path}.zip")
    print(f"best model (highest eval reward) saved to "
          f"{save_path.parent / 'best_model'}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on DroneLevitationEnv")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="override training.total_timesteps (e.g. 20000 for a smoke test)",
    )
    args = parser.parse_args()
    main(timesteps_override=args.timesteps)
