import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #for mac

import argparse
from pathlib import Path

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from env import DroneLevitationEnv

# Resolve everything relative to this script so training does not depend on the
# current working directory.
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


class FixedSeedReset(gym.Wrapper):
    """Force every reset() to use the SAME seed -> identical disturbance sequence.

    EvalCallback calls eval_env.reset() with no seed before each evaluation. With
    a single up-front seeding, the env's RNG stream just keeps advancing, so each
    evaluation draws a DIFFERENT disturbance/init sequence. Verified: the same
    checkpoint then scores ~33 reward apart between two back-to-back evals — pure
    noise that corrupts which checkpoint EvalCallback saves as "best".

    Wrapping the eval env so every reset reseeds to a fixed value makes the
    evaluation deterministic: reward differences across checkpoints reflect the
    POLICY, not which sequence it happened to draw. (Don't use this on the
    TRAINING env — there we want fresh sequences every episode.)
    """

    def __init__(self, env, seed):
        super().__init__(env)
        self._fixed_seed = seed

    def reset(self, **kwargs):
        kwargs["seed"] = self._fixed_seed
        return self.env.reset(**kwargs)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def main(timesteps_override=None, init_from=None, stage=None, lr_override=None,
         dist_stage=None):
    config = load_config()
    train_cfg = config.get("training", {})

    # Disturbance curriculum: --dist-stage {1,2} overrides the injection
    # force/duration AND the air damping with the chosen entry from
    # config.disturbance_stages (1-indexed). Harder stages need stiffer damping
    # to bleed off the larger residual horizontal momentum (stage1 0.5, stage2
    # 1.0). prob/grace stay shared from disturbance_injection. Omitted -> use
    # config as-is (stage 1 force/duration + default physics damping).
    if dist_stage is not None:
        stages = config.get("disturbance_stages", []) or []
        if not 1 <= dist_stage <= len(stages):
            raise SystemExit(
                f"--dist-stage {dist_stage} out of range; config defines "
                f"{len(stages)} disturbance_stages (1..{len(stages)})"
            )
        chosen = stages[dist_stage - 1]
        di = config.setdefault("disturbance_injection", {})
        for k in ("force_min", "force_max", "duration_min", "duration_max"):
            di[k] = chosen[k]
        # Per-stage damping override (falls back to the physics default if a
        # stage doesn't specify one).
        if "linear_damping" in chosen:
            phys = config.setdefault("physics", {})
            phys["linear_damping"] = chosen["linear_damping"]
            phys["angular_damping"] = chosen["linear_damping"]
        print(f"disturbance curriculum stage {dist_stage}: "
              f"force {di['force_min']}-{di['force_max']}N, "
              f"duration {di['duration_min']}-{di['duration_max']} frames, "
              f"damping {config.get('physics', {}).get('linear_damping')}")

    # Hyperparameters (fall back to defaults if a key is missing).
    total_timesteps = train_cfg.get("total_timesteps", 1_000_000)
    if timesteps_override is not None:
        total_timesteps = timesteps_override
    learning_rate = train_cfg.get("learning_rate", 3e-4)
    if lr_override is not None:
        learning_rate = lr_override
    n_steps = train_cfg.get("n_steps", 2048)
    batch_size = train_cfg.get("batch_size", 64)
    seed = train_cfg.get("seed", 0)

    # Every run writes into one self-contained directory: models/<stage>/.
    #   models/<stage>/final.zip  -> policy at the moment training stopped
    #   models/<stage>/best.zip   -> highest-eval-reward checkpoint (EvalCallback)
    #   models/<stage>/evaluations.npz -> eval history for that run
    # --stage names the run (default training.stage), so a stage-2 fine-tune
    # lands in models/stage2/ and never clobbers models/stage1/.
    stage_name = stage if stage is not None else train_cfg.get("stage", "stage1")
    stage_dir = SCRIPT_DIR / "models" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_path = stage_dir / "final"          # -> models/<stage>/final.zip
    best_model_dir = stage_dir               # EvalCallback writes best_model.zip here

    # Pass the full config so the env can read its reward section.
    env = DroneLevitationEnv(config=config)
    # Monitor tracks episode returns/lengths so PPO prints rollout/ep_rew_mean.
    env = Monitor(env)

    # Separate eval env: same config (keeps domain_randomization on), but EVERY
    # evaluation must replay the SAME disturbance/init sequence, so a new high
    # score means the policy genuinely recovers better — not that it drew an
    # easier sequence. FixedSeedReset forces every reset (EvalCallback issues one
    # per eval episode with no seed) to reuse the same seed; without it the RNG
    # stream advances and each evaluation sees a different sequence (verified to
    # swing the same checkpoint's score by ~33 reward -> corrupts best selection).
    eval_env = Monitor(FixedSeedReset(DroneLevitationEnv(config=config), seed=seed))
    eval_env.reset()

    # EvalCallback evaluates the current policy every eval_freq steps and saves
    # best_model.zip only when the mean eval reward sets a new high. So however
    # training ends, this is the best policy seen over the run. SB3 hardcodes the
    # filename "best_model.zip"; we rename it to best.zip after learn() for a
    # consistent models/<stage>/best.zip layout.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),  # -> models/<stage>/best_model.zip
        log_path=str(best_model_dir),
        eval_freq=25_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    if init_from is not None:
        # Fine-tune: load existing weights and keep training (stage 2 starts from
        # stage 1's attitude-recovery policy). obs/action spaces must match — they
        # do, since stage 2 only changes physics, not the 6-dim obs or action.
        init_path = SCRIPT_DIR / init_from if not os.path.isabs(init_from) else Path(init_from)
        print(f"fine-tuning from {init_path}")
        model = PPO.load(
            str(init_path),
            env=env,
            device="cpu",
            tensorboard_log=str(SCRIPT_DIR / "tb_logs"),
        )
        # PPO.load restores the SAVED learning rate (3e-4 from stage 1) and
        # ignores the constructor's. Fine-tuning a converged policy at that rate
        # tends to wreck it, so override both the value and the schedule PPO
        # actually reads each update (a constant-lr lambda).
        model.learning_rate = learning_rate
        model.lr_schedule = lambda _progress_remaining: learning_rate
        print(f"fine-tune learning rate set to {learning_rate}")
    else:
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

    # Normalize SB3's hardcoded best_model.zip -> best.zip. It may be absent if
    # no eval beat the initial -inf (e.g. a very short smoke-test run).
    sb3_best = best_model_dir / "best_model.zip"
    best_path = best_model_dir / "best.zip"
    if sb3_best.exists():
        sb3_best.replace(best_path)

    print(f"final model saved to {save_path}.zip")
    if best_path.exists():
        print(f"best model (highest eval reward) saved to {best_path}")

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
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="path to a .zip to fine-tune from (e.g. models/stage1/best.zip). "
             "Omit to train from scratch.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="run name -> models/<stage>/{final,best}.zip (default training.stage, "
             "e.g. 'stage2' so a fine-tune doesn't overwrite stage1).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="override learning rate (e.g. 3e-5 for a gentle fine-tune that "
             "doesn't wreck the loaded policy).",
    )
    parser.add_argument(
        "--dist-stage",
        type=int,
        default=None,
        help="disturbance curriculum stage (1/2) -> picks force/duration/damping "
             "from config.disturbance_stages. Omit to use config as-is.",
    )
    args = parser.parse_args()
    main(
        timesteps_override=args.timesteps,
        init_from=args.init_from,
        stage=args.stage,
        lr_override=args.lr,
        dist_stage=args.dist_stage,
    )
