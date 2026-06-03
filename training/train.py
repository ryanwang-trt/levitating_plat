from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import DroneLevitationEnv


def main():
    env = make_vec_env(DroneLevitationEnv, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    model.learn(total_timesteps=1_000_000)
    model.save("ppo_drone_levitation")


if __name__ == "__main__":
    main()
