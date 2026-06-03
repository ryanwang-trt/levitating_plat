import torch
from stable_baselines3 import PPO


def export(model_path="ppo_drone_levitation", onnx_path="policy.onnx"):
    model = PPO.load(model_path)
    policy = model.policy
    policy.eval()

    dummy_input = torch.zeros(1, 6)
    torch.onnx.export(
        policy,
        dummy_input,
        onnx_path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
    )


if __name__ == "__main__":
    export()
