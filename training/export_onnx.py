import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # for mac

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

# Resolve everything relative to this script so export does not depend on the
# current working directory (matches train.py).
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

# Observation / action dimensions, kept in sync with env.DroneLevitationEnv.
OBS_DIM = 6   # [height, vz, roll, pitch, roll_rate, pitch_rate]
ACTION_DIM = 4  # [m1, m2, m3, m4]


def load_default_model_path():
    """Default model to export: the best checkpoint of the configured stage.

    Reads training.stage from config and returns models/<stage>/best, matching
    the layout train.py writes (best.zip = highest-eval-reward checkpoint).
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    stage = config.get("training", {}).get("stage", "stage1")
    return f"models/{stage}/best"


class DeterministicActor(torch.nn.Module):
    """Wraps an SB3 policy to expose just obs -> deterministic action.

    SB3's MlpPolicy.forward() returns (actions, values, log_probs) and samples
    stochastically by default. For deployment on the Pi we want a clean
    single-output graph that returns the *mean* (deterministic) action, which is
    what you'd run at inference time.

    Note: policy._predict is an SB3-internal API (leading underscore). It's been
    stable on PPO's MlpPolicy, but it's the first thing to re-check on an SB3
    upgrade.

    The action space is Box(0, 1) but PPO's default DiagGaussianDistribution is
    unbounded — at train time env.step() clips to [0, 1]. The ONNX graph carries
    no such clip, so we bake a torch.clamp into the exported graph: the Pi gets
    in-range actions straight from the model (motor_agent can still clip again as
    defense-in-depth).
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        action = self.policy._predict(obs, deterministic=True)
        return torch.clamp(action, 0.0, 1.0)


def export(model_path=None, onnx_path=None):
    if model_path is None:
        model_path = SCRIPT_DIR / load_default_model_path()
    if onnx_path is None:
        onnx_path = SCRIPT_DIR / "policy.onnx"

    model = PPO.load(str(model_path), device="cpu")
    actor = DeterministicActor(model.policy)
    actor.eval()

    # Shape-check uses zeros; numerical check uses a realistic-magnitude sample
    # (zeros can sit at a degenerate point that hides issues). Roughly:
    # height~0.5m, vz~±0.3, roll/pitch~±0.3rad, rates~±0.5.
    dummy_input = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    sample_obs = torch.tensor(
        [[0.5, 0.3, 0.2, -0.2, 0.4, -0.3]], dtype=torch.float32
    )

    # ---- Sanity-check dims before exporting (6 obs -> 4 actions) ----
    with torch.no_grad():
        out = actor(dummy_input)
    assert out.shape == (1, ACTION_DIM), (
        f"expected action shape (1, {ACTION_DIM}), got {tuple(out.shape)}"
    )

    # dynamo=False forces the legacy TorchScript exporter. torch>=2.x defaults
    # to the dynamo exporter, which fails to trace SB3's policy internals
    # (_predict -> action distribution) under torch.export. The TorchScript path
    # traces this small MLP reliably and supports dynamic_axes as written below.
    torch.onnx.export(
        actor,
        dummy_input,
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )
    print(f"exported ONNX policy to {onnx_path}")
    print(f"  input  'obs'    : (batch, {OBS_DIM})")
    print(f"  output 'action' : (batch, {ACTION_DIM})")

    # ---- Verify the exported graph loads and matches torch output ----
    # Use a realistic-magnitude observation, not zeros, so the numerical check
    # and the printed action range are meaningful.
    with torch.no_grad():
        sample_out = actor(sample_obs)
    _verify(onnx_path, sample_obs, sample_out)


def _verify(onnx_path, sample_obs, torch_out):
    """Best-effort numerical check that the ONNX graph matches PyTorch."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  (skip verify: install onnx + onnxruntime to numerically check)")
        return

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["action"], {"obs": sample_obs.numpy()})[0]

    assert onnx_out.shape == (1, ACTION_DIM), (
        f"ONNX action shape {onnx_out.shape} != (1, {ACTION_DIM})"
    )
    np.testing.assert_allclose(onnx_out, torch_out.numpy(), rtol=1e-3, atol=1e-5)
    print("  verified: ONNX output matches PyTorch (6 obs -> 4 actions) ✓")

    # Confirm the baked-in clamp actually keeps actions in [0, 1] on a real obs.
    lo, hi = float(onnx_out.min()), float(onnx_out.max())
    print(f"  action range on sample obs: [{lo:.3f}, {hi:.3f}]")
    assert 0.0 <= lo and hi <= 1.0, "action escaped [0, 1] despite clamp"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained PPO policy to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="path to the trained model .zip (defaults to models/<training.stage>/best)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output .onnx path (defaults to policy.onnx next to this script)",
    )
    args = parser.parse_args()
    export(model_path=args.model, onnx_path=args.out)
