"""Integration test for export_onnx.

Trains a tiny PPO for a handful of steps, exports it to ONNX, and round-trips a
sample observation through onnxruntime. This locks in the export contract:
  - graph is obs(6) -> action(4)
  - the baked-in clamp keeps actions in [0, 1]
  - ONNX output matches PyTorch

It's a slow-ish test (it actually trains), and needs onnx/onnxruntime, so it
skips cleanly if those aren't installed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from stable_baselines3 import PPO  # noqa: E402

from env import DroneLevitationEnv  # noqa: E402
from export_onnx import DeterministicActor, OBS_DIM, ACTION_DIM, export  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    env = DroneLevitationEnv()
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu", seed=0)
    model.learn(total_timesteps=64)
    env.close()
    return model


def test_deterministic_actor_output_shape_and_range(tiny_model):
    actor = DeterministicActor(tiny_model.policy)
    actor.eval()
    obs = torch.tensor([[0.5, 0.3, 0.2, -0.2, 0.4, -0.3]], dtype=torch.float32)
    with torch.no_grad():
        out = actor(obs)
    assert out.shape == (1, ACTION_DIM)
    # The clamp must hold: every action component in [0, 1].
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_deterministic_actor_is_deterministic(tiny_model):
    actor = DeterministicActor(tiny_model.policy)
    actor.eval()
    obs = torch.tensor([[0.5, 0.0, 0.1, -0.1, 0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        a = actor(obs)
        b = actor(obs)
    np.testing.assert_allclose(a.numpy(), b.numpy())


def test_export_round_trips_through_onnx(tiny_model, tmp_path):
    model_path = tmp_path / "tiny_ppo"
    onnx_path = tmp_path / "policy.onnx"
    tiny_model.save(str(model_path))

    # export() loads the .zip, builds the actor, writes + verifies the ONNX file.
    export(model_path=str(model_path), onnx_path=str(onnx_path))
    assert onnx_path.exists()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    assert inp.name == "obs"
    assert out.name == "action"
    # Static dims: obs has 6 features, action has 4 (batch axis is dynamic).
    assert inp.shape[-1] == OBS_DIM
    assert out.shape[-1] == ACTION_DIM

    sample = np.array([[0.5, 0.3, 0.2, -0.2, 0.4, -0.3]], dtype=np.float32)
    action = sess.run(["action"], {"obs": sample})[0]
    assert action.shape == (1, ACTION_DIM)
    assert action.min() >= 0.0 and action.max() <= 1.0

def test_clamp_actually_bounds_out_of_range_actions():
    """Prove the baked-in clamp does its job — not just that actions happen to
    land in [0, 1] on a normal obs.

    A real trained policy on a normal observation usually outputs in-range
    already, so `min>=0, max<=1` can pass even if the clamp were deleted. Here we
    feed DeterministicActor a stub policy whose _predict deliberately returns
    out-of-range values; only the clamp can bring them back into [0, 1].
    """

    class OutOfRangePolicy:
        # Mimics the one method DeterministicActor calls on policy.
        def _predict(self, obs, deterministic=True):
            # Deliberately outside [0, 1] on both ends.
            return torch.tensor([[-3.0, 5.0, 0.5, 100.0]], dtype=torch.float32)

    actor = DeterministicActor(OutOfRangePolicy())
    obs = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        out = actor(obs)

    # Without the clamp this would be [-3, 5, 0.5, 100]; with it, all in [0, 1].
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    # And the clamp must hit the exact rails, not just "somewhere in range".
    expected = torch.tensor([[0.0, 1.0, 0.5, 1.0]], dtype=torch.float32)
    np.testing.assert_allclose(out.numpy(), expected.numpy())