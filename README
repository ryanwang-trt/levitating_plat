# levitating_plat

AI-controlled self-levitating drone platform. Hovers at a fixed height, detects and recovers from physical disturbances. Built with PPO reinforcement learning and a multi-agent C++ control loop on a Jetson Orin Nano.

Long-term open-source project. Pretrained model included — clone and fly.

---

## What it does

- Hovers at a target height (0.5m) using 4 brushless motors
- Runs a 100Hz C++ control loop onboard
- RL policy (PPO, exported as ONNX) makes real-time motor decisions
- Detects external pushes and autonomously recovers

---

## Stack

| Layer | Tech |
|---|---|
| RL training | Python, Stable-Baselines3, PyBullet |
| Onboard control | C++, ONNX Runtime |
| Compute | Jetson Orin Nano |
| Sensors | VL53L1X (height), MPU-6050 (IMU) |
| Motors | 4× 2212 brushless via ESC |

---

## Repo structure

```
training/       # RL training pipeline (Python) — runs on PC/cloud
src/            # Control loop (C++) — runs on Jetson
models/         # Pretrained ONNX policy
config.yaml     # Tunable params
```

---

## Agents (C++ threads)

```
Sensors → State Estimator → Disturbance Detector → Control Agent → Motor Agent → Motors
```

Each agent runs as an independent thread, communicating via message queues.

---

## Getting started

### Train from scratch
```bash
pip install -r requirements.txt
python training/train.py
python training/export_onnx.py
```

### Run on Jetson (pretrained)
```bash
cmake -B build && cmake --build build
./build/levitating_plat
```

---

## Roadmap

| Version | Focus |
|---|---|
| V1 | Stable hover + disturbance recovery |
| V2 | Hand controller, web dashboard |
| V3 | Camera-based localization |
| V4 | Online learning, LLM command layer |

---

## Safety

- Always tether-test before free flight
- Hardware kill switch required
- Auto-shutoff if height < 0.05m or > 2.5m
