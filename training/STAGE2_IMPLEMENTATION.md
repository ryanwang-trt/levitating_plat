# Stage 2 — In-Flight Disturbance Recovery: Implementation Notes

This document records the full implementation of **Stage 2** of the levitating-platform
RL training pipeline: teaching the policy to survive sustained external shoves while
staying level, using only observations available on the real hardware. It includes the
design, every diagnostic run with its numbers, the bugs found and fixed, and the
engineering trade-offs. It is meant as raw material for a project report.

---

## 1. System Context (what Stage 1 already gave us)

**Task ("Free Float" / Method B):** a 1 kg, 30 cm × 30 cm × 4 cm platform with 4 corner
thrusters must stay **alive, level, and still**. It is **not** rewarded for holding any
particular horizontal position — it may be pushed anywhere as long as it keeps a level
attitude and doesn't crash or tip over.

**Observation (6-dim, IMU-only):** `[height, vz, roll, pitch, roll_rate, pitch_rate]`.
Crucially, **horizontal velocity (vx, vy) and horizontal position are NOT observed** —
this is deliberate, because the real platform (Raspberry Pi 5 + IMU) cannot cheaply
measure them. This constraint drives most of the Stage 2 difficulty.

**Action (4-dim):** `[m1, m2, m3, m4]` in `[0, 1]`, each scaled to a corner thrust.
Max thrust per motor = `(1.0 × 9.81 / 4) × 2.0 ≈ 4.9 N` (≈2× hover).

**Reward (per step), from `reward.py`:**
```
reward = alive_bonus (+1.0)
       - w_tilt(5.0)     × (|roll| + |pitch|)
       - w_vz(2.0)       × |vz|
       - w_ang_rate(1.0) × (|roll_rate| + |pitch_rate|)
       - w_energy(0.1)   × Σ(action²)
```

**Termination:** `height < 0.05` or `> 2.5` m, or `|roll| > 80°` or `|pitch| > 80°`.
**Truncation:** at 1000 steps (the episode length cap). PyBullet runs at **dt = 1/240 s**,
so 1000 steps ≈ 4.17 s.

**Stage 1 result (baseline):** trained from scratch with domain randomization on the
initial conditions (±15° tilt, ±1 m/s vz, ±1 rad/s rates) but **no in-flight disturbance
injection**. It converged cleanly:

| Stage 1 eval (timesteps) | mean ep_len | mean reward |
|---|---|---|
| 25 k | 90.5 | −290 |
| 350 k | 1000.0 | −268 |
| 375 k | 1000.0 | −17 |
| 400 k | 1000.0 | +153 |
| 525 k–1 M | 1000.0 | +500 to +590 |

Stage 1's `best.zip` survives 10/10 episodes of pure free-float (no shoves) at full 1000
steps. It is the clean starting point for the Stage 2 curriculum.

---

## 2. Stage 2 Goal & The Four Mechanisms Added

Stage 2 fine-tunes Stage 1 to also withstand **in-flight shoves**. Four mechanisms were
added, each motivated by a concrete failure observed during development:

1. **Sustained-shove disturbance injection** — shoves are *held* for many frames, not a
   single tap.
2. **Post-shove reward grace** — don't punish the policy for the tilt the shove forced.
3. **Air drag (damping)** — let residual horizontal momentum decay (it's unobservable).
4. **A 2-stage force/duration/damping curriculum** — ramp difficulty, fine-tune the chain.

Plus a fifth, infrastructural fix:

5. **Deterministic eval best-selection** — the eval RNG bug that corrupted which
   checkpoint was saved as "best".

---

## 3. Mechanism 1 — Sustained-Shove Injection

### Problem
The original injection applied a **single-frame** random force (2–8 N) with small per-step
probability. But the demo/eval in `eval.py` applies a force *held* for 30 frames. The gap,
measured as **linear impulse** (J = force × dt × duration):

| | force | duration | impulse J |
|---|---|---|---|
| Original training (single frame) | 2–8 N | 1 frame (4.17 ms) | 8.3–33.3 mN·s |
| Eval demo (sustained) | ~18–19 N | 30 frames (125 ms) | ~2250–2400 mN·s |

So the eval's peak force was **2.4× larger**, but because it was *held 30× longer*, the
impulse was **~72× larger**. The eval push was massively out-of-distribution → the
platform always failed. Raising only `force_max` would never close a 72× gap; the
injection itself had to support **sustained, multi-frame** shoves.

### Implementation (`env.py`)
A shove now samples a **force, application offset, and a random duration** once, then
**holds them constant** for that many frames. State: `_shove_remaining`, `_shove_force`,
`_shove_offset`. `prob` was redefined to mean "probability of *starting* a new shove on a
step where none is in progress" (not re-rolled mid-shove), and lowered from 0.015 → 0.01
because each shove now occupies up to `duration_max` frames.

- Force: `uniform(force_min, force_max)` N, random direction on the unit sphere (all
  directions incl. horizontal).
- Application offset: `uniform(-0.15, 0.15)` m in x and y (within the platform's
  half-extents of 0.15 m), so the shove **both translates and torques** the platform.
- Duration: `integers(duration_min, duration_max+1)` frames.

### Verified
Tests lock that a shove is held for exactly `duration` frames with constant force, and
that `reset()` clears an in-progress shove so it doesn't leak across episodes.

---

## 4. Mechanism 2 — Post-Shove Reward Grace

### Problem (quantified)
Once sustained shoves were added, Stage 2 fine-tuning **degraded** the policy. The
`evaluations.npz` history showed reward **negative for the entire run** and ep_len falling
from Stage 1's 1000 down to ~700–880, with **no convergence over 1 M steps**.

Root cause, computed for a typical post-shove state (tilted 25° on both axes, tumbling at
1.5 rad/s):

```
alive bonus  : +1.00
tilt penalty : −4.36   (5.0 × (0.44 + 0.44))
vz penalty   : −1.60
ang_rate pen : −3.00   (1.0 × (1.5 + 1.5))
NET / step   : −7.96
```

The per-step penalty was **~8× the alive bonus**. Over a ~40-step recovery window a shove
cost ~**−319 reward**. The policy's optimal response became: **tip over early
(terminate) to stop the bleeding**, since `|tilt| > 80°` ends the episode and halts the
accumulating penalty. It learned to *give up* rather than recover. This explains the
negative reward and the depressed ep_len.

### Implementation (`reward.py`, `env.py`)
`compute_reward` gained an `attitude_grace` flag. When True (set by env during a shove and
for `grace_frames` after), the **tilt and ang_rate penalties are waived** — the platform
was knocked off-level *by the shove*, not by bad policy, so it shouldn't be charged for
*being hit*, only for *failing to recover* once grace ends. `alive_bonus`, `vz`, and
`energy` terms stay active so "die early" is no longer optimal and the policy can't thrash
the throttle for free.

`grace_frames: 60` (≈ 0.25 s). A shove **re-arms** grace to `grace_frames` on every active
frame, so grace covers the whole shove + 60 frames after it ends.

### Verified
Tests lock: grace waives tilt/ang_rate but keeps vz/energy; grace defaults off; grace
arms on a shove and decays to 0 over subsequent shove-free frames.

---

## 5. Mechanism 3 — Air Drag (the central discovery)

### Problem
Even with sustained shoves + grace + a gentle fine-tune learning rate, the GUI eval
showed a striking pattern: after a shove the policy **recovered attitude beautifully**
(roll/pitch converging smoothly toward 0), then **~150 frames later the platform
spontaneously tipped over** — long after it had leveled out.

Full-state tracing (printing vx/vy, which the policy can't see) revealed the mechanism:

```
step  40 (post-shove): vx=+0.438  roll=+0.072   <- attitude recovering
step 100:              vx=+0.417  roll=+0.038   <- still drifting at ~0.4 m/s
step 160:              vx=+0.334  roll=+0.042
step 180:              vx=+0.297  roll=+0.140   <- starts diverging
step 200:              vx=+0.198  roll=−0.175
step 222:              vx=+0.371  roll=−1.444   <- TIPPED OVER, terminated
```

The shove imparted ~0.44 m/s of **horizontal velocity that never decayed** — PyBullet
bodies are **undamped by default**. The platform drifted at near-constant speed, and
maintaining "translate + stay level" is a coupled control problem that eventually
destabilized. The policy **cannot fix it**, because vx/vy aren't in the observation — and
they aren't measurable on the real hardware either, so adding them to obs was rejected
(it would teach a sim-to-real crutch the hardware can't supply).

### Insight
On a **real** platform, horizontal momentum is bled off by **air drag**. The simulation
omitted this, making the sim *harder than reality* (undamped drift is a worse environment
than the real one). Adding linear/angular damping models real air drag, lets residual
momentum dissipate physically, and keeps the policy stable on **IMU-only** observations.

### Verification (no retraining — same policy, just add damping at eval)
Applying `changeDynamics(linearDamping=d)` to the existing policy and re-running the
post-shove eval:

| linearDamping | post-shove survival (full 1000 steps) |
|---|---|
| 0.0 (default) | **0/8** (mean 240 steps before tipping) |
| 0.5 | **8/8** |
| 1.0 | 8/8 |
| 2.0 | 8/8 |

A single physics term (damping = 0.5), **no retraining**, flipped post-shove survival from
0/8 to 8/8. Root cause and fix both confirmed.

### Implementation (`env.py`, `config.yaml`)
`reset()` calls `p.changeDynamics(drone_id, -1, linearDamping=…, angularDamping=…)`,
reading `physics.linear_damping` / `physics.angular_damping` from config. Test locks that
a sideways velocity kick decays more with damping than without.

---

## 6. Mechanism 5 — Deterministic Eval Best-Selection (an infrastructure bug)

### Problem
`EvalCallback` saves `best.zip` whenever an evaluation sets a new high mean reward. The
eval env was seeded **once** up front (`eval_env.reset(seed=seed)`), and subsequent
EvalCallback `reset()` calls passed no seed. The comment claimed this gave "a fixed
disturbance sequence across evaluations" — **this was false**. The RNG stream keeps
advancing, so **each evaluation drew a different disturbance sequence**.

### Verified
Same checkpoint, two consecutive evaluations under the old setup:

```
eval #1 mean reward = 232.0
eval #2 mean reward = 199.2   -> 32.8 apart, pure noise
```

A swing of ~33 reward between identical evaluations of the **same** policy. This noise
corrupted best-selection: a mediocre checkpoint that happened to draw an easy sequence
could outscore a genuinely better one and be saved as "best", then propagate down the
fine-tune chain. (It also explains part of the large eval-reward variance seen earlier.)

### Implementation (`train.py`)
A `FixedSeedReset` gym wrapper forces **every** eval reset to reuse the same seed. After
the fix, three consecutive evaluations of the same checkpoint returned:

```
691.21 / 691.21 / 691.21   (spread 0.0000)
```

Fully reproducible. The wrapper is applied **only** to the eval env — the training env
still gets fresh sequences every episode (we *want* training variety).

(Also added in `train.py`: `--lr` override for gentle fine-tuning. Important subtlety —
`PPO.load` restores the **saved** learning rate (3e-4 from Stage 1) and ignores the
constructor's; fine-tuning a converged policy at 3e-4 tends to wreck it. The fix
explicitly overrides `model.learning_rate` **and** `model.lr_schedule` after load. Stage 2
fine-tunes at **3e-5**.)

---

## 7. Mechanism 4 — The 2-Stage Curriculum (and the grace red herring)

### The grace duty-cycle investigation (a hypothesis that was disproven)
When the curriculum's Stage 2 (6–12 N, duration 1–20) was first trained, ep_len collapsed
to ~860–880 again. A plausible hypothesis: grace **re-arms to 60 every shove frame**, so
with longer/harder shoves the grace window stays open most of the time, effectively
turning attitude penalties off and resurrecting the "tip over early" anti-incentive.

Measured grace duty cycle vs. curriculum stage:

| stage | duration_max | grace-active fraction | ep_len |
|---|---|---|---|
| 1 | 10 | 45.1% | 1000 |
| 2 | 20 | 49.7% | 862 |
| 3 | 30 | 51.7% | 651 |

The correlation looked damning (ep_len collapses as grace crosses ~50%). Shrinking
`grace_frames` 60 → 15 dropped the duty cycle (to ~29% even at the hardest stage). But
**retraining with grace_frames=15 made things worse**, not better (reward −819 → −1699,
ep_len still ~880).

### The disproof (precise termination diagnostic)
Running 50 episodes of the grace=15 Stage-2 policy and recording **why** each episode
terminated:

```
ep_len: mean=905  min=282  max=1000
termination causes (50 eps): {tip-over: 13, height-low: 1, survived: 36}
of the 13 tip-overs: 0 / 13 occurred DURING the grace window
```

**All 13 tip-overs happened in free float, after grace had ended.** If grace duty cycle
were the cause, tip-overs would cluster *inside* the grace window — instead, zero did.
The reward drop (−819 → −1699) was a **measurement artifact**: with grace=15, more
post-shove tilt frames get penalized, so reward looks worse without the policy actually
being worse (ep_len was unchanged/slightly better). **Lesson: judge by ep_len and
tip-over rate, not raw reward — grace changes the reward scale and makes runs
incomparable.** `grace_frames` was reverted to 60.

### The real cause again — horizontal momentum, under-damped at higher force
Tracing the 13 tip-overs confirmed high residual horizontal speed before each:

```
tip ep0: peak_horiz_speed=3.74 m/s,  speed 30 steps before end=1.69 m/s
tip ep2: peak=1.35,  pre-end=1.19
tip ep5: peak=2.16,  pre-end=2.01
```

Damping 0.5 sufficed for Stage 1 (4–8 N) but not for the larger momentum injected by
Stage 2's 6–12 N shoves. A damping sweep against the actual policy:

| force range | damp 0.5 | damp 1.0 | damp 1.5 | damp 2.0 |
|---|---|---|---|---|
| **Stage 2** (6–12 N) | 13/30 tip | **0/30 tip** | 0/30 | 0/30 |
| **Stage 3** (8–15 N) | 21/30 tip | 10/30 tip | 3/30 tip | **0/30 tip** |

### Engineering decision — cap damping at 1.0, drop Stage 3
- **Stage 2 (6–12 N)** is fully handled by **damping = 1.0** (0/30 tips).
- **Stage 3 (8–15 N)** needs **damping = 2.0** to be tip-free — but 2.0 is far stiffer
  than the real air drag on a 30 cm platform at sub-m/s speeds. Training against it would
  teach a **sim-to-real crutch** (a strong drag the hardware doesn't provide).
- **Decision:** cap damping at **1.0** (judged still physically plausible) and **stop the
  curriculum at Stage 2**. Stage 3 was deleted.

### Final curriculum (`config.yaml` `disturbance_stages`, `train.py --dist-stage`)
Damping ramps **with** the disturbance difficulty:

| stage name | dist-stage | force | duration | linear_damping | init-from |
|---|---|---|---|---|---|
| stage2a | 1 | 4–8 N | 1–10 frames | 0.5 | stage1/best |
| stage2b | 2 | 6–12 N | 1–20 frames | 1.0 | stage2a/best |

Each stage fine-tunes from the previous stage's **best** at lr **3e-5**. `--dist-stage`
overrides force/duration **and** damping from the chosen `disturbance_stages` entry;
`prob` and `grace_frames` are shared. The global `physics.linear_damping` default is set
to **1.0** so that **eval** (which doesn't pass `--dist-stage`) matches the hardest trained
stage and the demo is honest.

---

## 8. Key Parameters (final values)

**Reward:** `alive_bonus=1.0, w_tilt=5.0, w_vz=2.0, w_ang_rate=1.0, w_energy=0.1`,
`grace_frames=60`.

**Disturbance (base):** `prob=0.01`, all-directions, off-center offset ±0.15 m.

**Curriculum stages:** stage1→2a (4–8 N, 1–10 fr, damp 0.5) → 2b (6–12 N, 1–20 fr, damp 1.0).

**Physics default:** `linear_damping=1.0, angular_damping=1.0`.

**Domain randomization (init conditions):** ±15° tilt, ±1.0 m/s vz, ±1.0 rad/s rates.

**Training:** PPO (SB3), MlpPolicy, lr 3e-4 from scratch / **3e-5 fine-tune**, n_steps 2048,
batch 64, seed 0, device CPU. EvalCallback: eval_freq 25 k, n_eval_episodes 10,
deterministic, fixed-seed eval env. Episode cap 1000 steps. PyBullet dt = 1/240 s.

**Hardware target:** Raspberry Pi 5, motor BCM GPIO `[12, 13, 18, 19]` (hardware PWM),
IMU on I²C bus 1.

---

## 9. Pipeline / Tooling

- **`train.py`** — `--stage <name>` (output dir `models/<name>/`), `--dist-stage {1,2}`,
  `--init-from <zip>` (fine-tune), `--lr <rate>`, `--timesteps <n>`. Writes
  `models/<stage>/{final.zip, best.zip, evaluations.npz}`. `FixedSeedReset` on the eval env.
- **`eval.py`** — GUI eval; takes a stage name or model path
  (`python eval.py stage2b`, default `stage2b`). Applies deterministic held shoves at
  fixed steps and prints the roll/pitch recovery trajectory; inherits config damping.
- **`run_curriculum.ps1`** — runs the 2-stage chain end-to-end (stage2a→stage2b), each at
  400 k steps / lr 3e-5, aborting if a stage fails so a broken checkpoint never feeds the
  next. Params: `-Steps`, `-Python`, `-Lr`.
- **`export_onnx.py`** — exports a trained policy to ONNX for Pi deployment (deterministic
  actor with a baked-in `[0,1]` clamp on the action).
- **Tests** (`tests/`, 40 passing) — env step well-formedness, reward contract incl. grace
  waiver, sustained-shove hold/duration, reset clears shove+grace, grace arm/decay, air-drag
  decay, disturbance off-by-default determinism, ONNX round-trip, full-episode integrity.

**Artifact layout:** `models/<stage>/{final,best}.zip` + `evaluations.npz`. Model zips and
`tb_logs/` are **gitignored** — only code + config are committed; models are reproduced by
re-running the curriculum.

---

## 10. Status & Open Items

- **Done & verified:** sustained shoves, reward grace, air drag (root-cause-confirmed),
  fixed-seed eval, 2-stage curriculum, per-stage damping, low-lr fine-tune, full test suite.
- **stage2a** (dist-stage 1, damp 0.5): trained, valid as the Stage-2 curriculum start.
- **stage2b** (dist-stage 2, damp 1.0, grace 60): **needs a clean retrain** with the final
  config (the earlier stage2b was the damp-0.5 / grace-15 bad run and was deleted).
  Command: `python train.py --stage stage2b --dist-stage 2 --init-from models/stage2a/best.zip --lr 3e-5 --timesteps 400000`.
  **Judge by ep_len** (target ~1000) and tip-over rate, **not raw reward**.
- **Known limitation (by design):** horizontal velocity is unobservable on hardware, so
  recovery from large horizontal momentum relies on air drag dissipating it. This caps the
  shove magnitude the policy can robustly handle (hence the Stage-3 cutoff). A future option
  is adding an optical-flow sensor (e.g. PMW3901) fused with the IMU to make vx/vy
  observable — which would lift that cap but adds hardware and sim-to-real calibration.

---

## 11. Methodological Notes (process lessons)

- **Quantify before tuning.** Every change was preceded by a measurement (impulse gap,
  reward breakdown, grace duty cycle, termination-cause histogram, damping sweep) rather
  than guessing parameters.
- **Reward magnitude is not a quality metric across config changes.** Grace and damping
  both shift the reward scale; ep_len and tip-over rate were the reliable signals.
- **A correlation (grace duty cycle ↔ ep_len) was disproven by a targeted diagnostic**
  (0/13 tip-overs in-grace). Worth highlighting in the report as an example of avoiding a
  plausible-but-wrong fix.
- **The sim-to-real constraint (IMU-only obs) shaped the solution**: the fix was to make
  the *physics* more realistic (air drag), not to feed the policy information the hardware
  lacks.
