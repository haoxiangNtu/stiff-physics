# Multi-Environment Simulation Guide (v0.6.7)

Status: released in **v0.6.7** (the "unification" release of the conservative
`v0.6.x` line). This guide covers the engine features that make N tiled
environments (parallel RL) behave exactly like N independent single
environments — same contact thickness, same convergence quality, and no
cross-environment interaction — inside ONE StiffGIPC scene.

Everything here is validated by self-asserting gates shipped in this repo
(`examples/test_env_isolation.py`) and by RL training runs (IsaacLab cartpole /
ant / Franka soft-lift on the Newton solver backend).

---

## The problem

StiffGIPC solves all environments in a single IPC scene. Three things
historically leaked between environments:

| Leak | Symptom |
|---|---|
| `dHat` (contact activation distance) scaled with the **whole-scene** bbox | contact got thicker as you added envs; behavior differed between `num_envs=2` and `num_envs=32` |
| The Newton convergence threshold also scaled with the whole-scene bbox | more envs → looser tolerance → sloppier solves ("2 envs walk, 32 envs stand") |
| Contact was only separated by **spatial spacing** | envs placed close (or objects flung during RL exploration) could collide across env boundaries |

v0.6.7 closes all three.

## 1. Fixed contact thickness + env-independent convergence: `absolute_dhat`

```python
from stiff_physics.engine import Config, Engine

cfg = Config(
    dt=1 / 120,
    assets_dir=ASSETS_DIR,
    relative_dhat=1e-3,     # keep the default
    absolute_dhat=3e-3,     # NEW: dHat is a FIXED 3 mm, at any env count
)
```

- `absolute_dhat > 0` overrides the `relative_dhat * scene_bbox` scaling: the
  IPC barrier activates at a fixed metric distance, independent of how many
  environments are tiled.
- The **Newton convergence threshold reuses the same fixed scale** (an
  effective bbox `= (absolute_dhat / relative_dhat)^2`), so the solve
  converges to the same quality at any `num_envs`. The implied velocity
  tolerance is `newton_tol * absolute_dhat / relative_dhat` (e.g.
  `0.01 * 3e-3 / 1e-3 = 0.03 m/s`, on par with libuipc's default
  `newton/velocity_tol = 0.05 m/s`).
- `absolute_dhat = 0` (default) keeps the legacy scene-bbox behavior for
  backward compatibility.

**Recommendation: always set `absolute_dhat` for multi-env RL.** Pick it like
a contact thickness: `1e-3`–`3e-3` m for tabletop manipulation scales.

## 2. Cross-env contact isolation: `Engine.set_vertex_env_ids`

Per-VERTEX environment tags. The broad-phase skips every contact pair whose
two vertices carry different (>= 0) env ids — environments never interact,
**regardless of spatial proximity**, with zero spacing required.

```python
import numpy as np

eng.finalize()

n = len(eng.get_vertices())
env_ids = np.zeros(n, dtype=np.int32)
# tag each vertex with its environment (example: 2 FEM bodies = 2 envs)
s0, c0 = eng.get_fem_body_vertex_range(0)
s1, c1 = eng.get_fem_body_vertex_range(1)
env_ids[s0:s0 + c0] = 0
env_ids[s1:s1 + c1] = 1
eng.set_vertex_env_ids(env_ids)
```

Semantics:

- Length = engine vertex count (ABD-body vertices in load order, then FEM
  particles).
- `env_id >= 0`: vertex belongs to that environment; pairs with a different
  non-negative id are skipped in every broad-phase path (PT, EE, and both CCD
  variants).
- `env_id < 0`: **shared** geometry that collides with every environment.
- The analytical ground plane is not filtered — every env keeps ground
  contact.
- May be called any time after `finalize()`; the device array is re-uploaded
  on each call.
- It is a **contact filter only** — decoupled from the linear solve, so it
  composes with everything else (joint constraints, friction, per-body
  exclusions).

**Why per-vertex (not per-body)?** All FEM particles live in one combined
collision body, so a per-body exclusion (`set_body_groups` /
`add_collision_exclusion`) cannot distinguish a FEM cube in env 0 from one in
env 3. Per-vertex tags isolate FEM and ABD uniformly. Keep using
`add_collision_exclusion` for *within-env* pair exclusions (e.g. robot
self-collision); use `set_vertex_env_ids` for the cross-env cut.

Gate: `examples/test_env_isolation.py` — two overlapping-drop FEM cubes stack
with strict non-penetration by default, and pass through each other once
tagged into different envs.

## 3. Engine-native joint limits

Revolute and prismatic joints now enforce position limits inside the engine:
a **one-sided, mass-scaled penalty spring** toward the violated bound
(`K_lim = joint_limit_strength_ratio * (m_parent + m_child)`, default ratio
20000). It is an implicit energy term (gradient + SPD Gauss-Newton Hessian) —
stable when stiff, and **not** a log-barrier, so it costs nothing while the
joint is inside its range. Set the ratio to 0 to disable.

Under RL exploration this is what keeps random-action joint targets from
folding an articulation through itself.

## 4. Per-env resets

`teleport_fem_vertices(positions, ...)` hard-teleports the FEM block
(position + rest/predictor state, zeroed velocity) and correctly offsets by
the FEM block start in mixed ABD+FEM scenes (fixed in v0.6.7). Combined with
per-env vertex ranges this gives episode resets that touch ONLY the resetting
environment — never teleport non-resetting envs, or their contact predictor
state is corrupted.

## 5. Grounded contact correctness (v0.6.7 fix)

The ground-barrier d=0 guard is now a conditional NaN guard
(`dist2 == 0 ? 1e-12 : dist2`) instead of an `fmax` clamp. The clamp silently
**capped the barrier force** for vertices resting exactly on the ground,
collapsing the line-search step and freezing bodies in place (or pegging the
Newton iteration cap). With the fix, the barrier force correctly grows
unbounded as d → 0 and strict non-penetration is preserved.

---

## Putting it together (multi-env RL recipe)

```python
cfg = Config(dt=1/120, assets_dir=..., absolute_dhat=3e-3)
eng = Engine(cfg)
# ... load N replicated envs (robots, objects) ...
eng.finalize()

env_ids = build_per_vertex_env_ids(eng, num_envs)   # section 2
eng.set_vertex_env_ids(env_ids)

# training loop: step; on episode reset, teleport ONLY that env's
# FEM slice / ABD bodies (section 4)
```

Validated end-to-end (v0.6.7 wheels, IsaacLab-3.0 Newton backend):
cartpole converges to the full episode length; the ant walking policy replays
at the training env count (30/32 envs > 0.5 m displacement); the Franka
soft-cube lift trains with per-vertex isolation active and zero Newton-cap
hits.
