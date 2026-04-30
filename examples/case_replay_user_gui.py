#!/usr/bin/env python3
"""Interactive GUI replay of a recorded xarm7 + cloth trajectory (case_26).

Loads a 477-frame qpos trajectory bundled at
`trajectories/case26_user_replay.h5` (8 columns: 7 arm-joint targets in
radians + 1 gripper command), drives the arm through it while
Polyscope renders the scene live. Defaults are the v0.3.0 audited
"perf_extreme" configuration (joint_strength_ratio=100, gripper-mul=0.1,
dt=0.020) so you should see ~50 fps median on an RTX 4090.

Features:
  - Polyscope 3D viz with arm + cloth
  - Live HUD: current frame, instantaneous ms/step, running-30 fps
  - Run / Pause / Restart buttons
  - Auto-loops trajectory so you can watch indefinitely

Common variations:
    PYTHONPATH=. python examples/case_replay_user_gui.py
        [--no-cloth]              arm-only (no FEM cloth)
        [--scale 0.3]             arm size factor (default matches recording)
        [--strength 100]          joint_strength_ratio (default v0.3.0 tuned)
        [--gripper-mul 0.1]       per-joint multiplier on gripper finger/knuckle joints
        [--cloth-y 0.3]           cloth y-position (raise to keep gap when scaling arm)
        [--qpos PATH]             override bundled trajectory
        [--label-orient]          apply per-face orient labels (libuipc-style winding fix)
"""
import argparse, math, sys, time
from collections import deque
from pathlib import Path
import numpy as np
import h5py

import polyscope as ps
import polyscope.imgui as psim

from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"
# Bundled sample trajectory in the public repo's trajectories/ directory.
# 477 frames at dt=0.020s recorded for the case_26 arm + cloth scene
# (8 columns: 7 arm-joint targets in radians + 1 gripper command).
DEFAULT_QPOS = str(Path(__file__).resolve().parent.parent
                   / "trajectories" / "case26_user_replay.h5")


def make_arm_tf(scale: float):
    from scipy.spatial.transform import Rotation
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[1, 3] = -0.95
    return tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", default=DEFAULT_QPOS)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--no-cloth", action="store_true")
    ap.add_argument("--cloth-y", type=float, default=0.3,
                    help="cloth y-position (default 0.3 = case_26 default; "
                         "raise this for bigger arm scales to keep gap. "
                         "Avoid extreme values >2-3m: scene bbox grows, "
                         "dHat = relative_dhat^2 * bbox^2 grows, broad-phase "
                         "pair count exceeds the fixed-size triplet buffer "
                         "allocated at finalize() and the kernel writes OOB.")
    ap.add_argument("--cloth-x", type=float, default=0.25,
                    help="cloth x-position (default 0.25 = case_26 default; "
                         "small shifts <=1m are OK, larger values blow up "
                         "scene bbox => dHat => pair-count buffer overflow")
    ap.add_argument("--cloth-z", type=float, default=0.0,
                    help="cloth z-position (default 0 = case_26 default; "
                         "horizontal sideways shift from arm centerline)")
    ap.add_argument("--strength", type=float, default=100.0,
                    help="joint_strength_ratio (100=v0.3.0 perf_tuned default, 50=more aggressive)")
    ap.add_argument("--gripper-mul", type=float, default=0.1,
                    help="per-joint multiplier for gripper finger/knuckle joints "
                         "(0.1=v0.3.0 perf_extreme; requires set_gripper_strength API in v0.3.0+)")
    ap.add_argument("--beta-tol", type=float, default=5e-2,
                    help="semi_implicit_beta_tol (5e-2=perf_tuned, 1e-3=strict accuracy)")
    ap.add_argument("--newton-tol", type=float, default=5e-2,
                    help="newton_tol (5e-2=perf_tuned, 1e-2=strict accuracy)")
    ap.add_argument("--dt", type=float, default=0.020,
                    help="timestep (0.020=perf_tuned, 0.010=strict accuracy)")
    ap.add_argument("--no-loop", action="store_true",
                    help="stop at end instead of looping")
    ap.add_argument("--label-orient", action="store_true",
                    help="run flood-fill orient labeling on every ABD body before "
                         "finalize. Fixes per-face winding inconsistency in URDF "
                         "collision meshes (libuipc-style, no topology change).")
    args = ap.parse_args()
    use_cloth = not args.no_cloth

    with h5py.File(args.qpos, "r") as f:
        qpos_all = f["qpos"][:]
    print(f"Loaded {len(qpos_all)} qpos frames")

    cfg = Config(
        dt=args.dt,
        cloth_thickness=1e-3, cloth_young_modulus=1e4, bend_young_modulus=1e3,
        cloth_density=200, strain_rate=100,
        soft_motion_rate=1.0, poisson_rate=0.49, friction_rate=0.4,
        relative_dhat=1e-3,
        joint_strength_ratio=args.strength,
        revolute_driving_strength_ratio=args.strength,
        semi_implicit_enabled=True, semi_implicit_beta_tol=args.beta_tol,
        semi_implicit_min_iter=1, newton_tol=args.newton_tol, pcg_tol=1e-4,
        assets_dir=ASSETS_DIR,
    )

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1280, 720)
    ps.set_autocenter_structures(False)
    ps.set_autoscale_structures(False)

    engine = Engine(cfg)
    assets = engine.native.get_assets_dir()
    engine.native.load_urdf(assets + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
                            make_arm_tf(args.scale), True, False, 1e7)
    for bid in range(engine.abd_body_count):
        engine.add_ground_collision_skip(bid)

    if use_cloth:
        shirt_tf = np.eye(4); shirt_tf[:3, :3] *= 0.5
        shirt_tf[0, 3] = args.cloth_x
        shirt_tf[1, 3] = args.cloth_y
        shirt_tf[2, 3] = args.cloth_z
        engine.load_mesh("triMesh/shirt_6436v.obj", dimensions=2, body_type="FEM",
                         transform=shirt_tf, young_modulus=1e2)

    # libuipc-style per-face orient labeling — must run BEFORE finalize().
    # Fixes inconsistent winding in URDF collision .obj/.STL without
    # mutating face vertex order (collision/BVH/render unaffected).
    if args.label_orient:
        n_total = 0
        for body_id in range(engine.abd_body_count):
            n_total += engine.label_face_orient_for_abd_body(body_id)
        print(f"[orient] flood-fill labeled {n_total} faces across "
              f"{engine.abd_body_count} ABD bodies")

    engine.finalize()
    robot = Robot(engine)

    n_grip = 0
    if args.gripper_mul != 1.0:
        if hasattr(robot, "set_gripper_strength"):
            n_grip = robot.set_gripper_strength(args.gripper_mul)
        else:
            print(f"WARNING: --gripper-mul {args.gripper_mul} requested but "
                  "set_gripper_strength API not available on this binary; ignored.")

    print(f"# scale={args.scale}  cloth={use_cloth}  strength={args.strength}  "
          f"gripper_mul={args.gripper_mul}({n_grip} joints)  "
          f"arm_bodies={engine.abd_body_count}  surface_verts={len(engine.get_vertices())}")

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    # Decent default camera; adjust to taste in the viewer.
    ps.reset_camera_to_home_view()
    ps.look_at(camera_location=(0.45, -0.05, 0.45), target=(0.05, -0.65, 0.0))

    state = {
        "running": False,
        "frame": 0,
        "ms_window": deque(maxlen=30),
        "looped": 0,
    }

    def cb():
        psim.SetNextWindowPos((1280 - 540 - 14, 14), psim.ImGuiCond_Always)
        psim.SetNextWindowSize((540, 0), psim.ImGuiCond_Always)
        opened = psim.Begin("HUD", True)
        if isinstance(opened, tuple):
            opened = opened[0]
        if opened:
            psim.SetWindowFontScale(1.6)
            psim.TextColored((0.7, 0.9, 1.0, 1.0),
                             f"USER REPLAY  K={args.strength:.0f}  grip×{args.gripper_mul}  "
                             f"cloth={'on' if use_cloth else 'off'}")
            psim.SetWindowFontScale(1.0)
            psim.TextColored((0.7, 0.7, 0.7, 1.0),
                             f"  dt={args.dt}  beta_tol={args.beta_tol}  "
                             f"newton_tol={args.newton_tol}  scale={args.scale}")
            psim.Separator()

            ms_recent = list(state["ms_window"])
            if ms_recent:
                avg_ms = sum(ms_recent) / len(ms_recent)
                fps_avg = 1000.0 / max(avg_ms, 1e-6)
                psim.SetWindowFontScale(3.2)
                color = (0.4, 1.0, 0.4, 1.0) if fps_avg > 25 else \
                        (1.0, 0.8, 0.4, 1.0) if fps_avg > 12 else \
                        (1.0, 0.4, 0.4, 1.0)
                psim.TextColored(color, f"{fps_avg:5.1f} fps")
                psim.SetWindowFontScale(1.0)
                psim.Text(f"running-30 avg  ({avg_ms:6.1f} ms/step)")
                psim.Separator()
                psim.Text(f"Last step  : {ms_recent[-1]:6.1f} ms")
                psim.Text(f"Min last 30: {min(ms_recent):6.1f} ms")
                psim.Text(f"Max last 30: {max(ms_recent):6.1f} ms")
            psim.Separator()
            psim.Text(f"Frame: {state['frame']} / {len(qpos_all)}   loops: {state['looped']}")

            psim.Spacing()
            if state["running"]:
                if psim.Button("Pause"):
                    state["running"] = False
            else:
                if psim.Button("Run"):
                    state["running"] = True
            psim.SameLine()
            if psim.Button("Restart"):
                state["frame"] = 0
                state["ms_window"].clear()
        psim.End()

        if state["running"]:
            f = state["frame"]
            if f >= len(qpos_all):
                if args.no_loop:
                    state["running"] = False
                    return
                state["frame"] = 0
                state["looped"] += 1
                f = 0
            a = qpos_all[f]
            for i in range(1, 8):
                robot.set_revolute_position(i, float(a[i - 1]), degree=False)
            for jid in [0, 8, 9, 10, 11, 12]:
                robot.set_revolute_position(jid, float(a[7]), degree=False)

            t0 = time.perf_counter()
            engine.step()
            ms = (time.perf_counter() - t0) * 1000.0
            state["ms_window"].append(ms)
            state["frame"] = f + 1
            mesh.update_vertex_positions(engine.get_vertices())

    ps.set_user_callback(cb)
    ps.show()


if __name__ == "__main__":
    main()
