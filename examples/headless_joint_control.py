#!/usr/bin/env python3
"""Headless joint control — no GUI, pure script.

Demonstrates how to programmatically control robot joint angles and read
back simulation results without any visualization dependency.

Only requires: stiff-physics, numpy, scipy

Usage:
    python examples/headless_joint_control.py
"""

import math
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"


def main():
    # ---- Configure and create engine ----
    config = Config(
        assets_dir=ASSETS_DIR,
        dt=0.01,
        joint_strength_ratio=1000.0,
        revolute_driving_strength_ratio=1000.0,
        semi_implicit_enabled=True,
        semi_implicit_beta_tol=1e-3,
        semi_implicit_min_iter=1,
    )
    engine = Engine(config)

    # ---- Load XArm7 robot ----
    arm_tf = np.eye(4)
    arm_tf[:3, :3] = 0.3 * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    arm_tf[1, 3] = -0.9
    engine.native.load_urdf(
        ASSETS_DIR + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        arm_tf, True, False, 1e7,
    )

    # Skip ground collision for all arm bodies
    for bid in range(engine.abd_body_count):
        engine.add_ground_collision_skip(bid)

    # ---- Load shirt (cloth) ----
    shirt_tf = np.eye(4)
    shirt_tf[:3, :3] *= 0.5
    shirt_tf[0, 3] = 0.25
    shirt_tf[1, 3] = 0.3
    engine.load_mesh("triMesh/shirt_6436v.obj", dimensions=2, body_type="FEM",
                     transform=shirt_tf, young_modulus=1e2)

    engine.finalize()
    robot = Robot(engine)

    # ---- Print joint info ----
    print("=== Joint Information ===")
    for ji in robot.revolute_joints:
        print(f"  [{ji.index:2d}] {ji.name:30s}  "
              f"range: [{ji.lower_limit_deg:7.1f}, {ji.upper_limit_deg:7.1f}] deg")
    print()

    # ---- Run simulation with scripted joint control ----
    total_frames = 100
    print(f"=== Running {total_frames} frames ===")

    for frame in range(total_frames):
        # Sweep joint1 from 0 to 60 degrees over 100 frames
        target_deg = (frame / total_frames) * 60.0
        robot.set_revolute_position(0, target_deg, degree=True)

        # You can also control by name:
        # robot.set_joint_position("joint3", 30.0, degree=True)

        engine.step()

        if frame % 10 == 0:
            verts = engine.get_vertices()
            print(f"  Frame {frame:3d}: joint1={target_deg:5.1f}°  "
                  f"verts={verts.shape}  "
                  f"vert[0]=({verts[0,0]:.4f}, {verts[0,1]:.4f}, {verts[0,2]:.4f})")

    print()
    print("=== Simulation complete ===")

    # ---- Read final state ----
    final_verts = engine.get_vertices()
    final_faces = engine.get_surface_faces()
    print(f"Final mesh: {final_verts.shape[0]} vertices, {final_faces.shape[0]} faces")


if __name__ == "__main__":
    main()
