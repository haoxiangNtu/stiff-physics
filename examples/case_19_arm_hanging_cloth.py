#!/usr/bin/env python3
"""Arm + Hanging cloth demo — replicates ./gipc 19.

XArm7+Gripper URDF + pre-made cloth_high.obj mesh (FEM 2D) with top corners
pinned. The arm can interact with the hanging cloth via joint sliders.

Similar to the existing arm_hanging_cloth_demo.py (case 20) but uses the
pre-made cloth_high.obj mesh instead of a procedurally generated one.

Usage:
    python examples/case_19_arm_hanging_cloth.py
"""

import math
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot
from pathlib import Path

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"



def _make_arm_transform(scale: float = 0.3) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[0, 3] = 0.0
    tf[1, 3] = -0.75
    tf[2, 3] = 0.0
    return tf


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(
        assets_dir=ASSETS_DIR,
        dt=0.01,
        cloth_thickness=1e-3,
        joint_strength_ratio=1000.0,
        revolute_driving_strength_ratio=1000.0,
    )
    engine = Engine(config)
    assets_dir = ASSETS_DIR

    arm_tf = _make_arm_transform(0.3)
    engine.native.load_urdf(
        assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        arm_tf, True, False, 1e7,
    )

    arm_body_count = engine.abd_body_count
    for bid in range(arm_body_count):
        engine.add_ground_collision_skip(bid)

    fem_vert_start = engine.vertex_count_host

    from scipy.spatial.transform import Rotation
    cloth_scale = 0.4
    cloth_tf = np.eye(4)
    cloth_tf[:3, :3] = cloth_scale * Rotation.from_rotvec([math.pi / 2, 0, 0]).as_matrix()
    cloth_tf[0, 3] = 0.5
    cloth_tf[1, 3] = -0.3
    cloth_tf[2, 3] = 0.0

    engine.load_mesh("triMesh/cloth_high.obj", dimensions=2, body_type="FEM",
                     transform=cloth_tf, young_modulus=1e4)

    fem_vert_end = engine.vertex_count_host
    eps = 1e-4
    max_y = -1e30
    min_x = 1e30
    max_x = -1e30
    for i in range(fem_vert_start, fem_vert_end):
        x, y, z = engine.get_vertex_position_host(i)
        if y > max_y: max_y = y
        if x < min_x: min_x = x
        if x > max_x: max_x = x

    fixed = 0
    for i in range(fem_vert_start, fem_vert_end):
        x, y, z = engine.get_vertex_position_host(i)
        if y > max_y - eps and (x < min_x + eps or x > max_x - eps):
            engine.set_vertex_boundary(i, 1)
            fixed += 1
    print(f"[case_19] Pinned {fixed} cloth vertices at top corners")

    engine.finalize()
    robot = Robot(engine)

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    running = [False]
    step_count = [0]

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((340, 0), psim.ImGuiCond_Once)

        expanded, _ = psim.Begin("Arm + Hanging Cloth (Case 19)", True)
        if expanded:
            if running[0]:
                if psim.Button("Pause"):
                    running[0] = False
            else:
                if psim.Button("Run"):
                    running[0] = True

            psim.SameLine()
            psim.Text(f"Step: {step_count[0]}")
            psim.Separator()

            if robot.revolute_joints:
                psim.Text("Revolute Joints")
                psim.Separator()
                for i, ji in enumerate(robot.revolute_joints):
                    lo = math.degrees(ji.lower_limit)
                    hi = math.degrees(ji.upper_limit)
                    cur = robot.get_revolute_target_deg(i)
                    changed, new_val = psim.SliderFloat(ji.name, cur, lo, hi)
                    if changed:
                        robot.set_revolute_position(i, new_val, degree=True)

            psim.Spacing()
            if psim.Button("Reset All Joints"):
                robot.reset_all()

        psim.End()

        if running[0]:
            engine.step()
            step_count[0] += 1
            mesh.update_vertex_positions(engine.get_vertices())

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
