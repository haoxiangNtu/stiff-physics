#!/usr/bin/env python3
"""XArm7 + Gripper + Table + Cup demo — replicates ./gipc 16.

Loads the XArm7 robot with gripper, a fixed table (ABD), and a soft cup (FEM)
on the table. Joint sliders allow interactive control.

Usage:
    python examples/case_16_xarm_gripper_cup.py
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
    """Replicate the gl_main.cu arm transform: translate(0,-0.75,0), scale, rotate -90 deg X."""
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
        young_modulus=1e7,
        friction_rate=0.4,
        joint_strength_ratio=1000.0,
        revolute_driving_strength_ratio=1000.0,
    )

    engine = Engine(config)

    # --- XArm7 + Gripper (ABD bodies) ---
    arm_tf = _make_arm_transform(0.3)
    engine.native.load_urdf(
        engine.native.get_assets_dir() + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        arm_tf, True, False, 1e7,
    )

    arm_body_count = engine.abd_body_count
    for bid in range(arm_body_count):
        engine.add_ground_collision_skip(bid)

    # --- Table (ABD, Fixed) ---
    table_tf = np.eye(4)
    table_tf[0, 0] = 1.0
    table_tf[1, 1] = 0.02
    table_tf[2, 2] = 1.0
    table_tf[0, 3] = 0.15
    table_tf[1, 3] = -0.79
    table_tf[2, 3] = 0.0

    engine.load_mesh(
        "tetMesh/cube.msh",
        dimensions=3,
        body_type="ABD",
        transform=table_tf,
        young_modulus=1e9,
        boundary_type="Fixed",
    )

    table_body_id = arm_body_count
    for arm_id in range(arm_body_count):
        engine.add_collision_exclusion(table_body_id, arm_id)
    engine.add_ground_collision_skip(table_body_id)

    # --- Cup (FEM, soft) ---
    cup_scale = 0.5
    table_cx, table_cz, table_top_y = 0.15, 0.0, -0.78
    cup_tf = np.eye(4)
    cup_tf[0, 0] = cup_scale
    cup_tf[1, 1] = cup_scale
    cup_tf[2, 2] = cup_scale
    cup_tf[0, 3] = table_cx - cup_scale * 0.1
    cup_tf[1, 3] = table_top_y - cup_scale * 0.01 + 0.02
    cup_tf[2, 3] = table_cz

    engine.load_mesh(
        "sim_data/tetmesh/softgriper_cup.msh",
        dimensions=3,
        body_type="FEM",
        transform=cup_tf,
        young_modulus=1e4,
        boundary_type="Free",
    )

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

        expanded, _ = psim.Begin("XArm7 + Table + Cup (Case 16)", True)
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
