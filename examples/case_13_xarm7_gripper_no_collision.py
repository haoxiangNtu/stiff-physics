#!/usr/bin/env python3
"""XArm7+Gripper interactive (no collision) demo — replicates ./gipc 13.

Loads xarm7_with_gripper.urdf with skip_all_collision=True, useful for
testing joint motion without IPC overhead. Interactive joint sliders.

Usage:
    python examples/case_13_xarm7_gripper_no_collision.py
"""

import math
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot
from pathlib import Path

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"



def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(
        assets_dir=ASSETS_DIR,
        dt=0.01,
        joint_strength_ratio=100.0,
        revolute_driving_strength_ratio=100.0,
        skip_all_collision=True,
    )
    engine = Engine(config)

    tf = np.eye(4)
    tf[:3, :3] *= 0.3
    tf[1, 3] = 0.0
    engine.native.load_urdf(
        engine.native.get_assets_dir() + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        tf, True, False, 1e7,
    )

    engine.finalize()
    robot = Robot(engine)

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("robot", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    running = [False]
    step_count = [0]

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((340, 0), psim.ImGuiCond_Once)

        expanded, _ = psim.Begin("XArm7+Gripper No-Collision (Case 13)", True)
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
