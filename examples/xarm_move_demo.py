#!/usr/bin/env python3
"""XArm interactive joint control demo using StiffGIPC.

Equivalent to the rbs-physics xarm_move_demo.py but using the StiffGIPC
IPC engine as the physics backend with Polyscope visualization.

Usage:
    python examples/xarm_move_demo.py
"""

import math
import polyscope as ps
import polyscope.imgui as psim
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot
from pathlib import Path

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"



def main():
    # Must init Polyscope (OpenGL) BEFORE creating the CUDA engine
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(
        assets_dir=ASSETS_DIR,
        dt=0.01,
        young_modulus=1e7,
        friction_rate=0.4,
        joint_strength_ratio=100.0,
        revolute_driving_strength_ratio=100.0,
    )

    engine = Engine(config)
    engine.load_urdf(
        "sim_data/urdf/xarm/xarm6_robot_white.urdf",
        scale=0.3,
        root_fixed=True,
    )
    engine.finalize()
    robot = Robot(engine)

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("robot", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    running = [False]
    step_count = [0]

    def _begin_window(title):
        """Compatible with both old polyscope (returns bool) and new (returns tuple)."""
        result = psim.Begin(title, True)
        return result[0] if isinstance(result, tuple) else result

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((340, 0), psim.ImGuiCond_Once)

        if _begin_window("StiffGIPC Control"):
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
