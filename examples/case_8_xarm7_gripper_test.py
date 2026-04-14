#!/usr/bin/env python3
"""XArm7+Gripper test demo — replicates ./gipc 8.

Same URDF as case 7 but a separate test case. Loads xarm7_with_gripper.urdf
at scale 0.3, Y=1.5, joint_strength_ratio=1000. Gravity-only visualization.

Usage:
    python examples/case_8_xarm7_gripper_test.py
"""

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from stiff_physics.engine import Engine, Config
from pathlib import Path

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"



def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(
        assets_dir=ASSETS_DIR,
        dt=0.01,
        joint_strength_ratio=1000.0,
    )
    engine = Engine(config)

    tf = np.eye(4)
    tf[:3, :3] *= 0.3
    tf[1, 3] = 1.5
    engine.native.load_urdf(
        engine.native.get_assets_dir() + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        tf, True, False, 1e7,
    )

    engine.finalize()

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("robot", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    running = [False]
    step_count = [0]

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((300, 0), psim.ImGuiCond_Once)
        expanded, _ = psim.Begin("XArm7+Gripper Test (Case 8)", True)
        if expanded:
            if running[0]:
                if psim.Button("Pause"):
                    running[0] = False
            else:
                if psim.Button("Run"):
                    running[0] = True
            psim.SameLine()
            psim.Text(f"Step: {step_count[0]}")
        psim.End()

        if running[0]:
            engine.step()
            step_count[0] += 1
            mesh.update_vertex_positions(engine.get_vertices())

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
