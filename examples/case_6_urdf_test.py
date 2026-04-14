#!/usr/bin/env python3
"""URDF test robot demo — replicates ./gipc 6.

Loads a small test robot URDF (test_robot.urdf) with motor-driven
revolute joints. Visualization only, no interactive sliders.

Usage:
    python examples/case_6_urdf_test.py
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

    config = Config(assets_dir=ASSETS_DIR,dt=0.01)
    engine = Engine(config)

    tf = np.eye(4)
    tf[:3, :3] *= 0.4
    tf[1, 3] = 1.0
    engine.native.load_urdf(
        engine.native.get_assets_dir() + "sim_data/urdf/test_robot/test_robot.urdf",
        tf, True, True, 1e7,
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
        expanded, _ = psim.Begin("URDF Test (Case 6)", True)
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
