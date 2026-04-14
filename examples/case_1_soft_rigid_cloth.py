#!/usr/bin/env python3
"""Soft-rigid-cloth coupling demo — replicates ./gipc 1.

ABD bunny (rigid) + FEM bunny (soft) + FEM cloth sheet.

Usage:
    python examples/case_1_soft_rigid_cloth.py
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

    scale = 0.2

    tf_abd = np.eye(4)
    tf_abd[:3, :3] *= scale
    tf_abd[0, 3] = 0.0
    tf_abd[1, 3] = 0.5
    tf_abd[2, 3] = 0.0
    engine.load_mesh("tetMesh/bunny2.msh", dimensions=3, body_type="ABD",
                     transform=tf_abd, young_modulus=1e4)

    tf_fem = np.eye(4)
    tf_fem[:3, :3] *= scale
    tf_fem[0, 3] = 0.0
    tf_fem[1, 3] = -0.65
    tf_fem[2, 3] = 0.0
    engine.load_mesh("tetMesh/bunny2.msh", dimensions=3, body_type="FEM",
                     transform=tf_fem, young_modulus=1e4)

    tf_cloth = np.eye(4)
    engine.load_mesh("triMesh/cloth_high.obj", dimensions=2, body_type="FEM",
                     transform=tf_cloth, young_modulus=1e4)

    engine.finalize()

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    running = [False]
    step_count = [0]

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((300, 0), psim.ImGuiCond_Once)
        expanded, _ = psim.Begin("Soft-Rigid-Cloth (Case 1)", True)
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
