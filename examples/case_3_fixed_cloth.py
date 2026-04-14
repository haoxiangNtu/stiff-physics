#!/usr/bin/env python3
"""Fixed cloth demo — replicates ./gipc 3.

A cloth sheet (cloth_high.obj) with its top corners pinned, hanging under gravity.

Usage:
    python examples/case_3_fixed_cloth.py
"""

import math
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

    from scipy.spatial.transform import Rotation
    scale = 0.6
    cloth_tf = np.eye(4)
    cloth_tf[:3, :3] = scale * Rotation.from_rotvec([math.pi / 2, 0, 0]).as_matrix()
    cloth_tf[0, 3] = 0.0
    cloth_tf[1, 3] = 1.0
    cloth_tf[2, 3] = 0.0

    engine.load_mesh("triMesh/cloth_high.obj", dimensions=2, body_type="FEM",
                     transform=cloth_tf, young_modulus=1e4)

    n_verts = engine.vertex_count_host
    eps = 1e-4
    max_y = -1e30
    min_x = 1e30
    max_x = -1e30
    for i in range(n_verts):
        x, y, z = engine.get_vertex_position_host(i)
        if y > max_y: max_y = y
        if x < min_x: min_x = x
        if x > max_x: max_x = x

    fixed = 0
    for i in range(n_verts):
        x, y, z = engine.get_vertex_position_host(i)
        if y > max_y - eps and (x < min_x + eps or x > max_x - eps):
            engine.set_vertex_boundary(i, 1)
            fixed += 1
    print(f"[case_3] Pinned {fixed} vertices at top corners")

    engine.finalize()

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("cloth", verts, faces, smooth_shade=True)
    mesh.set_color((0.85, 0.55, 0.35))

    running = [False]
    step_count = [0]

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((300, 0), psim.ImGuiCond_Once)
        expanded, _ = psim.Begin("Fixed Cloth (Case 3)", True)
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
