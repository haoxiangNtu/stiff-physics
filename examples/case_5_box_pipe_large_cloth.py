#!/usr/bin/env python3
"""Large box pipe + cloth demo — replicates ./gipc 5.

Large grid of ABD cubes (8x8 x 15 layers, two stiffness tiers) with a
FEM cloth sheet pinned at X edges draped over the top.

WARNING: This scene is very large (~1920 ABD cubes + cloth). Initialization
takes significant time and memory.

Usage:
    python examples/case_5_box_pipe_large_cloth.py
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
        relative_dhat=1e-3,
        strain_rate=1e6,
        linear_system_buff_scale=2.0,
    )
    engine = Engine(config)

    scale = 0.3
    dist = scale / 2
    count = 8
    count_Y = 15
    global_offset = 1.0
    fem_height = global_offset + 1 - 0.8
    abd_height = fem_height - dist

    for k in range(count_Y):
        for i in range(count):
            for j in range(count):
                px = i * dist - dist * (count - 1) / 2.0
                pz = j * dist - dist * (count - 1) / 2.0
                py = abd_height + 2 * dist * k

                tf = np.eye(4)
                tf[:3, :3] *= scale
                tf[0, 3] = px
                tf[1, 3] = py
                tf[2, 3] = pz
                engine.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                                 transform=tf, young_modulus=1e6)

    for k in range(count_Y):
        for i in range(count):
            for j in range(count):
                px = i * dist - dist * (count - 1) / 2.0
                pz = j * dist - dist * (count - 1) / 2.0
                py = fem_height + 2 * dist * k

                tf = np.eye(4)
                tf[:3, :3] *= scale
                tf[0, 3] = px
                tf[1, 3] = py
                tf[2, 3] = pz
                engine.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                                 transform=tf, young_modulus=5e4)

    cloth_tf = np.eye(4)
    cloth_tf[:3, :3] *= 1.5
    cloth_tf[1, 3] = 0.35 * 1.5
    engine.load_mesh("triMesh/cloth_high.obj", dimensions=2, body_type="FEM",
                     transform=cloth_tf, young_modulus=1e4)

    n_verts = engine.vertex_count_host
    eps = 1e-4
    fixed = 0
    for i in range(n_verts):
        x, y, z = engine.get_vertex_position_host(i)
        if x < -1.5 + eps or x > 1.5 - eps:
            engine.set_vertex_boundary(i, 1)
            fixed += 1
    print(f"[case_5] Pinned {fixed} cloth edge vertices")

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
        expanded, _ = psim.Begin("Box Pipe + Cloth (Case 5)", True)
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
