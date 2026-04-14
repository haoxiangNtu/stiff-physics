#!/usr/bin/env python3
"""Box pipe demo — replicates ./gipc 0.

Grid of 4x4x4 ABD cubes above 4x4x4 FEM cubes, falling under gravity.

Usage:
    python examples/case_0_box_pipe.py
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

    dist = 0.2
    count = 4
    count_Y = 4
    abd_height = -0.6
    fem_height = -0.8
    scale = 0.4

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
                                 transform=tf, young_modulus=1e5)

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

                engine.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="FEM",
                                 transform=tf, young_modulus=1e4)

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
        expanded, _ = psim.Begin("Box Pipe (Case 0)", True)
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
