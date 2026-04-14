#!/usr/bin/env python3
"""Table + cloth demo (no arm) — replicates ./gipc 17.

Fixed ABD table, ABD cube on it, and a FEM cloth sheet above.

Usage:
    python examples/case_17_table_cloth.py
"""

import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from stiff_physics.engine import Engine, Config
from pathlib import Path

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"



def generate_cloth_obj(assets_dir: str, n: int = 100) -> str:
    out_dir = os.path.join(assets_dir, "triMesh")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cloth_{n}x{n}.obj")
    if os.path.exists(path):
        return path
    with open(path, "w") as f:
        for j in range(n + 1):
            for i in range(n + 1):
                f.write(f"v {i / n - 0.5} 0 {j / n - 0.5}\n")
        for j in range(n):
            for i in range(n):
                v0 = j * (n + 1) + i + 1
                f.write(f"f {v0} {v0 + n + 1} {v0 + 1}\n")
                f.write(f"f {v0 + 1} {v0 + n + 1} {v0 + n + 2}\n")
    print(f"[generate_cloth] {n}x{n} -> {(n+1)**2} verts: {path}")
    return path


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(assets_dir=ASSETS_DIR,dt=0.01)
    engine = Engine(config)
    assets_dir = ASSETS_DIR

    table_cx, table_cz, table_top_y = 0.15, 0.0, -0.78

    table_tf = np.eye(4)
    table_tf[0, 0] = 1; table_tf[1, 1] = 0.02; table_tf[2, 2] = 1
    table_tf[0, 3] = table_cx; table_tf[1, 3] = -0.79; table_tf[2, 3] = 0.0
    engine.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                     transform=table_tf, young_modulus=1e9, boundary_type="Fixed")
    table_body_id = engine.abd_body_count - 1
    engine.add_ground_collision_skip(table_body_id)

    cube_scale = 0.1
    px = table_cx + 0.01
    pz = table_cz
    cube_tf = np.eye(4)
    cube_tf[:3, :3] *= cube_scale
    cube_tf[0, 3] = px
    cube_tf[1, 3] = table_top_y + cube_scale * 0.5 - 0.05
    cube_tf[2, 3] = pz
    engine.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                     transform=cube_tf, young_modulus=1e8)

    cloth_res = 100
    cloth_scale = 0.15
    cloth_path = generate_cloth_obj(assets_dir, cloth_res)
    cloth_tf = np.eye(4)
    cloth_tf[:3, :3] *= cloth_scale
    cloth_tf[0, 3] = table_cx
    cloth_tf[1, 3] = table_top_y + 0.1
    cloth_tf[2, 3] = table_cz
    engine.load_mesh(cloth_path, dimensions=2, body_type="FEM",
                     transform=cloth_tf, young_modulus=1e4)

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
        expanded, _ = psim.Begin("Table + Cloth (Case 17)", True)
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
