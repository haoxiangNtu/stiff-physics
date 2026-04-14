#!/usr/bin/env python3
"""XArm7+Gripper + Table + Cube + Cloth demo — replicates ./gipc 15.

Loads the XArm7 with gripper, a fixed table (ABD), a small ABD cube on it,
and a FEM cloth sheet above. Interactive joint control.

Usage:
    python examples/case_15_xarm_gripper_soft_cube.py
"""

import math
import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot
from pathlib import Path

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"



def _make_arm_transform(scale: float = 0.3) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    tf = np.eye(4)
    tf[:3, :3] = scale * Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    tf[0, 3] = 0.0
    tf[1, 3] = -0.75
    tf[2, 3] = 0.0
    return tf


def generate_cloth_obj(assets_dir: str, n: int = 15) -> str:
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
    return path


def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(
        assets_dir=ASSETS_DIR,
        dt=0.01,
        joint_strength_ratio=1000.0,
        revolute_driving_strength_ratio=1000.0,
    )
    engine = Engine(config)
    assets_dir = ASSETS_DIR

    arm_tf = _make_arm_transform(0.3)
    engine.native.load_urdf(
        assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf",
        arm_tf, True, False, 1e7,
    )

    arm_body_count = engine.abd_body_count
    for bid in range(arm_body_count):
        engine.add_ground_collision_skip(bid)

    table_cx, table_cz, table_top_y = 0.15, 0.0, -0.78
    table_tf = np.eye(4)
    table_tf[0, 0] = 1; table_tf[1, 1] = 0.02; table_tf[2, 2] = 1
    table_tf[0, 3] = 0.15; table_tf[1, 3] = -0.79; table_tf[2, 3] = 0.0
    engine.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                     transform=table_tf, young_modulus=1e9, boundary_type="Fixed")

    table_body_id = arm_body_count
    for arm_id in range(arm_body_count):
        engine.add_collision_exclusion(table_body_id, arm_id)
    engine.add_ground_collision_skip(table_body_id)

    cube_scale = 0.1
    px = table_cx + 0.01
    cube_tf = np.eye(4)
    cube_tf[:3, :3] *= cube_scale
    cube_tf[0, 3] = px
    cube_tf[1, 3] = table_top_y + cube_scale * 0.5 - 0.05
    cube_tf[2, 3] = table_cz
    engine.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                     transform=cube_tf, young_modulus=1e8)

    cloth_res = 15
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
    robot = Robot(engine)

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=True)
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

        if _begin_window("XArm + Table + Cube + Cloth (Case 15)"):
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
