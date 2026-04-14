#!/usr/bin/env python3
"""Franka Panda (coarse mesh) + Table + Shirt + trajectory — replicates ./gipc 23.

Same as case 21 but uses panda_arm_hand_coarse.urdf for a lower-poly robot.
Trajectory: franka_fold.txt.

Usage:
    python examples/case_23_franka_coarse.py
"""

import math
import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot
from stiff_physics.trajectory import Trajectory
from pathlib import Path

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"



def main():
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    config = Config(
        assets_dir=ASSETS_DIR,
        dt=0.01,
        cloth_thickness=1e-3,
        cloth_young_modulus=1e4,
        cloth_density=1000,
        strain_rate=100,
        soft_motion_rate=1.0,
        poisson_rate=0.49,
        friction_rate=0.4,
        relative_dhat=1e-3,
        joint_strength_ratio=1000.0,
        revolute_driving_strength_ratio=1000.0,
        prismatic_driving_strength_ratio=1000.0,
    )
    engine = Engine(config)
    assets_dir = ASSETS_DIR

    from scipy.spatial.transform import Rotation
    arm_tf = np.eye(4)
    arm_tf[:3, :3] = Rotation.from_rotvec([-math.pi / 2, 0, 0]).as_matrix()
    arm_tf[0, 3] = 0.0
    arm_tf[1, 3] = -0.71
    arm_tf[2, 3] = 0.0
    engine.native.load_urdf(
        assets_dir + "sim_data/urdf/franka_panda/panda_arm_hand_coarse.urdf",
        arm_tf, True, False, 1e7,
    )

    arm_body_count = engine.abd_body_count
    for bid in range(arm_body_count):
        engine.add_ground_collision_skip(bid)

    table_cx, table_cz = 0.55, 0.0
    table_tf = np.eye(4)
    table_tf[0, 0] = 2.0; table_tf[1, 1] = 0.5; table_tf[2, 2] = 2.0
    table_tf[0, 3] = table_cx; table_tf[1, 3] = -0.62; table_tf[2, 3] = table_cz
    engine.load_mesh("tetMesh/cube.msh", dimensions=3, body_type="ABD",
                     transform=table_tf, young_modulus=1e9, boundary_type="Fixed")

    table_body_id = arm_body_count
    for arm_id in range(arm_body_count):
        engine.add_collision_exclusion(table_body_id, arm_id)
    engine.add_ground_collision_skip(table_body_id)

    shirt_tf = np.eye(4)
    shirt_tf[0, 3] = table_cx
    shirt_tf[1, 3] = -0.15
    shirt_tf[2, 3] = table_cz
    engine.load_mesh("triMesh/shirt_831v.obj", dimensions=2, body_type="FEM",
                     transform=shirt_tf, young_modulus=1e4)

    engine.finalize()
    robot = Robot(engine)

    traj = Trajectory(
        os.path.join(assets_dir, "trajectories", "franka_fold.txt"),
        num_revolute=engine.num_revolute_joints,
        num_prismatic=engine.num_prismatic_joints,
    )

    verts = engine.get_vertices()
    faces = engine.get_surface_faces()
    mesh = ps.register_surface_mesh("scene", verts, faces, smooth_shade=True)
    mesh.set_color((0.6, 0.7, 0.8))

    running = [False]
    step_count = [0]
    sim_time = [0.0]
    settling_frames = 10

    def callback():
        psim.SetNextWindowPos((330, 10), psim.ImGuiCond_Once)
        psim.SetNextWindowSize((380, 0), psim.ImGuiCond_Once)

        expanded, _ = psim.Begin("Franka Coarse + Table + Shirt (Case 23)", True)
        if expanded:
            if running[0]:
                if psim.Button("Pause"):
                    running[0] = False
            else:
                if psim.Button("Run"):
                    running[0] = True

            psim.SameLine()
            psim.Text(f"Step: {step_count[0]}  t={sim_time[0]:.3f}s")
        psim.End()

        if running[0]:
            if step_count[0] >= settling_frames:
                sim_time[0] += config._cfg.dt
                rev, pri = traj.interpolate(sim_time[0])
                for i, a in enumerate(rev):
                    if i < engine.num_revolute_joints:
                        engine.set_revolute_target(i, a)
                for i, d in enumerate(pri):
                    if i < engine.num_prismatic_joints:
                        engine.set_prismatic_target(i, d)

            engine.step()
            step_count[0] += 1
            mesh.update_vertex_positions(engine.get_vertices())

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
