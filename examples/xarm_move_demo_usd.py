#!/usr/bin/env python3
"""XArm joint control demo — URDF-to-USD pipeline (Path B).

Same robot as xarm_move_demo.py but uses the unified USD pipeline:
    UrdfLoader → Urdf2Usd → StiffGipcUsdParser → engine

All collision meshes go through the Python approximation pipeline
(physics:approximation → CoACD / convex_hull / etc.) matching rbs-physics.

Comparison with Path A (xarm_move_demo.py):
  - Path A:  engine.load_urdf()  → C++ UrdfSceneImporter  (direct)
  - Path B:  Urdf2Usd → USD stage → StiffGipcUsdParser    (unified)

Usage:
    conda run -n env_isaaclab python examples/xarm_move_demo_usd.py

Requires: pxr (USD), trimesh, urdf-parser-py, scipy
"""

import math
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from pxr import Usd

from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot
from stiff_physics.urdf2usd import Urdf2Usd
from stiff_physics.usd_scene_parser import StiffGipcUsdParser
from pathlib import Path

ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"



def main():
    ps.init()
    ps.set_up_dir("z_up")
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

    # --- Step 1: Convert URDF → USD (in-memory) ---
    urdf_path = ASSETS_DIR + "sim_data/urdf/xarm/xarm7_robot.urdf"
    stage = Usd.Stage.CreateInMemory()
    Urdf2Usd.setup_stage(stage, up_axis="Z", meters_per_unit=1.0)
    converter = Urdf2Usd(stage)
    converter.from_urdf_file(urdf_path)
    print(f"[demo] URDF → USD conversion done: {urdf_path}")

    # --- Step 2: Parse USD stage into engine ---
    # skip_mesh_approximation=True because the URDF collision meshes are
    # already simple.  Set to False to enable automatic CoACD / convex_hull.
    parser = StiffGipcUsdParser(engine=engine, stage=stage)
    info = parser.parse_and_build(
        env_scope_path="/Robot",
        skip_mesh_approximation=True,
    )
    print(f"[demo] Loaded {engine.abd_body_count} ABD bodies, "
          f"{engine.num_revolute_joints} revolute joints")

    engine.finalize()
    robot = Robot(engine)

    # --- Visualize ---
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
        psim.SetNextWindowSize((380, 0), psim.ImGuiCond_Once)

        if _begin_window("XArm (URDF→USD Pipeline)"):
            if running[0]:
                if psim.Button("Pause"):
                    running[0] = False
            else:
                if psim.Button("Run"):
                    running[0] = True

            psim.SameLine()
            psim.Text(f"Step: {step_count[0]}")
            psim.Separator()

            psim.Text(f"ABD bodies: {engine.abd_body_count}")
            psim.Text(f"Revolute joints: {engine.num_revolute_joints}")
            psim.Text("Pipeline: URDF -> Urdf2Usd -> StiffGipcUsdParser")
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
