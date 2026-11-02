"""Xense parallel gripper with two visuotactile gel pads pinches an engraved mahjong tile and lifts it.

The tactile pinch of Taccel's Figure 1 (previously done here with an Allegro hand) on a real two-finger
gripper: the Xense gripper (URDF package in ``Assets/xense_gripper``) closes its two fingers on a mahjong tile
standing on the table, regulates the grip force from the pads' own contact force, lifts the tile 40 mm, holds it,
puts it back and opens.  Each finger carries a 24.0 x 16.0 x 2.5 mm FEM gel pad on its sensing window; the left
pad presses the engraved face, so its deformation map shows the character.

Simulation model (one StiffGIPC Newton solve per step, IPC contact everywhere):

    anchor (Fixed)  --prismatic "lift" (world y)-->  HAND (adapter + mount plate + gripper body + D435i)
    HAND  --prismatic "slider_left"  (closing = world -z)-->  finger_left  ==pinned==  gel pad (FEM)
    HAND  --prismatic "slider_right" (closing = world +z)-->  finger_right ==pinned==  gel pad (FEM)
    tile (free ABD, 15 g) on the ground plane

The bodies, joint origins and axes come from the URDF forward kinematics (``xense_assets.XenseGripper``); the
URDF's mimic (slider_right = -slider_left) is applied by commanding both slider joints with the same closing
travel.  Joint travel is measured from the load pose (fingers loaded part-closed, 3 mm clear of the tile).

Output: ``<out>/xense_mahjong.npz`` -- per recorded frame the hand / finger / tile poses, both pads' vertex
positions (for the tactile images and the render), pad normal forces, joint travel and the controller state;
plus the static meshes and the pad assets needed to render it.  ``render_xense_mahjong.py`` turns that into
tactile images (depth / deformation / RGB / marker flow), the glyph IoU score and the video.

    PYTHONPATH=. python examples/xense_mahjong/xense_mahjong.py --out out/xense_mahjong
    PYTHONPATH=. python examples/xense_mahjong/xense_mahjong.py --grip-force 10 --char 發

No torch is needed here (the solver machine has none); the tactile images are made from the recording.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from xense_assets import (GEL_E, REPO, FINGERS, PAD_T, R_WORLD_BASE, TILE, XenseGripper, load_pad_asset,  # noqa: E402
                          T_of, tile)

TILE_GAP = 1.0e-3        # tile starts this far above the table
TIP_CLEAR = 1.5e-3       # fingertips this far above the table at the grasp height
APPROACH_GAP = 2.0e-3    # pad coat -> tile face at load


def abd_world_tf(C: np.ndarray) -> np.ndarray:
    """get_abd_body_transforms() stores the affine block TRANSPOSED; the physical rigid motion has
    R = C[:3, :3].T (verified on the FR3 lab scene: without it every joint move shows a 2*theta error)."""
    T = np.array(C, np.float64)
    T[:3, :3] = C[:3, :3].T
    return T


class XenseMahjongScene:
    def __init__(self, char="中", dt=0.005, newton_tol=1e-4, pcg_tol=1e-6, dhat=1e-4, mu=0.8, gel_E=GEL_E,
                 log_level=0):
        from stiff_physics.engine import Config, Engine
        from stiff_physics.sensors.backend_stiffgipc import GelBody, clear_mesh_cache

        self.dt = dt
        self.g = g = XenseGripper()
        self.tile_data = tile(char)
        self.pad_asset = load_pad_asset(gel_E)

        # ---------------------------------------------------------------- placement
        tv = self.tile_data["verts"]
        self.half_t = 0.5 * (tv[:, 2].max() - tv[:, 2].min())
        half_h = 0.5 * (tv[:, 1].max() - tv[:, 1].min())
        self.T_tile0 = T_of(p=[0.0, half_h + TILE_GAP, 0.0])        # upright, engraved face +z
        # gripper-body origin height: fingertips TIP_CLEAR above the table
        self.h0 = g.tip_z() + TIP_CLEAR
        # load the fingers part-closed so the coats are APPROACH_GAP clear of the tile faces
        wl = g.window("left")
        self.q0 = wl["y"] - PAD_T - self.half_t - APPROACH_GAP
        self.link_W0 = g.world_fk([0.0, self.h0, 0.0], self.q0)
        R_root = self.link_W0[g.root]
        sim = g.body_meshes(sim=True)                                 # root frame, q = 0
        # fingers: body meshes are at q = 0 in the root frame; shift to the load q along the closing axis
        self.body_T0 = {}
        T_q0 = g.fk(self.q0)
        T_00 = g.fk(0.0)
        self.body_T0["hand"] = R_root
        for s, link in FINGERS.items():
            self.body_T0[s] = R_root @ T_q0[link] @ np.linalg.inv(T_00[link])

        # ---------------------------------------------------------------- engine
        # engine assets dir: finalize() reads <assets>/scene/abd_system_config.json (shipped with this example, identical
        # to the copy inside the stiff-physics wheel) and keeps its mesh caches there
        assets_root = osp.join(REPO, "Assets")
        clear_mesh_cache(assets_root)
        cfg = Config(dt=dt, density=1000.0, young_modulus=1e7, poisson_rate=0.45, friction_rate=mu,
                     gravity=(0.0, -9.81, 0.0), ground_offset=0.0, ground_normal=(0.0, 1.0, 0.0),
                     prismatic_strength_ratio=2000.0, prismatic_driving_strength_ratio=100.0,
                     max_prismatic_step_per_frame=1.5e-3, absolute_dhat=dhat, preconditioner_type=0,
                     newton_tol=newton_tol, pcg_tol=pcg_tol, assets_dir=assets_root + "/", soft_motion_rate=1e4)
        self.eng = eng = Engine(cfg)
        eng.set_log_level(log_level)
        self.ids, self.v_off, self.rest = {}, {}, {}

        def load_abd(name, v, f, T, fixed=False):
            wv = np.asarray(v) @ T[:3, :3].T + T[:3, 3]
            eng.load_mesh_from_data(wv, np.asarray(f, np.int32), verts_per_face=3, dimensions=3, body_type="ABD",
                                    transform=np.eye(4), young_modulus=1e8,
                                    boundary_type="Fixed" if fixed else "Free")
            rec = eng.get_load_records()[-1]
            self.ids[name] = int(rec.body_offset)
            self.v_off[name] = int(rec.vertex_offset)
            self.rest[name] = (wv, np.asarray(f, np.int32))
            return self.ids[name]

        # anchor far above, the hand, the two fingers, the tile (ABD first, FEM last)
        a = 0.006
        av = np.array([[x, y, z] for x in (-a, a) for y in (-a, a) for z in (-a, a)])
        af = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                       [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int32)
        load_abd("anchor", av, af, T_of(p=[0.0, self.h0 + 0.30, 0.0]), fixed=True)
        load_abd("hand", sim["hand"].vertices, sim["hand"].faces, self.body_T0["hand"])
        for s in ("left", "right"):
            load_abd(s, sim[s].vertices, sim[s].faces, self.body_T0[s])
        load_abd("tile", tv, self.tile_data["faces"], self.T_tile0)
        for n in ("anchor", "hand"):
            eng.add_ground_collision_skip(self.ids[n])
        # Mass properties.  The gripper's decimated meshes are not closed, so the volume integrals the engine would
        # take from them are meaningless (scaling them with set_abd_body_mass() froze the whole solve; the public
        # 0.8.5.5 wheel already returns NaN energy with the mesh-derived values).  Use the URDF <inertial> tags:
        # hand = adapter + mount plate + gripper body + D435i combined (0.83 kg), fingers 0.07 kg each.
        # Tile: closed mesh, 2400 kg/m^3 -> 15 g (a real tile).
        hand_links = ("tool_adapter_link", "gripper_mount_plate", "gripper_base_link", "wrist_d435i_link")
        for n, links in (("hand", hand_links), ("left", (FINGERS["left"],)), ("right", (FINGERS["right"],))):
            m, c, I = g.group_inertia(links, self.link_W0)
            eng.set_abd_body_inertia(self.ids[n], m, c, I)
            self.mass = getattr(self, "mass", {}) | {n: m}
        eng.set_abd_body_density(self.ids["tile"], 2400.0)
        eng.set_body_friction(self.ids["tile"], mu)

        # joints: lift (anchor -> hand, world +y) and the two sliders (hand -> finger, + = closing)
        hand_o = self.body_T0["hand"][:3, 3]
        self.j_lift = eng.add_prismatic_joint(self.ids["anchor"], self.ids["hand"], hand_o, [0, 1, 0],
                                              -0.10, 0.10, "lift")
        close_dir = {"left": R_WORLD_BASE @ [0, -1, 0], "right": R_WORLD_BASE @ [0, 1, 0]}
        self.j_slide = {}
        for s, link in FINGERS.items():
            o = self.link_W0[link][:3, 3]
            # travel limit: q0 + d <= q_max (URDF), and the pads must not cross (checked by the controller)
            self.j_slide[s] = eng.add_prismatic_joint(self.ids["hand"], self.ids[s], o, close_dir[s],
                                                      -self.q0, g.q_max - self.q0, f"slider_{s}")
        eng.add_collision_exclusion(self.ids["left"], self.ids["right"])

        # gel pads pinned on the finger sensing windows
        self.gels, self.pad_tf0 = {}, {}
        for s, link in FINGERS.items():
            W = self.link_W0[link] @ g.pad_frame_in_finger(s)
            gel = GelBody(self.pad_asset, eng, world_tf=W)
            gel.add_pinned(self.ids[s], self.v_off[s])
            self.gels[s], self.pad_tf0[s] = gel, W
            for n in ("hand", "left", "right"):
                eng.add_collision_exclusion(gel.body_offset, self.ids[n])
        eng.add_collision_exclusion(self.gels["left"].body_offset, self.gels["right"].body_offset)
        self._check_start()

        t0 = time.time()
        eng.finalize()
        print(f"[xm] finalize {time.time()-t0:.1f}s: {eng.vertex_count} verts, {eng.abd_body_count} ABD, "
              f"{eng.fem_body_count} FEM", flush=True)
        ids = np.array([self.ids[n] for n in ("hand", "left", "right", "tile")], np.int32)
        self._pose_ids = ids
        C0 = eng.get_abd_body_transforms(ids)
        self.C0inv = {n: np.linalg.inv(abd_world_tf(c)) for n, c in zip(("hand", "left", "right", "tile"), C0)}
        for j in (self.j_lift, *self.j_slide.values()):
            eng.native.set_prismatic_force(j, 0.0)

    def _check_start(self):
        """Fail fast on bad placement (IPC cannot recover from initial penetration)."""
        tv = self.tile_data["verts"] @ self.T_tile0[:3, :3].T + self.T_tile0[:3, 3]
        lo, hi = tv.min(0), tv.max(0)
        for s, W in self.pad_tf0.items():
            p = self.pad_asset.points @ W[:3, :3].T + W[:3, 3]
            inside = np.all((p > lo + 1e-5) & (p < hi - 1e-5), axis=1).sum()
            coat = p[self.pad_asset.coat_mask]
            gap = (coat[:, 2].min() - hi[2]) if s == "left" else (lo[2] - coat[:, 2].max())
            print(f"[xm] pad {s}: coat z {coat[:,2].min()*1e3:+.2f}..{coat[:,2].max()*1e3:+.2f} mm, "
                  f"y {coat[:,1].min()*1e3:.1f}..{coat[:,1].max()*1e3:.1f} mm, gap to tile {gap*1e3:.2f} mm, "
                  f"inside {inside}", flush=True)
            if inside or gap < 0.5e-3:
                raise SystemExit(f"pad {s} starts inside / touching the tile")
        for s in ("left", "right"):
            v = self.rest[s][0]
            print(f"[xm] finger {s}: lowest point y = {v[:,1].min()*1e3:.2f} mm", flush=True)

    # ------------------------------------------------------------------- state
    def poses(self) -> dict:
        C = self.eng.get_abd_body_transforms(self._pose_ids)
        return {n: abd_world_tf(c) @ self.C0inv[n] for n, c in zip(("hand", "left", "right", "tile"), C)}

    def travel(self) -> dict:
        return {s: float(self.eng.get_prismatic_current_distance(j)) for s, j in self.j_slide.items()}

    def lift(self) -> float:
        return float(self.eng.get_prismatic_current_distance(self.j_lift))

    def pad_forces(self, P=None) -> dict:
        """Per pad: (normal force N, local force vector) from one contact-force fetch."""
        P = P or self.poses()
        F = np.asarray(self.eng.get_vertex_contact_forces(False, "total"))
        out = {}
        for s, gel in self.gels.items():
            W = P[s] @ self.pad_tf0[s]
            Fl = F[gel.slice] @ W[:3, :3]
            out[s] = (float(np.clip(Fl[:, 2], 0.0, None).sum()), Fl.sum(0))
        return out


class GripController:
    """Slider servo in the two vendor-style modes the demo needs.

    POSITION: move the (common) closing travel towards a target at <= vmax; while either pad's normal force
              exceeds fmax the travel stops advancing (force-capped position move).
    FORCE:    integral regulation of the mean pad normal force to ``grip_force`` (the tactile force loop);
              until first touch it closes at the approach speed.
    Both sliders always get the same target: that is the URDF mimic (slider_right = -slider_left).
    """

    def __init__(self, scene: XenseMahjongScene, stiffness=25.0):
        self.s, self.k = scene, stiffness
        self.mode, self.d = "pos", 0.0
        self.cmd, self.vmax, self.fmax = 0.0, 0.03, 20.0
        # The pressed pad is stiff (~80 N/mm per side at E = 0.2 MPa over the full face, ~200 N/mm at 0.5 MPa):
        # with ki = 2e-5 m/N the loop gain ki*k was 4 and the force limit-cycled 7 <-> 17 N every step.
        # ki = 3e-6 keeps ki*k <= 0.6 for both; the approach is slow so first touch does not overshoot.
        self.grip_force, self.ki, self.d_step = 10.0, 3.0e-6, 0.02e-3
        self.approach_v = 0.005
        self.d_max = scene.g.q_max - scene.q0
        self.lift_cmd, self.lift_d, self.lift_v = 0.0, 0.0, 0.04

    def update(self, fl, fr):
        dt = self.s.dt
        f = max(fl, fr)
        if self.mode == "pos":
            step = float(np.clip(self.cmd - self.d, -self.vmax * dt, self.vmax * dt))
            if step > 0 and f > self.fmax:
                step = 0.0
            self.d += step
        else:
            fm = 0.5 * (fl + fr)
            if fm < 0.2:
                self.d += self.approach_v * dt
            else:
                self.d += float(np.clip(self.ki * (self.grip_force - fm), -self.d_step, self.d_step))
        self.d = float(np.clip(self.d, -self.s.q0, self.d_max))
        self.lift_d += float(np.clip(self.lift_cmd - self.lift_d, -self.lift_v * dt, self.lift_v * dt))
        e = self.s.eng
        for j in self.s.j_slide.values():
            e.native.set_prismatic_strength(j, self.k)
            e.set_prismatic_target(j, self.d)
        e.native.set_prismatic_strength(self.s.j_lift, self.k)
        e.set_prismatic_target(self.s.j_lift, self.lift_d)


def timeline(grip_force):
    """(duration s, label, controller settings)."""
    return [
        (0.20, "settle", dict(mode="pos", cmd=0.0)),
        (0.70, "close", dict(mode="force", grip_force=grip_force)),
        (0.40, "squeeze", dict(mode="force", grip_force=grip_force)),
        (1.20, "lift", dict(mode="force", grip_force=grip_force, lift_cmd=0.040)),
        (0.60, "hold", dict(mode="force", grip_force=grip_force, lift_cmd=0.040)),
        (1.20, "lower", dict(mode="force", grip_force=grip_force, lift_cmd=0.0)),
        (0.80, "release", dict(mode="pos", cmd=-0.004)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--char", default="中")
    ap.add_argument("--grip-force", type=float, default=10.0, help="pad normal force setpoint, N (URDF effort 20 N)")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--mu", type=float, default=0.8, help="gel / tile friction")
    ap.add_argument("--gel-E", type=float, default=GEL_E, help="gel Young's modulus, Pa")
    ap.add_argument("--rec-every", type=int, default=2)
    ap.add_argument("--max-time", type=float, default=0.0, help="stop early (s), 0 = whole timeline")
    ap.add_argument("--out", default="out/xense_mahjong")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sc = XenseMahjongScene(char=a.char, dt=a.dt, mu=a.mu, gel_E=a.gel_E)
    ctl = GripController(sc)
    tl = timeline(a.grip_force)
    t_end = sum(d for d, _, _ in tl)
    if a.max_time > 0:
        t_end = min(t_end, a.max_time)
    nsteps = int(round(t_end / a.dt))
    bounds = np.cumsum([0.0] + [d for d, _, _ in tl])

    rec = {k: [] for k in ("t", "stage", "hand", "left", "right", "tile", "gel_left", "gel_right", "fn_left",
                           "fn_right", "f_left", "f_right", "travel_cmd", "travel_left", "travel_right",
                           "lift_cmd", "lift", "newton")}
    tile0 = sc.poses()["tile"][:3, 3].copy()
    t0 = time.time()
    stage_i = -1
    status = "ok"
    for k in range(nsteps):
        t = k * a.dt
        i = int(np.searchsorted(bounds, t, side="right") - 1)
        if i != stage_i:
            stage_i = i
            for key, v in tl[i][2].items():
                setattr(ctl, key, v)
            print(f"[xm] t={t:5.2f}s stage {tl[i][1]}", flush=True)
        P = sc.poses()
        pf = sc.pad_forces(P)
        ctl.update(pf["left"][0], pf["right"][0])
        sc.eng.step()
        if not np.isfinite(np.asarray(sc.eng.get_vertices())[:10]).all():
            status = "nan"
            print("[xm] NaN -> stop", flush=True)
            break
        if k % a.rec_every == 0 or k == nsteps - 1:
            P = sc.poses()
            pf = sc.pad_forces(P)
            tr = sc.travel()
            rec["t"].append(t + a.dt); rec["stage"].append(i)
            for n in ("hand", "left", "right", "tile"):
                rec[n].append(P[n])
            for s in ("left", "right"):
                rec[f"gel_{s}"].append(sc.gels[s].verts_world().astype(np.float32))
                rec[f"fn_{s}"].append(pf[s][0]); rec[f"f_{s}"].append(pf[s][1])
                rec[f"travel_{s}"].append(tr[s])
            rec["travel_cmd"].append(ctl.d); rec["lift_cmd"].append(ctl.lift_d); rec["lift"].append(sc.lift())
            rec["newton"].append(0)
        if k % 20 == 0:
            P = sc.poses()
            pf = sc.pad_forces(P)
            tz = (P["tile"][:3, 3] - tile0) * 1e3
            print(f"[xm] t={t+a.dt:5.2f} {tl[i][1]:8s} travel {ctl.d*1e3:6.2f} mm (L {sc.travel()['left']*1e3:6.2f}) "
                  f"F L {pf['left'][0]:6.2f} R {pf['right'][0]:6.2f} N  lift {sc.lift()*1e3:6.2f} mm  "
                  f"tile d {tz.round(2)} mm  wall {time.time()-t0:6.1f}s", flush=True)

    out = osp.join(a.out, "xense_mahjong.npz")
    stat = dict(status=status, char=a.char, grip_force=a.grip_force, dt=a.dt, mu=a.mu, gel_E=a.gel_E, steps=k + 1,
                wall_s=time.time() - t0, q0_mm=sc.q0 * 1e3, h0_mm=sc.h0 * 1e3, tile=TILE,
                stages=[[float(bounds[i]), lab] for i, (_, lab, _) in enumerate(tl)])
    np.savez_compressed(
        out, **{k: np.asarray(v) for k, v in rec.items()},
        hand_rest_v=sc.rest["hand"][0], hand_rest_f=sc.rest["hand"][1],
        left_rest_v=sc.rest["left"][0], left_rest_f=sc.rest["left"][1],
        right_rest_v=sc.rest["right"][0], right_rest_f=sc.rest["right"][1],
        tile_rest_v=sc.rest["tile"][0], tile_rest_f=sc.rest["tile"][1],
        link_W0=np.array([sc.link_W0[n] for n in sorted(sc.link_W0)]), link_names=np.array(sorted(sc.link_W0)),
        pad_tf0_left=sc.pad_tf0["left"], pad_tf0_right=sc.pad_tf0["right"],
        tile_T0=sc.T_tile0, stats=np.array(json.dumps(stat)))
    with open(osp.join(a.out, "stats.json"), "w") as f:
        json.dump(stat, f, indent=1)
    st_arr = np.asarray(rec["stage"])
    labels = [lab for _, lab, _ in tl]
    i_lift = int(np.argmax(st_arr == labels.index("lift"))) if (st_arr == labels.index("lift")).any() else 0
    tile_dy = (np.asarray(rec["tile"])[:, 1, 3] - np.asarray(rec["tile"])[i_lift, 1, 3]) * 1e3   # from lift start
    lift = (np.asarray(rec["lift"]) - rec["lift"][i_lift]) * 1e3
    held = st_arr == labels.index("hold")
    slip = float(np.max(np.abs(lift[held] - tile_dy[held]))) if held.any() else float("nan")
    print(f"[xm] done {status}: {k+1} steps, wall {time.time()-t0:.0f}s; tile raised {tile_dy.max():.1f} mm "
          f"(lift {lift.max():.1f} mm), slip during hold {slip:.2f} mm -> {out}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
