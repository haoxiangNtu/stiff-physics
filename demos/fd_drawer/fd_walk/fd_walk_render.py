#!/usr/bin/env python3
"""FD humanoid walking animation via Kimodo-G1 retarget (kinematic FK, no physics).

Pipeline:
  Kimodo-G1 qpos CSV  ->  retarget (1:1 leg mapping + sign calib)  ->  FK on FD
  full visual URDF  ->  polyscope offscreen frames  ->  mp4 (ffmpeg).

Everything is done in the URDF-native / MuJoCo-shared **Z-up, X-forward** frame, so
the G1 root trajectory (MuJoCo z-up) drops straight onto the FD base with no frame
conversion.  Only per-joint SIGN and a constant body-yaw offset need calibration,
which is done by eyeballing rendered frames.

Config knobs (top of file) are meant to be edited during sign calibration.
Run:
  test_stiff08/bin/python3.11 fd_walk_render.py <qpos.csv> <out_dir> [--frames N] [--stride S] [--static]
"""
import os, sys, argparse
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import trimesh

URDF = "/home/ps/Downloads/FD-light/fd-urdf-full/FD-URDF/fd.urdf"
URDF_DIR = os.path.dirname(URDF)

# ---- retarget config (edit during calibration) -----------------------------
# FD leg joints in G1-qpos order: [hip_pitch, hip_roll, hip_yaw, knee, ank_pitch, ank_roll] x (L,R)
FD_LEG_JOINTS = ["leg_l1_joint","leg_l2_joint","leg_l3_joint","leg_l4_joint","leg_l5_joint","leg_l6_joint",
                 "leg_r1_joint","leg_r2_joint","leg_r3_joint","leg_r4_joint","leg_r5_joint","leg_r6_joint"]
# qpos columns 7..18 map 1:1 to FD_LEG_JOINTS above.
# SIGN flips URDF-vs-MuJoCo axis convention per joint (calibrated by rendering).
SIGN = np.array([1,1,1, 1, 1,1,   1,1,1, 1, 1,1], dtype=float)  # [L6, R6]
# constant offset added to each FD leg joint (radians) if zero-poses differ.
OFFSET = np.zeros(12)
# body-frame alignment: extra yaw (deg about Z) so FD "forward" matches G1 +X heading.
BODY_YAW_OFFSET_DEG = 0.0
# drop the base to the ground each frame (ignore G1 root height) -> avoids float/sink.
LOCK_BASE_HEIGHT = None   # set to a float z to pin, or None to use G1 root z


def rot_axis(axis, angle):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-9 or abs(angle) < 1e-12:
        return np.eye(3)
    return R.from_rotvec(axis / n * angle).as_matrix()


def T(Rm=None, t=None):
    M = np.eye(4)
    if Rm is not None: M[:3, :3] = Rm
    if t is not None:  M[:3, 3] = t
    return M


class URDFModel:
    """Minimal URDF FK with visual meshes (Z-up native frame)."""
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.links = {}     # name -> list of (Vlocal, F, color)  visual meshes in link frame
        self.joints = {}    # name -> dict(parent, child, type, origin(4x4), axis)
        self.children = {}  # parent_link -> [joint_name]
        self.parent_joint = {}  # child_link -> joint_name
        for l in root.findall("link"):
            name = l.get("name")
            metas = []
            for v in l.findall("visual"):
                o = v.find("origin")
                oxyz = [float(x) for x in (o.get("xyz","0 0 0").split() if o is not None else [0,0,0])]
                orpy = [float(x) for x in (o.get("rpy","0 0 0").split() if o is not None else [0,0,0])]
                Torigin = T(R.from_euler("xyz", orpy).as_matrix(), oxyz)
                g = v.find("geometry"); m = g.find("mesh") if g is not None else None
                if m is None:
                    continue
                fn = m.get("filename").replace("package://","")
                mp = fn if os.path.isabs(fn) else os.path.join(URDF_DIR, fn)
                if not os.path.exists(mp):
                    continue
                scale = m.get("scale")
                s = np.array([float(x) for x in scale.split()]) if scale else np.ones(3)
                mesh = trimesh.load(mp, force="mesh", process=False)
                Vl = np.asarray(mesh.vertices, float) * s
                # apply visual origin into link frame
                Vl = (Torigin[:3,:3] @ Vl.T).T + Torigin[:3,3]
                Fl = np.asarray(mesh.faces, np.int64)
                col = (0.80,0.82,0.86)
                mat = v.find("material"); c = mat.find("color") if mat is not None else None
                if c is not None and c.get("rgba"):
                    col = tuple(float(x) for x in c.get("rgba").split()[:3])
                metas.append((Vl, Fl, col))
            self.links[name] = metas
        for j in root.findall("joint"):
            n = j.get("name")
            p = j.find("parent").get("link"); c = j.find("child").get("link")
            o = j.find("origin")
            oxyz = [float(x) for x in (o.get("xyz","0 0 0").split() if o is not None else [0,0,0])]
            orpy = [float(x) for x in (o.get("rpy","0 0 0").split() if o is not None else [0,0,0])]
            a = j.find("axis"); axis = [float(x) for x in (a.get("xyz").split() if a is not None else [1,0,0])]
            self.joints[n] = dict(parent=p, child=c, type=j.get("type"),
                                  origin=T(R.from_euler("xyz", orpy).as_matrix(), oxyz), axis=axis)
            self.children.setdefault(p, []).append(n)
            self.parent_joint[c] = n
        # root link = the one that is never a child
        childs = {jd["child"] for jd in self.joints.values()}
        self.root_link = [ln for ln in self.links if ln not in childs][0]

    def fk(self, base_T, jangles):
        """Return {link: world 4x4}. jangles: {joint_name: angle}."""
        world = {self.root_link: base_T}
        stack = [self.root_link]
        while stack:
            pl = stack.pop()
            for jn in self.children.get(pl, []):
                jd = self.joints[jn]
                ang = jangles.get(jn, 0.0)
                if jd["type"] in ("revolute","continuous"):
                    Rj = rot_axis(jd["axis"], ang)
                    Tj = T(Rj)
                elif jd["type"] == "prismatic":
                    Tj = T(t=np.asarray(jd["axis"],float)*ang)
                else:
                    Tj = np.eye(4)
                world[jd["child"]] = world[pl] @ jd["origin"] @ Tj
                stack.append(jd["child"])
        return world

    def posed_scene(self, base_T, jangles):
        """Concatenate all visual meshes at their world poses -> (V, F, Cface)."""
        world = self.fk(base_T, jangles)
        Vs, Fs, Cs = [], [], []
        off = 0
        for ln, metas in self.links.items():
            if ln not in world:
                continue
            W = world[ln]
            for (Vl, Fl, col) in metas:
                Vw = (W[:3,:3] @ Vl.T).T + W[:3,3]
                Vs.append(Vw); Fs.append(Fl + off)
                Cs.append(np.tile(col, (len(Fl),1)))
                off += len(Vw)
        return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Cs)


def load_qpos(csv):
    q = np.loadtxt(csv, delimiter=",")
    if q.ndim == 1: q = q[None]
    assert q.shape[1] >= 19, f"expected >=19 cols, got {q.shape}"
    root_pos = q[:, 0:3]
    root_quat_wxyz = q[:, 3:7]           # mujoco free joint = w,x,y,z
    leg = q[:, 7:19]                     # 12 leg dofs
    return root_pos, root_quat_wxyz, leg


def retarget(root_pos, root_quat_wxyz, leg):
    Tn = len(leg)
    fd_leg = leg * SIGN[None, :] + OFFSET[None, :]     # [T,12]
    yaw_off = R.from_euler("z", np.deg2rad(BODY_YAW_OFFSET_DEG)).as_matrix()
    base = np.zeros((Tn, 4, 4))
    for t in range(Tn):
        w,x,y,z = root_quat_wxyz[t]
        Rb = R.from_quat([x,y,z,w]).as_matrix() @ yaw_off
        pos = root_pos[t].copy()
        if LOCK_BASE_HEIGHT is not None:
            pos[2] = LOCK_BASE_HEIGHT
        base[t] = T(Rb, pos)
    return base, fd_leg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--static", action="store_true", help="freeze base at origin (leg motion only)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sign", type=str, default=None, help="comma list of 12 signs (overrides SIGN)")
    ap.add_argument("--yaw", type=float, default=None, help="body yaw offset deg (overrides BODY_YAW_OFFSET_DEG)")
    ap.add_argument("--ground_clamp", action="store_true", help="shift base so lowest vertex sits on z=0 each frame")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    global SIGN, BODY_YAW_OFFSET_DEG
    if args.sign is not None:
        SIGN = np.array([float(x) for x in args.sign.split(",")], dtype=float)
        assert len(SIGN) == 12, "need 12 signs"
    if args.yaw is not None:
        BODY_YAW_OFFSET_DEG = args.yaw

    root_pos, root_quat, leg = load_qpos(args.csv)
    base, fd_leg = retarget(root_pos, root_quat, leg)
    print(f"[data] {len(leg)} frames; leg range "
          f"[{fd_leg.min():.2f},{fd_leg.max():.2f}] rad; root x {root_pos[:,0].min():.2f}..{root_pos[:,0].max():.2f}")

    print("[urdf] parsing + loading visual meshes ...")
    model = URDFModel(URDF)
    print(f"[urdf] root_link={model.root_link}  links={len(model.links)}  joints={len(model.joints)}")

    import polyscope as ps
    try:
        ps.set_allow_headless_backends(True)   # use EGL when no X display
    except Exception:
        pass
    ps.set_program_name("FD walk"); ps.init()
    ps.set_up_dir("z_up"); ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_view_projection_mode("orthographic")

    idx = list(range(0, len(leg), args.stride))
    if args.frames: idx = idx[:args.frames]
    for k, t in enumerate(idx):
        jang = {jn: fd_leg[t, i] for i, jn in enumerate(FD_LEG_JOINTS)}
        bT = np.eye(4) if args.static else base[t]
        if args.static:
            bT = T(t=[0,0,root_pos[t,2] if not np.isnan(root_pos[t,2]) else 0.9])
        V, F, C = model.posed_scene(bT, jang)
        if args.ground_clamp:
            V[:, 2] -= V[:, 2].min()   # plant lowest vertex on z=0
        ps.remove_all_structures()
        m = ps.register_surface_mesh("fd", V, F, smooth_shade=True)
        m.add_color_quantity("col", C, defined_on="faces", enabled=True)
        # camera: follow the robot from the side-front
        cx = 0.0 if args.static else base[t,0,3]
        cy = 0.0 if args.static else base[t,1,3]
        ps.look_at((cx+2.6, cy-2.6, 1.2), (cx, cy, 0.8))
        out = os.path.join(args.out_dir, f"f{k:04d}.png")
        ps.screenshot(out, transparent_bg=False)
        if k % 10 == 0: print(f"  rendered {k+1}/{len(idx)} -> {out}")
    print(f"[done] {len(idx)} frames in {args.out_dir}")


if __name__ == "__main__":
    main()
