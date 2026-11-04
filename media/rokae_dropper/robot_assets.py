"""Rokae AR5 dual-arm robot with two Wuji five-finger hands: URDF builder + forward kinematics.

Sources (Assets/rokae_wuji/src):
  rokae/urdf/AR5-5_08LR-W4C4A6-ZY2.urdf   the dual-arm description (two 7-DoF AR5 arms on a column)
  wuji_usd/{left,right}/*.stl              Wuji hand meshes (visual + *_collision) in the joint description's link
                                           frames (exported from the dual-arm Wuji USD)
  wuji_usd_joints.json                     Wuji hand joint frames for both hands (palm -> fingers)

The arm URDF carries no hand. The hands are mounted on the arm flanges with a roll about the flange
axis (HAND_ROLL); the recording's camera images show both hands grasping palm-down at the grasps,
which fixes the roll at about -60 deg (right) / +60 deg (left).

Outputs (Assets/rokae_wuji):
  ar5_wuji_sim.urdf   decimated meshes in metres (engine)
  ar5_wuji_vis.urdf   full meshes in metres (rendering)
"""
import json
import os
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(HERE))
ASSET = osp.join(REPO, "Assets", "rokae_wuji")
SRC = osp.join(ASSET, "src")
ARM_URDF = osp.join(SRC, "rokae", "urdf", "AR5-5_08LR-W4C4A6-ZY2.urdf")
ARM = {"left": "AR5-5_08L-W4C4A6-ZY2", "right": "AR5-5_08R-W4C4A6-ZY2"}
HAND_ROLL = {"left": np.radians(60.0), "right": np.radians(-60.0)}
PALM_STANDOFF = 0.016          # Wuji docking plate -> palm (from the Wuji description)
ARM_FACES = 2500               # decimation target per arm link (engine meshes)
ARM_JOINTS = {s: [f"{ARM[s]}_joint_{i}" for i in range(1, 8)] for s in ("left", "right")}
FINGER_JOINTS = {s: [f"{s}_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)] for s in ("left", "right")}


def _T(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _origin_xml(T):
    xyz = T[:3, 3]
    rpy = Rot.from_matrix(T[:3, :3]).as_euler("xyz")
    return '<origin xyz="%.9g %.9g %.9g" rpy="%.9g %.9g %.9g"/>' % (*xyz, *rpy)


def _quat_T(p, wxyz):
    w, x, y, z = wxyz
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = p
    return T


def _mesh_out(src, dst, scale, faces, closed=False):
    """Copy a mesh in metres; faces > 0 decimates; closed=True makes it watertight (convex hull if it is not)
    -- the engine derives ABD mass/inertia from the closed surface (open meshes gave NaN / frozen solves)."""
    if osp.exists(dst):
        return
    m = trimesh.load(src, force="mesh", process=True)
    m.apply_scale(scale)
    m.merge_vertices()
    if faces and len(m.faces) > faces:
        import fast_simplification
        v, f = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
                                            target_count=faces)
        m = trimesh.Trimesh(v, f, process=True)
    if closed and not m.is_watertight:
        m = m.convex_hull
    os.makedirs(osp.dirname(dst), exist_ok=True)
    m.export(dst)


def _arm_part(sim, left_shift=(0.0, 0.0, 0.0)):
    """Arm links/joints of the source URDF, root = base_link (the world/waist/foot links are dropped)."""
    root = ET.parse(ARM_URDF).getroot()
    out, links = [], []
    kind = "sim" if sim else "vis"
    for l in root.findall("link"):
        name = l.get("name")
        if not name.startswith("AR5"):
            continue
        vis = l.find("visual")
        if vis is None:
            continue
        m = vis.find("geometry/mesh")
        fn = osp.basename(m.get("filename"))
        scale = float((m.get("scale") or "1 1 1").split()[0])
        dst_rel = f"meshes/{kind}/{osp.splitext(fn)[0]}.stl"
        _mesh_out(osp.join(SRC, "rokae", "meshes", fn), osp.join(ASSET, dst_rel), scale, ARM_FACES if sim else 0, closed=sim)
        o = vis.find("origin")
        T = _T([float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()]) if o is not None else np.eye(4)
        g = f'<geometry><mesh filename="{dst_rel}"/></geometry>'
        out.append(f'  <link name="{name}">\n    <visual>{_origin_xml(T)}{g}</visual>\n'
                   f'    <collision>{_origin_xml(T)}{g}</collision>\n  </link>')
        links.append(name)
    for j in root.findall("joint"):
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        if child not in links:
            continue
        o = j.find("origin")
        T = _T([float(v) for v in o.get("xyz").split()], [float(v) for v in (o.get("rpy") or "0 0 0").split()])
        if parent == "base_link" and "08L" in child:
            T[:3, 3] += np.asarray(left_shift, float)
        if j.get("type") == "fixed":
            out.append(f'  <joint name="{j.get("name")}" type="fixed">{_origin_xml(T)}'
                       f'<parent link="{parent}"/><child link="{child}"/></joint>')
        else:
            ax = j.find("axis").get("xyz")
            lim = j.find("limit")
            out.append(f'  <joint name="{j.get("name")}" type="revolute">{_origin_xml(T)}'
                       f'<parent link="{parent}"/><child link="{child}"/><axis xyz="{ax}"/>'
                       f'<limit lower="{lim.get("lower")}" upper="{lim.get("upper")}" effort="{lim.get("effort")}" '
                       f'velocity="{lim.get("velocity")}"/></joint>')
    # flange: fixed 0.09 m beyond link7 (no mesh -> folded into the hand mount)
    flange = {}
    for j in root.findall("joint"):
        if j.get("name").endswith("flan_joint"):
            o = j.find("origin")
            flange[j.find("parent").get("link")] = _T([float(v) for v in o.get("xyz").split()],
                                                       [float(v) for v in (o.get("rpy") or "0 0 0").split()])
    return out, flange


def _hand_part(side, sim, flange_T):
    J = json.load(open(osp.join(SRC, "wuji_usd_joints.json")))
    kind = "sim" if sim else "vis"
    out, links = [], set()
    # mount: link7 -> flange (+0.09 z) -> roll about z -> docking -> palm (+16 mm)
    Tm = flange_T @ _T(rpy=(0, 0, HAND_ROLL[side])) @ _T(xyz=(0, 0, PALM_STANDOFF))
    out.append(f'  <joint name="{side}_hand_mount" type="fixed">{_origin_xml(Tm)}'
               f'<parent link="{ARM[side]}_link7"/><child link="{side}_palm_link"/></joint>')
    links.add(f"{side}_palm_link")
    for j in J:
        if not j["name"].startswith(side + "_finger"):
            continue
        T0, T1 = _quat_T(j["p0"], j["r0"]), _quat_T(j["p1"], j["r1"])
        origin = T0 @ np.linalg.inv(T1)
        links.update([j["parent"], j["child"]])
        if j["type"] == "PhysicsRevoluteJoint":
            ax = T1[:3, :3] @ np.array({"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[j["axis"]], float)
            out.append(f'  <joint name="{j["name"]}" type="revolute">{_origin_xml(origin)}'
                       f'<parent link="{j["parent"]}"/><child link="{j["child"]}"/>'
                       f'<axis xyz="{ax[0]:.9g} {ax[1]:.9g} {ax[2]:.9g}"/>'
                       f'<limit lower="{np.radians(j["lo"]):.6f}" upper="{np.radians(j["hi"]):.6f}" effort="5" velocity="6"/></joint>')
        else:
            out.append(f'  <joint name="{j["name"]}" type="fixed">{_origin_xml(origin)}'
                       f'<parent link="{j["parent"]}"/><child link="{j["child"]}"/></joint>')
    for l in sorted(links):
        base = l[len(side) + 1:]
        # meshes in the link frames of the joint description (exported from the Wuji USD; the left-hand STLs of
        # the Wuji mesh package use a different, axis-permuted frame)
        src = osp.join(SRC, "wuji_usd", side, base + ("_collision.stl" if sim else ".stl"))
        if not osp.exists(src):
            src = osp.join(SRC, "wuji_usd", side, base + ".stl")
        dst_rel = f"meshes/{kind}/wuji_{side}_{base}.stl"
        _mesh_out(src, osp.join(ASSET, dst_rel), 1.0, 600 if sim else 0, closed=sim)
        g = f'<geometry><mesh filename="{dst_rel}"/></geometry>'
        out.append(f'  <link name="{l}">\n    <visual><origin xyz="0 0 0" rpy="0 0 0"/>{g}</visual>\n'
                   f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>{g}</collision>\n  </link>')
    return out


def build(left_shift=(0.0, 0.0, 0.0), force=False):
    """Write the sim / vis URDFs (meshes are cached; the URDF text is always rewritten).
    left_shift: translation of the LEFT arm's mount in the robot base frame (m) -- the calibration
    offset that puts the left hand's recorded grasp on the beaker the right hand dripped into."""
    paths = {}
    for sim in (True, False):
        path = osp.join(ASSET, "ar5_wuji_sim.urdf" if sim else "ar5_wuji_vis.urdf")
        paths["sim" if sim else "vis"] = path
        arm, flange = _arm_part(sim, left_shift)
        # root: a small column block under the arm mounts (base_link of the source URDF)
        kind = "sim" if sim else "vis"
        col = osp.join(ASSET, f"meshes/{kind}/column.stl")
        if not osp.exists(col):
            os.makedirs(osp.dirname(col), exist_ok=True)
            trimesh.creation.box((0.12, 0.12, 0.10), _T(xyz=(-0.10, 0, 1.0))).export(col)
        parts = ['<?xml version="1.0"?>', '<robot name="ar5_wuji_dual">',
                 f'  <link name="base_link"><visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="meshes/{kind}/column.stl"/></geometry></visual>'
                 f'<collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="meshes/{kind}/column.stl"/></geometry></collision></link>']
        parts += arm
        for side in ("left", "right"):
            parts += _hand_part(side, sim, flange[f"{ARM[side]}_link7"])
        parts.append("</robot>")
        open(path, "w").write("\n".join(parts) + "\n")
    return paths


# ------------------------------------------------------------------ forward kinematics (numpy)

class URDFKin:
    """Fixed + revolute forward kinematics of a URDF (no engine)."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.child_of, self.links, self.meshes = {}, [], {}
        for l in root.findall("link"):
            self.links.append(l.get("name"))
            v = l.find("visual")
            if v is not None and v.find("geometry/mesh") is not None:
                o = v.find("origin")
                self.meshes[l.get("name")] = (osp.join(osp.dirname(path), v.find("geometry/mesh").get("filename")),
                                              _T([float(x) for x in o.get("xyz").split()], [float(x) for x in o.get("rpy").split()]))
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            ax = j.find("axis")
            d = dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                     child=j.find("child").get("link"),
                     T=_T([float(x) for x in o.get("xyz").split()], [float(x) for x in (o.get("rpy") or "0 0 0").split()]),
                     axis=np.array([float(x) for x in ax.get("xyz").split()]) if ax is not None else None)
            lim = j.find("limit")
            if lim is not None:
                d["lim"] = (float(lim.get("lower")), float(lim.get("upper")))
            self.joints[d["name"]] = d
            self.child_of[d["child"]] = d
        self.root = [l for l in self.links if l not in self.child_of][0]

    def fk(self, q, base=np.eye(4)):
        out = {self.root: base}

        def get(link):
            if link in out:
                return out[link]
            j = self.child_of[link]
            T = get(j["parent"]) @ j["T"]
            if j["type"] == "revolute":
                R = np.eye(4)
                R[:3, :3] = Rot.from_rotvec(j["axis"] * q.get(j["name"], 0.0)).as_matrix()
                T = T @ R
            out[link] = T
            return T

        for l in self.links:
            get(l)
        return out


if __name__ == "__main__":
    import sys
    p = build()
    for k, v in p.items():
        k_ = URDFKin(v)
        nrev = sum(1 for j in k_.joints.values() if j["type"] == "revolute")
        print(k, v, f"{len(k_.links)} links, {nrev} revolute joints")
