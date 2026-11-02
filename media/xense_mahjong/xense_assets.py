"""Assets for the Xense-gripper mahjong pinch: gripper geometry from the URDF, gel pads, the tile.

Everything here is geometry only (no engine, no GPU):

* ``XenseGripper`` reads ``xense_gripper_with_mount_camera.urdf`` (the vendor package unpacked under
  ``Assets/xense_gripper/xense_gripper_model``), runs its forward kinematics and groups the links into the three
  bodies that move independently in the simulation: the HAND (tool adapter + mount plate + gripper body + wrist
  D435i shell, rigidly fixed to each other in the URDF) and the two FINGERS (prismatic ``slider_left`` /
  ``slider_right``, ``slider_right = -slider_left``).  The full-resolution meshes are kept for rendering; the
  simulation gets decimated copies (the gripper body alone has 249k triangles and never touches anything; the
  fingers carry the gel pads, which are what touches the tile).
* ``fabricate_pad`` tetrahedralises the gel pad that sits on the finger's sensing window (the flat 16.6 x 28.4 mm
  inner face of each Xense finger, measured from the STL).
* ``tile`` builds the mahjong tile with the engraved character (``stiff_physics.sensors.mahjong``) and caches it,
  together with the ground-truth glyph mask, so the solver machine needs no CJK font.

Frames.  World is Y-up, the table is the ground plane y = 0.  The gripper points DOWN: gripper-body (URDF
``gripper_base_link``) +z -> world -y, +y (the left finger side, closing axis) -> world +z, +x -> world +x.
So the fingers close along world z and the tile stands upright with its engraved face towards +z (the left
finger).
"""

from __future__ import annotations

import os
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
ASSET_DIR = osp.join(REPO, "Assets", "xense_gripper")
PKG_DIR = osp.join(ASSET_DIR, "xense_gripper_model")
URDF = osp.join(PKG_DIR, "xense_gripper_with_mount_camera.urdf")
CACHE_DIR = osp.join(ASSET_DIR, "sim_cache")


def _ensure_sensors():
    """The tactile layer (``stiff_physics.sensors``, pure Python, ported from Taccel) is not in the public
    ``stiff-physics`` wheel yet.  If the installed package lacks it, serve the copy shipped with this example
    (``_vendor/sensors``) under the same import name."""
    try:
        import stiff_physics.sensors  # noqa: F401
        return
    except ImportError:
        pass
    import stiff_physics
    stiff_physics.__path__.append(osp.join(HERE, "_vendor"))
    import stiff_physics.sensors  # noqa: F401


_ensure_sensors()

HAND_LINKS = ("tool_adapter_link", "gripper_mount_plate", "gripper_base_link", "wrist_d435i_link")
FINGERS = {"left": "finger_left", "right": "finger_right"}
JOINTS = {"left": "slider_left", "right": "slider_right"}

# gripper body frame -> world (columns: images of base x, y, z)
R_WORLD_BASE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# ---- gel pad (pad-local frame: coat at z = 0 facing the object, body towards +z) -------------------------
PIXEL_MM = 0.079375                          # Taccel camera pitch
PAD_CAM = (302, 202)                         # px: height (along the finger), width (across it)
PAD_T = 2.5e-3                               # gel thickness
PAD_SIZE = (PAD_CAM[0] * PIXEL_MM * 1e-3, PAD_CAM[1] * PIXEL_MM * 1e-3, PAD_T)   # 23.97 x 16.03 x 2.5 mm
PAD_MAX_VOLUME = 1.0e-10                     # tetgen -a (m^3); the stroke edges need this (Allegro mahjong ablation)
PAD_SEG_MM = 0.5                             # surface subdivision of the pad box before tetgen
PAD_Z_ON_FINGER = 69.0e-3                    # pad centre along gripper-body z (sensing window spans 53.6..82.0 mm)

# ---- tile -------------------------------------------------------------------------------------------------
TILE = dict(width_mm=20.0, height_mm=26.0, thick_mm=12.0, engrave_mm=0.9, glyph_mm=13.0, grid=121)


def _fl(s, d="0 0 0"):
    return np.array([float(v) for v in (s or d).split()])


def rpy_T(xyz, rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def T_of(R=None, p=(0, 0, 0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


class XenseGripper:
    """URDF geometry + FK of the Xense gripper with mount plate and wrist camera."""

    def __init__(self, urdf: str = URDF):
        import trimesh
        self.urdf = urdf
        root = ET.parse(urdf).getroot()
        self.links = {l.get("name"): l for l in root.findall("link")}
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        children = {j.find("child").get("link") for j in self.joints.values()}
        self.root = [n for n in self.links if n not in children][0]
        self.meshes = {}                  # link -> trimesh in the LINK frame (visual origin applied)
        self.colors = {}
        for n, l in self.links.items():
            v = l.find("visual")
            if v is None or v.find("geometry/mesh") is None:
                continue
            o = v.find("origin")
            To = rpy_T(_fl(o.get("xyz") if o is not None else None), _fl(o.get("rpy") if o is not None else None))
            m = trimesh.load(osp.join(osp.dirname(urdf), v.find("geometry/mesh").get("filename")), force="mesh")
            m.apply_transform(To)
            self.meshes[n] = m
            c = v.find("material/color")
            self.colors[n] = _fl(c.get("rgba"))[:3] if c is not None else np.array([0.3, 0.3, 0.32])
        j = self.joints["slider_left"]
        lim = j.find("limit")
        self.q_max = float(lim.get("upper"))

    def fk(self, q: float = 0.0) -> dict:
        """Link transforms in the URDF root frame at slider_left = q (slider_right = -q via the mimic)."""
        T = {self.root: np.eye(4)}
        todo = list(self.joints.values())
        while todo:
            for j in list(todo):
                p = j.find("parent").get("link")
                if p not in T:
                    continue
                o = j.find("origin")
                Tj = rpy_T(_fl(o.get("xyz") if o is not None else None), _fl(o.get("rpy") if o is not None else None))
                if j.get("type") == "prismatic":
                    qv = q
                    m = j.find("mimic")
                    if m is not None:
                        qv = float(m.get("multiplier", 1)) * q + float(m.get("offset", 0))
                    Tj = Tj @ T_of(p=_fl(j.find("axis").get("xyz")) * qv)
                T[j.find("child").get("link")] = T[p] @ Tj
                todo.remove(j)
        return T

    def world_fk(self, base_origin_world, q: float = 0.0) -> dict:
        """Link transforms in the WORLD for the downward-pointing gripper whose gripper-body origin is at
        ``base_origin_world``."""
        T = self.fk(q)
        W_base = T_of(R_WORLD_BASE, base_origin_world)
        W_root = W_base @ np.linalg.inv(T["gripper_base_link"])
        return {n: W_root @ t for n, t in T.items()}

    # --------------------------------------------------------------- mass properties
    def inertial(self, link: str):
        """(mass, com in the link frame, 3x3 inertia about the com in the LINK frame) from the URDF <inertial>."""
        e = self.links[link].find("inertial")
        if e is None:
            return 0.0, np.zeros(3), np.zeros((3, 3))
        o = e.find("origin")
        To = rpy_T(_fl(o.get("xyz") if o is not None else None), _fl(o.get("rpy") if o is not None else None))
        a = {k: float(v) for k, v in e.find("inertia").attrib.items()}
        I = np.array([[a["ixx"], a.get("ixy", 0), a.get("ixz", 0)], [a.get("ixy", 0), a["iyy"], a.get("iyz", 0)],
                      [a.get("ixz", 0), a.get("iyz", 0), a["izz"]]])
        R = To[:3, :3]
        return float(e.find("mass").get("value")), To[:3, 3], R @ I @ R.T

    def group_inertia(self, links, link_W: dict):
        """Combined (mass, world com, world inertia about the com) of rigidly joined links at world poses link_W."""
        parts = []
        for n in links:
            m, c, I = self.inertial(n)
            if m <= 0:
                continue
            W = link_W[n]
            parts.append((m, W[:3, :3] @ c + W[:3, 3], W[:3, :3] @ I @ W[:3, :3].T))
        M = sum(p[0] for p in parts)
        C = sum(p[0] * p[1] for p in parts) / M
        I = np.zeros((3, 3))
        for m, c, Ii in parts:
            d = c - C
            I += Ii + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        return M, C, I

    # --------------------------------------------------------------- measured geometry
    def window(self, side: str) -> dict:
        """The finger's flat sensing window, in the gripper-body frame at q = 0: inward face coordinate y
        (left finger: smallest y of the window, right: largest), x / z extent."""
        T = self.fk(0.0)
        Tb_inv = np.linalg.inv(T["gripper_base_link"])
        m = self.meshes[FINGERS[side]]
        Tf = Tb_inv @ T[FINGERS[side]]
        fc = m.triangles_center @ Tf[:3, :3].T + Tf[:3, 3]
        fn = m.face_normals @ Tf[:3, :3].T
        inward = -1.0 if side == "left" else 1.0
        sel = (fn[:, 1] * inward > 0.98) & (fc[:, 2] > 0.050)
        # the window is the most inward large flat patch
        ys = fc[sel, 1]
        y_in = ys.min() if side == "left" else ys.max()
        on = sel & (np.abs(fc[:, 1] - y_in) < 1e-4)
        return dict(y=float(y_in), x=(float(fc[on, 0].min()), float(fc[on, 0].max())),
                    z=(float(fc[on, 2].min()), float(fc[on, 2].max())),
                    area_mm2=float(m.area_faces[on].sum() * 1e6))

    def tip_z(self) -> float:
        """Largest gripper-body z of either finger (the fingertip end) at q = 0."""
        T = self.fk(0.0)
        Tb_inv = np.linalg.inv(T["gripper_base_link"])
        z = []
        for link in FINGERS.values():
            Tf = Tb_inv @ T[link]
            z.append((self.meshes[link].vertices @ Tf[:3, :3].T + Tf[:3, 3])[:, 2].max())
        return float(max(z))

    def pad_frame_in_finger(self, side: str) -> np.ndarray:
        """Pad-local -> FINGER-link frame.  Pad x = along the finger (gripper-body z), pad z = from the coat into
        the finger (left: body +y, right: body -y); the stick face (z = PAD_T) lies on the sensing window."""
        T = self.fk(0.0)
        w = self.window(side)
        ez, ex, ey = np.eye(3)[2], np.eye(3)[0], np.eye(3)[1]
        if side == "left":
            R = np.column_stack([ez, ex, ey])
            p = [0.5 * sum(w["x"]), w["y"] - PAD_T, PAD_Z_ON_FINGER]
        else:
            R = np.column_stack([ez, -ex, -ey])
            p = [0.5 * sum(w["x"]), w["y"] + PAD_T, PAD_Z_ON_FINGER]
        T_base_pad = T_of(R, p)
        T_base_finger = np.linalg.inv(T["gripper_base_link"]) @ T[FINGERS[side]]
        return np.linalg.inv(T_base_finger) @ T_base_pad

    # --------------------------------------------------------------- body meshes
    def body_meshes(self, sim: bool = True) -> dict:
        """{'hand' | 'left' | 'right': trimesh in the URDF ROOT frame at q = 0}.  sim=True -> decimated, cached."""
        import trimesh
        T = self.fk(0.0)
        groups = {"hand": HAND_LINKS, "left": (FINGERS["left"],), "right": (FINGERS["right"],)}
        target = {"hand": 12000, "left": 6000, "right": 6000}
        out = {}
        for g, links in groups.items():
            path = osp.join(CACHE_DIR, f"{g}_{target[g]}.npz")
            if sim and osp.exists(path):
                d = np.load(path)
                out[g] = trimesh.Trimesh(d["v"], d["f"], process=False)
                continue
            parts = []
            for n in links:
                m = self.meshes[n].copy()
                m.apply_transform(T[n])
                parts.append(m)
            m = trimesh.util.concatenate(parts)
            m.merge_vertices()
            if sim:
                m = _decimate(m, target[g])
                os.makedirs(CACHE_DIR, exist_ok=True)
                np.savez_compressed(path, v=np.asarray(m.vertices), f=np.asarray(m.faces, np.int32))
            out[g] = m
        return out


def _decimate(m, n_faces: int):
    import fast_simplification
    import trimesh
    if len(m.faces) <= n_faces:
        return m
    v, f = fast_simplification.simplify(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64),
                                        target_reduction=1.0 - n_faces / len(m.faces))
    out = trimesh.Trimesh(v, f, process=True)
    out.remove_unreferenced_vertices()
    return out


# ------------------------------------------------------------------------------------------- gel pad
def fabricate_pad(force: bool = False) -> str:
    """Tetrahedralise the pad once (tetgen); returns the Taccel-style fabrication json."""
    from stiff_physics.sensors.fabricate import FabricationSpec, fabricate, write_fabrication_json
    from stiff_physics.sensors.linkerhand import pad_box_stl
    pad_dir = osp.join(ASSET_DIR, "pad")
    j = osp.join(pad_dir, f"tac_fabr_{PAD_MAX_VOLUME:g}.json")
    if osp.exists(j) and not force:
        return j
    os.makedirs(osp.join(pad_dir, "meshes"), exist_ok=True)
    stl = pad_box_stl(osp.join(pad_dir, "meshes", "xense_pad.stl"), size=PAD_SIZE, seg_mm=PAD_SEG_MM)
    spec = FabricationSpec(stl_path=stl, max_volume=PAD_MAX_VOLUME, coat_z_max=1e-4,
                           stick_z_min=PAD_SIZE[2] - 1e-4, marker_extent_mm=6.5,
                           marker_grid=7, out_dir=osp.join(pad_dir, "meshes"), name="pad")
    fabricate(spec, verbose=True)
    write_fabrication_json(j, "finger", f"meshes/pad_maxv={PAD_MAX_VOLUME:g}.vtk", PAD_MAX_VOLUME)
    return j


GEL_E = 2.0e5                                # Pa.  VBTS silicones are ~0.1-0.3 MPa; Taccel's default 0.5 MPa pressed
                                             # flat over the whole 16 x 24 mm face is ~200 N/mm (confined layer), so
                                             # a 10 N grip would indent it only ~0.05 mm.


def load_pad_asset(E: float = GEL_E):
    from stiff_physics.sensors import CameraConfig, SensorAsset
    from stiff_physics.sensors.sensor_asset import MaterialConfig
    j = fabricate_pad()
    cam = CameraConfig(height=PAD_CAM[0], width=PAD_CAM[1], max_depth=PAD_T)
    return SensorAsset.from_taccel_fabrication(j, 0, camera=cam, material=MaterialConfig(E=E))


# ------------------------------------------------------------------------------------------- tile
def tile(char: str = "中", force: bool = False) -> dict:
    """Engraved tile (metres, centred, engraved face +z, upright = height along y) + ground-truth glyph mask
    sampled on the tile face (mask[i, j]: i along x = width, j along y = height)."""
    path = osp.join(ASSET_DIR, f"tile_{ord(char):x}.npz")
    if osp.exists(path) and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    from stiff_physics.sensors.mahjong import tile_mesh, tile_mesh_check
    v, f = tile_mesh(char=char, **TILE)
    chk = tile_mesh_check(v, f, TILE["width_mm"], TILE["height_mm"], TILE["thick_mm"], TILE["engrave_mm"])
    assert chk["closed"], chk
    n = TILE["grid"]
    front = v[: n * n].reshape(n, n, 3)                     # height field over the engraved face
    mask = front[:, :, 2] < (TILE["thick_mm"] / 2 - 0.5 * TILE["engrave_mm"]) * 1e-3
    out = dict(verts=v, faces=f, mask=mask, gx=front[:, 0, 0], gy=front[0, :, 1],
               char=np.array(char), engraved_mm3=np.array(chk["engraved_mm3"]))
    np.savez_compressed(path, **out)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="build / check the cached assets")
    ap.add_argument("--char", default="中")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    g = XenseGripper()
    for s in ("left", "right"):
        w = g.window(s)
        print(f"{s} window: y {w['y']*1e3:.2f} mm, x {w['x'][0]*1e3:.1f}..{w['x'][1]*1e3:.1f}, "
              f"z {w['z'][0]*1e3:.1f}..{w['z'][1]*1e3:.1f} mm, {w['area_mm2']:.0f} mm2")
    print(f"fingertip z {g.tip_z()*1e3:.1f} mm, slider range 0..{g.q_max*1e3:.0f} mm")
    for k, m in g.body_meshes(sim=True).items():
        print(f"sim mesh {k}: {len(m.vertices)} v / {len(m.faces)} f, extent {np.round(m.extents*1e3, 1)} mm")
    t = tile(a.char, force=a.force)
    print(f"tile {str(t['char'])}: {len(t['verts'])} v / {len(t['faces'])} f, glyph {t['mask'].sum()} grid cells, "
          f"engraved {float(t['engraved_mm3']):.1f} mm3")
    pa = load_pad_asset()
    print(pa.summary())
