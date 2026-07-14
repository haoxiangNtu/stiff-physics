#!/usr/bin/env python3
"""Stage-2 physics scene: dual arm+hands, drawer cabinet, FEM doll (Z-up, world coords).

Layout (robot stands at waist (2.331,0.036,1.050) facing +X, yaw -5.8° baked in URDF):
  * cabinet: fixed ABD box assembly, top at CAB_TOP, front face toward the robot
  * drawer:  free ABD (open-top box + front panel + handle bar), prismatic joint
             along -facing (pulls out toward the robot), gravity off (rail mode)
  * doll:    FEM, resting on the cabinet top in the right hand's descent path

Collision whitelist: doll<->(GRIP_r|GRIP_l|cabinet|drawer|ground),
                     drawer<->(GRIP_l|GRIP_r). Everything else excluded.
build() returns a dict with engine + handles; used by the keyframe runner and benches.
"""
import sys, os
import numpy as np
os.environ.setdefault("STIFF_COLL_BUF_ACTIVE", "1")
os.environ.setdefault("STIFF_SKIP_CCD_SANITY", "1")
os.environ.setdefault("BVHSKIP2", "1")
sys.path.insert(0, "/home/ps/Downloads/FD-light/fd_walk")
sys.path.insert(0, "/home/ps/Downloads/Stiff-GIPC-v08")
sys.path.insert(0, "/home/ps/Downloads/Stiff-GIPC-v08/examples")
sys.path.insert(0, "/home/ps/Downloads/geo")
for _f in list(sys.meta_path):
    if 'ScikitBuild' in type(_f).__name__:
        sys.meta_path.remove(_f)
import trimesh
from stiff_physics.engine import Engine, Config
from tetify_toys import tetify

D = "/home/ps/Downloads/FD-light/fd-urdf-full/FD-URDF"
URDF = os.path.join(D, "fd_dual_arm.urdf")
WAIST = np.array([2.3313, 0.0359, 1.0500])
YAW = -0.101823
FACING = np.array([np.cos(YAW), np.sin(YAW), 0.0])       # robot forward (+X-ish)
LEFTY  = np.array([-np.sin(YAW), np.cos(YAW), 0.0])      # robot left (+Y-ish)

# ---- layout ------------------------------------------------------------------
CAB_TOP   = 0.95
CAB_FRONT = float(os.environ.get("FD_CAB_FRONT", "0.32"))   # waist -> cabinet front face
CAB_DEPTH = 0.40
CAB_WIDTH = 0.90
CAB_V2 = os.environ.get("FD_CAB_V2", "0") == "1"
CAB_MESH = os.environ.get("FD_CAB_MESH")                       # v3: user OBJ (two drawers, world coords)
DRAWER_W, DRAWER_D, DRAWER_H = (0.62 if CAB_V2 else 0.42), 0.30, 0.14
DRAWER_SILL = 0.76        # drawer floor height
DRAWER_MAX = 0.24         # pull-out distance
HANDLE_Z = DRAWER_SILL + (0.08 if CAB_V2 else 0.105)   # v2: 面板几何中心; v1: 左手IK可达
DRAWER_LAT = 0.0 if CAB_V2 else -0.10                  # v2: 抽屉居中(对称)
_ds=os.environ.get("FD_DOLL_SPOT")
DOLL_SPOT = (np.array([float(x) for x in _ds.split(",")]) if _ds
             else np.array([2.7200, 0.2600, 0.0]))           # 10x玩偶: 再向抽屉前缘移6cm
TSCALE = float(os.environ.get("FD_TSCALE", "0.7"))
YOUNG  = float(os.environ.get("FD_YOUNG", "5e5"))

FING_R = [f + s for f in ("if_", "mf_", "rf_", "lf_") for s in ("proximal_link_r", "distal_link_r")]
FING_L = [f + s for f in ("if_", "mf_", "rf_", "lf_") for s in ("proximal_link_l", "distal_link_l")]
THUMB_R = ["th_root_link_r", "th_proximal_link_r", "th_distal_link_r"]
THUMB_L = ["th_root_link_l", "th_proximal_link_l", "th_distal_link_l"]
GRIP_R = set(["base_link_r"] + FING_R + THUMB_R[1:])
GRIP_L = set(["base_link_l"] + FING_L + THUMB_L[1:])
ARMJ_R = [f"arm_r{i}_joint" for i in range(1, 8)]
ARMJ_L = [f"arm_l{i}_joint" for i in range(1, 8)]


def _handle_specs(front_pt, z):
    """centered handle: bar 18cm + two stems, 6.5cm off the given front plane."""
    return [
        (np.r_[(front_pt - FACING * 0.065)[:2], z], [0.022, 0.18, 0.022]),
        (np.r_[(front_pt - FACING * 0.030 + LEFTY * 0.08)[:2], z], [0.062, 0.016, 0.016]),
        (np.r_[(front_pt - FACING * 0.030 - LEFTY * 0.08)[:2], z], [0.062, 0.016, 0.016]),
    ]


def _boxes(specs):
    """specs: list of (center(3), extents(3)) -> merged trimesh."""
    ms = []
    for c, e in specs:
        b = trimesh.creation.box(extents=e); b.apply_translation(c); ms.append(b)
    return trimesh.util.concatenate(ms)


def build(ncap=100, log0=True):
    eng = Engine(Config(gravity=(0, 0, -9.8), dt=0.01, ground_normal=(0, 0, 1),
                        ground_offset=0.0, newton_iter_cap=ncap,
                        velocity_damping=0.3, friction_rate=1.0,
                        density=float(os.environ.get("FD_DENSITY", "1000")),
                        joint_strength_ratio=float(os.environ.get("FD_JSR", "100")),
                        revolute_driving_strength_ratio=float(os.environ.get("FD_RDSR", "100")),
                        max_prismatic_step_per_frame=0.005))
    _ip = os.environ.get("FD_INIT_POSE")
    _ja0 = {}
    if _ip and os.path.exists(_ip):
        import json as _json
        _ja0 = {k: float(v) for k, v in _json.load(open(_ip)).items()}
    eng.load_urdf(os.path.abspath(URDF), translation=tuple(WAIST),
                  root_fixed=True, revolute_as_motor=True, default_young=1e7,
                  initial_joint_angles=_ja0)
    recs = {r.label: r for r in eng.get_load_records()}

    # ---- cabinet (fixed): top slab + sides + back + plinth, cavity at front ----
    fc = WAIST + FACING * CAB_FRONT                       # front-face center (xy)
    if CAB_MESH:
        return _build_cab_v3(eng, recs)
    
    cc = fc + FACING * (CAB_DEPTH / 2)                    # cabinet center (xy)
    t = 0.02                                              # wall thickness
    cav_top = DRAWER_SILL + DRAWER_H + 0.03               # cavity ceiling
    cab_specs = [
        (np.r_[cc[:2], CAB_TOP - t/2],                [CAB_DEPTH, CAB_WIDTH, t]),          # top slab
        (np.r_[cc[:2], (DRAWER_SILL - 0.02) / 2],     [CAB_DEPTH, CAB_WIDTH, DRAWER_SILL - 0.02]),  # plinth below drawer
        (np.r_[(cc + LEFTY * (CAB_WIDTH/2 - t/2))[:2], ((CAB_TOP - t) if CAB_V2 else CAB_TOP)/2], [CAB_DEPTH, t, (CAB_TOP - t) if CAB_V2 else CAB_TOP]),  # left wall
        (np.r_[(cc - LEFTY * (CAB_WIDTH/2 - t/2))[:2], ((CAB_TOP - t) if CAB_V2 else CAB_TOP)/2], [CAB_DEPTH, t, (CAB_TOP - t) if CAB_V2 else CAB_TOP]),  # right wall
        (np.r_[(cc + FACING * (CAB_DEPTH/2 - t/2))[:2], ((CAB_TOP - t) if CAB_V2 else CAB_TOP)/2], [t, CAB_WIDTH, (CAB_TOP - t) if CAB_V2 else CAB_TOP]), # back wall
    ]
    if CAB_V2:
        # symmetric facade: piers close the bay slot outside the (centered) drawer;
        # lower bay = closed drawer, render-only (proud panel + centered handle, fixed)
        BAY_HALF = DRAWER_W / 2 + 0.006
        pier_w = CAB_WIDTH / 2 - BAY_HALF
        pier_zc, pier_h = (0.74 + cav_top) / 2, cav_top - 0.74
        for sgn in (+1, -1):
            cab_specs.append((np.r_[(fc + FACING * (t/2) + LEFTY * sgn * (BAY_HALF + pier_w/2))[:2], pier_zc],
                              [t, pier_w, pier_h]))
        P2_ZC, P2_H = 0.635, 0.17                     # lower drawer face: z 0.55..0.72, centered
        cab_specs.append((np.r_[(fc + FACING * (0.028/2 - 0.008))[:2], P2_ZC], [0.028, DRAWER_W, P2_H]))
        for spec in _handle_specs(fc - FACING * 0.008, P2_ZC):
            cab_specs.append(spec)
    else:
        cab_specs.append((np.r_[cc[:2], (cav_top + CAB_TOP - t) / 2],
                          [CAB_DEPTH, CAB_WIDTH, CAB_TOP - t - cav_top]))  # band above cavity
    cab = _boxes(cab_specs)
    eng.load_mesh_from_data(np.asarray(cab.vertices), np.asarray(cab.faces),
                            verts_per_face=3, dimensions=3, body_type="ABD",
                            transform=np.eye(4), boundary_type="Fixed", young_modulus=1e8)
    cab_id = eng.get_load_records()[-1].body_offset

    # ---- drawer (free ABD + prismatic): open-top box + front panel + handle ----
    # 初始【半开】: 把手悬空可勾, 左手不必穿柜体。关节零点=半开, 限位 ±HALF
    HALF = DRAWER_MAX / 2
    dc = (fc + FACING * (DRAWER_D / 2) - FACING * HALF
          + LEFTY * DRAWER_LAT)                           # v2: 居中; v1: 右移10cm缩短运送
    dw_specs = [
        (np.r_[dc[:2], DRAWER_SILL + t/2],            [DRAWER_D, DRAWER_W, t]),            # floor
        (np.r_[(dc + LEFTY * (DRAWER_W/2 - t/2))[:2], DRAWER_SILL + DRAWER_H/2], [DRAWER_D, t, DRAWER_H]),
        (np.r_[(dc - LEFTY * (DRAWER_W/2 - t/2))[:2], DRAWER_SILL + DRAWER_H/2], [DRAWER_D, t, DRAWER_H]),
        (np.r_[(dc + FACING * (DRAWER_D/2 - t/2))[:2], DRAWER_SILL + DRAWER_H/2], [t, DRAWER_W, DRAWER_H]),  # back
        (np.r_[(dc - FACING * (DRAWER_D/2 - t/2))[:2], DRAWER_SILL + DRAWER_H/2], [t, DRAWER_W, DRAWER_H + 0.02]),  # front panel
    ]
    if CAB_V2:
        dw_specs += _handle_specs(dc - FACING * (DRAWER_D/2), HANDLE_Z)   # centered handle
    else:
        dw_specs += [
            # handle: EXTENDED bar (6.5cm off the panel, 5.4cm clear gap for fingers)
            (np.r_[(dc - FACING * (DRAWER_D/2 + 0.065) + LEFTY*0.04)[:2], HANDLE_Z], [0.022, 0.18, 0.022]),
            (np.r_[(dc - FACING * (DRAWER_D/2 + 0.030) + LEFTY*0.12)[:2], HANDLE_Z], [0.062, 0.016, 0.016]),
            (np.r_[(dc - FACING * (DRAWER_D/2 + 0.030) - LEFTY*0.04)[:2], HANDLE_Z], [0.062, 0.016, 0.016]),
        ]
    dw = _boxes(dw_specs)
    eng.load_mesh_from_data(np.asarray(dw.vertices), np.asarray(dw.faces),
                            verts_per_face=3, dimensions=3, body_type="ABD",
                            transform=np.eye(4), boundary_type="Free", young_modulus=1e8)
    drw_rec = eng.get_load_records()[-1]
    drw_id = drw_rec.body_offset
    eng.native.set_abd_body_density(drw_id, 300.0)

    # ---- doll (FEM) on the cabinet top ----
    verts, cells = tetify(os.environ.get("FD_TOY", "/home/ps/Downloads/geo/Toy00.obj")); verts = verts * TSCALE
    print(f"[scene] TSCALE={TSCALE} env={os.environ.get('FD_TSCALE')!r} "
          f"doll_bbox={np.round(verts.max(0)-verts.min(0),3)} density={eng if False else os.environ.get('FD_DENSITY')!r}",
          flush=True)
    lo, hi = verts.min(0), verts.max(0); ctr = (lo + hi) / 2
    Tt = np.eye(4)
    Tt[:3, 3] = [DOLL_SPOT[0] - ctr[0], DOLL_SPOT[1] - ctr[1], CAB_TOP - lo[2] + 0.003]
    eng.load_mesh_from_data(verts, cells, verts_per_face=4, dimensions=3, body_type="FEM",
                            transform=Tt, young_modulus=YOUNG, boundary_type="Free")
    trec = eng.get_load_records()[-1]
    ta, tb = trec.vertex_offset, trec.vertex_offset + trec.vertex_count
    n_abd = eng.native.get_abd_body_count()
    doll_gid = n_abd + trec.body_offset

    # ---- prismatic joint: cabinet <-> drawer, axis -FACING (pull toward robot) ----
    anchor = np.r_[dc[:2], DRAWER_SILL + DRAWER_H / 2]
    pj = eng.native.add_prismatic_joint(cab_id, drw_id, np.asarray(anchor, float),
                                        (-FACING).astype(float), -HALF, HALF, "drawer")
    eng.native.set_body_apply_gravity(drw_id, False)      # rail mode: no sag

    # ---- collision whitelist ----
    gripR = {recs[l].body_offset for l in GRIP_R}
    gripL = {recs[l].body_offset for l in GRIP_L}
    allids = list(range(n_abd + eng.native.get_fem_body_count()))
    allow = set()
    for g in gripR | gripL:
        allow.add(frozenset((g, doll_gid)))
    for g in gripL:                       # 只有左手与抽屉接触(右手全程不该碰它)
        allow.add(frozenset((g, drw_id)))
    allow |= {frozenset((doll_gid, cab_id)), frozenset((doll_gid, drw_id))}
    for g in gripR:                       # 右手 <-> 柜体/桌面: 真碰撞(抓取时可依托桌面)
        allow.add(frozenset((g, cab_id)))
    for i in range(len(allids)):
        for j in range(i + 1, len(allids)):
            if frozenset((allids[i], allids[j])) not in allow:
                eng.add_collision_exclusion(allids[i], allids[j])
    for b in allids:
        if b != doll_gid:
            eng.add_ground_collision_skip(b)
    eng.finalize()
    if os.environ.get("FD_DRAWER_PASSIVE", "1") == "1":
        try:
            eng.native.set_prismatic_strength(pj, 0.0)    # 完全自由滑轨: 无任何回位弹簧
        except Exception:
            pass
    if log0:
        eng.native.set_log_level(0)
    rj = {eng.native.get_revolute_joint_info(i).name: i
          for i in range(eng.native.get_num_revolute_joints())}
    fstr = float(os.environ.get("FD_FSTR", "1"))
    if fstr != 1.0:
        for fn in FING_R + THUMB_R:
            if fn in rj:
                try: eng.native.set_revolute_strength(rj[fn], fstr)
                except Exception: pass
    astr = float(os.environ.get("FD_ASTR", "1"))               # arm stiffness: kills the 5-6cm gravity sag
    if astr != 1.0:
        for jn in ARMJ_R + ARMJ_L:
            if jn in rj:
                try: eng.native.set_revolute_strength(rj[jn], astr)
                except Exception: pass

    S = dict(eng=eng, recs=recs, rj=rj, pj=pj, ta=ta, tb=tb,
             cab_id=cab_id, drw_id=drw_id, doll_gid=doll_gid, DRAWER_HALF=DRAWER_MAX/2,
             DOLL_SPOT=DOLL_SPOT, CAB_TOP=CAB_TOP, DRAWER_MAX=DRAWER_MAX,
             HANDLE0=np.r_[(dc - FACING * (DRAWER_D/2 + 0.065) + LEFTY * (0.0 if CAB_V2 else 0.04))[:2], HANDLE_Z],
             FACING=FACING, LEFTY=LEFTY, WAIST=WAIST)

    def set_fingers(side, g):
        FING = FING_R if side == "r" else FING_L
        TH = THUMB_R if side == "r" else THUMB_L
        for fn in FING:
            if fn in rj: eng.native.set_revolute_target(rj[fn], g * -1.9)
        for fn in TH:
            if fn in rj: eng.native.set_revolute_target(rj[fn], g * 1.4)
    def set_arm(side, tgts):
        for jn, v in tgts.items():
            if jn in rj: eng.native.set_revolute_target(rj[jn], float(v))
    def doll_c():
        return np.asarray(eng.get_vertices())[ta:tb].mean(0)
    def drawer_pull():
        return float(eng.native.get_prismatic_current_distance(pj))
    S.update(set_fingers=set_fingers, set_arm=set_arm, doll_c=doll_c,
             drawer_pull=drawer_pull)
    return S


def _build_cab_v3(eng, recs):
    """v3: user two-drawer cabinet OBJ (world coords). Right drawer (robot's right,
    lat<0) = physics, built half-open; left drawer + body = fixed. Handle measured."""
    m = trimesh.load(CAB_MESH, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int32)
    W0 = WAIST * np.array([1, 1, 0])
    rel = V - W0
    fF = rel @ FACING; fL = rel @ LEFTY; fz = V[:, 2]
    inA = (fF > 0.30) & (fF < 0.85) & (fL > -0.47) & (fL < -0.021) & (fz > 0.735) & (fz < 0.925)
    selA = np.all(inA[F], axis=1)                          # right drawer (incl. centered handle)
    # user mesh has большой cover plates sealing the drawer top (z~0.911) — a drawer
    # must be open-top to receive the doll: drop wide horizontal lid faces, keep rims.
    triA = V[F]
    flat_top = np.all(np.abs(triA[:, :, 2] - 0.911) < 0.004, axis=1)
    fspan = (triA[:, :, 0].max(1) - triA[:, :, 0].min(1))
    lspan = (triA[:, :, 1].max(1) - triA[:, :, 1].min(1))
    lid = selA & flat_top & (fspan > 0.05) & (lspan > 0.05)
    if lid.sum():
        print(f"[scene-v3] 开顶: 删除抽屉盖板面片 {lid.sum()} (保留壁沿)", flush=True)
    selA = selA & ~lid
    keep = ~lid                                            # lids vanish from render too (open drawer)
    def submesh(sel):
        idx = np.unique(F[sel])
        remap = np.zeros(len(V), np.int64); remap[idx] = np.arange(len(idx))
        return V[idx].copy(), remap[F[sel]].astype(np.int32)
    Vc, Fc = submesh(~selA & keep)
    eng.load_mesh_from_data(Vc, Fc, verts_per_face=3, dimensions=3, body_type="ABD",
                            transform=np.eye(4), boundary_type="Fixed", young_modulus=1e8)
    cab_id = eng.get_load_records()[-1].body_offset
    V3_OPEN = 0.20                                         # v02=rest真关位; 开20cm(落斗窗~13cm, 关程=已验证臂程)
    Va, Fa = submesh(selA)
    Va = Va - FACING * V3_OPEN                             # build 18cm-open (pull=0)
    eng.load_mesh_from_data(Va, Fa, verts_per_face=3, dimensions=3, body_type="ABD",
                            transform=np.eye(4), boundary_type="Free", young_modulus=1e8)
    drw_rec = eng.get_load_records()[-1]
    drw_id = drw_rec.body_offset
    eng.native.set_abd_body_density(drw_id, 300.0)
    relA = Va - W0
    bar = relA @ FACING < (relA @ FACING).min() + 0.05     # front-most cluster = handle bar
    HANDLE0 = Va[bar].mean(0)
    dcA = Va.mean(0)
    print(f"[scene-v3] drawerA {len(Va)}v handle0={np.round(HANDLE0,3)} "
          f"lat_c={np.dot(dcA-W0, LEFTY):+.3f}", flush=True)

    # ---- doll (identical to v1) ----
    verts, cells = tetify(os.environ.get("FD_TOY", "/home/ps/Downloads/geo/Toy00.obj")); verts = verts * TSCALE
    lo, hi = verts.min(0), verts.max(0); ctr = (lo + hi) / 2
    Tt = np.eye(4)
    Tt[:3, 3] = [DOLL_SPOT[0] - ctr[0], DOLL_SPOT[1] - ctr[1], CAB_TOP - lo[2] + 0.003]
    eng.load_mesh_from_data(verts, cells, verts_per_face=4, dimensions=3, body_type="FEM",
                            transform=Tt, young_modulus=YOUNG, boundary_type="Free")
    trec = eng.get_load_records()[-1]
    ta, tb = trec.vertex_offset, trec.vertex_offset + trec.vertex_count
    n_abd = eng.native.get_abd_body_count()
    doll_gid = n_abd + trec.body_offset

    anchor = np.r_[dcA[:2], dcA[2]]
    pj = eng.native.add_prismatic_joint(cab_id, drw_id, np.asarray(anchor, float),
                                        (-FACING).astype(float), -(V3_OPEN + 0.01), DRAWER_MAX - V3_OPEN, "drawer")
    eng.native.set_body_apply_gravity(drw_id, False)

    gripR = {recs[l].body_offset for l in GRIP_R}
    gripL = {recs[l].body_offset for l in GRIP_L}
    allids = list(range(n_abd + eng.native.get_fem_body_count()))
    allow = set()
    for g in gripR | gripL:
        allow.add(frozenset((g, doll_gid)))
    for g in gripR | gripL:                                # v3: RIGHT hand closes the right drawer
        allow.add(frozenset((g, drw_id)))
    allow |= {frozenset((doll_gid, cab_id)), frozenset((doll_gid, drw_id))}
    for g in gripR:
        allow.add(frozenset((g, cab_id)))
    for i in range(len(allids)):
        for j in range(i + 1, len(allids)):
            if frozenset((allids[i], allids[j])) not in allow:
                eng.add_collision_exclusion(allids[i], allids[j])
    for b in allids:
        if b != doll_gid:
            eng.add_ground_collision_skip(b)
    eng.finalize()
    if os.environ.get("FD_DRAWER_PASSIVE", "1") == "1":
        try:
            eng.native.set_prismatic_strength(pj, 0.0)
        except Exception:
            pass
    eng.native.set_log_level(0)
    rj = {eng.native.get_revolute_joint_info(i).name: i
          for i in range(eng.native.get_num_revolute_joints())}
    fstr = float(os.environ.get("FD_FSTR", "1"))
    if fstr != 1.0:
        for fn in FING_R + THUMB_R:
            if fn in rj:
                try: eng.native.set_revolute_strength(rj[fn], fstr)
                except Exception: pass
    astr = float(os.environ.get("FD_ASTR", "1"))               # arm stiffness: kills the 5-6cm gravity sag
    if astr != 1.0:
        for jn in ARMJ_R + ARMJ_L:
            if jn in rj:
                try: eng.native.set_revolute_strength(rj[jn], astr)
                except Exception: pass
    S = dict(eng=eng, recs=recs, rj=rj, pj=pj, ta=ta, tb=tb,
             cab_id=cab_id, drw_id=drw_id, doll_gid=doll_gid, DRAWER_HALF=V3_OPEN,
             DOLL_SPOT=DOLL_SPOT, CAB_TOP=CAB_TOP, DRAWER_MAX=DRAWER_MAX,
             HANDLE0=HANDLE0, FACING=FACING, LEFTY=LEFTY, WAIST=WAIST)
    def set_fingers(side, g):
        FING = FING_R if side == "r" else FING_L
        TH = THUMB_R if side == "r" else THUMB_L
        for fn in FING:
            if fn in rj: eng.native.set_revolute_target(rj[fn], g * -1.9)
        for fn in TH:
            if fn in rj: eng.native.set_revolute_target(rj[fn], g * 1.4)
    def set_arm(side, tgts):
        for jn, v in tgts.items():
            if jn in rj: eng.native.set_revolute_target(rj[jn], float(v))
    def doll_c():
        return np.asarray(eng.get_vertices())[ta:tb].mean(0)
    def drawer_pull():
        return float(eng.native.get_prismatic_current_distance(pj))
    S.update(set_fingers=set_fingers, set_arm=set_arm, doll_c=doll_c,
             drawer_pull=drawer_pull)
    return S


if __name__ == "__main__":
    import time, functools
    print = functools.partial(print, flush=True)
    S = build()
    eng = S["eng"]
    print(f"[scene] bodies={eng.native.get_abd_body_count()}+1FEM  "
          f"verts={len(np.asarray(eng.get_vertices()))}")
    t0 = time.time()
    for _ in range(30):
        eng.step()
    print(f"[scene] settle {((time.time()-t0)/30*1000):.0f} ms/step  "
          f"doll={np.round(S['doll_c'](),3)} (spot={np.round(S['DOLL_SPOT'],3)}, top={S['CAB_TOP']})")
    # drawer drive check
    eng.native.set_prismatic_target(S["pj"], S["DRAWER_MAX"])
    for _ in range(70):
        eng.step()
    print(f"[scene] drawer pull={S['drawer_pull']():.3f} (target {S['DRAWER_MAX']})")
    eng.native.set_prismatic_target(S["pj"], 0.0)
    for _ in range(70):
        eng.step()
    print(f"[scene] drawer close={S['drawer_pull']():.3f} (target 0)  "
          f"doll={np.round(S['doll_c'](),3)}")
    V = np.asarray(eng.get_vertices())
    print(f"[scene] finite={np.isfinite(V).all()}")
    print("[scene] OK")
