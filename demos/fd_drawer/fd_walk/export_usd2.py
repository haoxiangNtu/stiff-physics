#!/usr/bin/env python3
"""Export the 4-stage animation to USD, frame-aligned with the composite mp4
(incl. the frozen-scene camera-blend segments).

Outputs:
  fd_4stage_animation.usdc       decimated (~4%) links + transform animation
  fd_4stage_animation_flat.usdc  decimated (~1.5%) links, WORLD-baked points
                                 every frame, no xforms (viewer-proof)
Usage: python export_usd2.py [lite|flat|both]
"""
import sys, os, json
import numpy as np
from scipy.spatial.transform import Rotation as R
import fast_simplification
sys.path.insert(0, "/home/ps/Downloads/FD-light/fd_walk")
from fd_walk_render import URDFModel, T, URDF
from pxr import Usd, UsdGeom, Gf, Vt

MODE = (sys.argv[1] if len(sys.argv) > 1 else "both")
WALK = "/home/ps/Downloads/FD-light/fd_walk"
ARM_SWING = 0.45

def kabsch(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt_ = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt_.T @ U.T))
    Rm = Vt_.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Rm, cQ - Rm @ cP

d = np.load(os.path.join(WALK, "gait_traj.npz"), allow_pickle=True)
names = list(d["joint_names"]); q = d["joint_angles"]
bp = d["base_pos"]; bq = d["base_quat_wxyz"]
model = URDFModel(URDF)

def base_T(t):
    w, x, y, z = bq[t]
    return T(R.from_quat([x, y, z, w]).as_matrix(), bp[t])
def jang_gait(t, arm_override=None):
    ja = {names[i]: float(q[t, i]) for i in range(len(names))}
    if arm_override in ("swing", "zero"):
        for n in list(ja):
            if not n.startswith("leg_"):
                ja[n] = 0.0
    if arm_override == "swing":
        ja["arm_r1_joint"] = ARM_SWING * ja.get("leg_l1_joint", 0.0)
        ja["arm_l1_joint"] = -ARM_SWING * ja.get("leg_r1_joint", 0.0)
    return ja

S2d = np.load(os.path.join(WALK, "stage2_out", "stage2_states.npz"))
s2_times = S2d["times"]; s2_verts = S2d["verts"]; s2_faces = S2d["faces"]
ta, tb = int(S2d["ta"]), int(S2d["tb"])
layout = json.load(open(os.path.join(WALK, "stage2_out", "layout.json")))

isdoll = np.all((s2_faces >= ta) & (s2_faces < tb), axis=1)
rmask = np.zeros(s2_verts.shape[1], bool)
for e in layout["robot"]:
    rmask[e["off"]:e["off"] + e["cnt"]] = True
isrobot = np.all(rmask[s2_faces], axis=1)
furnF = s2_faces[~isdoll & ~isrobot]
dollF = s2_faces[isdoll] - ta
FIDX = np.unique(furnF)
furnF_local = np.searchsorted(FIDX, furnF)

IDX_SETTLE = int(np.searchsorted(s2_times, 2.0))
V_settle = s2_verts[IDX_SETTLE]
V_final = s2_verts[-1]

FROZEN_T = 178
Wfrozen = model.fk(base_T(FROZEN_T), jang_gait(FROZEN_T))
s2_link_proxy = {e["label"]: (e["off"], e["off"] + e["cnt"]) for e in layout["robot"]}
rng = np.random.RandomState(0)
proxy_idx = {}
for lb, (lo_, hi_) in s2_link_proxy.items():
    n = hi_ - lo_
    proxy_idx[lb] = lo_ + rng.choice(n, min(48, n), replace=False)
P0 = {lb: s2_verts[0][idx] for lb, idx in proxy_idx.items()}
W0_vis = {lb: Wfrozen[lb] for lb in s2_link_proxy if lb in Wfrozen}

WAIST = np.array([2.3313, 0.0359, 1.05])
F_ = np.array([np.cos(-0.101823), np.sin(-0.101823), 0])
L_ = np.array([-np.sin(-0.101823), np.cos(-0.101823), 0])

def san(nm):
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in nm)
    return out if not out[0].isdigit() else "_" + out

def decimate_links(reduction):
    if reduction <= 0:                                        # full resolution passthrough
        out = {}
        for ln, metas in model.links.items():
            Vs, Fs, off = [], [], 0
            for (Vl, Fl, _c) in metas:
                Vl = np.asarray(Vl, np.float32); Fl = np.asarray(Fl, np.int32)
                Vs.append(Vl); Fs.append(Fl + off); off += len(Vl)
            if Vs:
                out[ln] = (np.concatenate(Vs), np.concatenate(Fs))
        return out

    out = {}
    for ln, metas in model.links.items():
        Vs, Fs, off = [], [], 0
        for (Vl, Fl, _c) in metas:
            Vl = np.asarray(Vl, np.float32); Fl = np.asarray(Fl, np.int32)
            if len(Fl) > 200:
                Vl, Fl = fast_simplification.simplify(Vl, Fl, target_reduction=reduction)
            Vs.append(Vl); Fs.append(Fl + off); off += len(Vl)
        if Vs:
            out[ln] = (np.concatenate(Vs), np.concatenate(Fs))
    return out

# ---------------- frame program (identical to composite_render) --------------
def frame_program():
    """yield (kind, payload):
       kind='links' payload=(W_dict, furnV_or_None, cam)  — one output frame"""
    mid = np.array([(bp[0][0] + 2.97) / 2, (bp[0][1]) / 2])
    span = abs(2.97 - bp[0][0]) + 1.2
    s1cam = ((mid[0] + 0.3, mid[1] - span * 1.35, 1.9), (mid[0] + 0.1, mid[1], 0.85))
    eye2 = tuple(WAIST + F_ * 0.30 - L_ * 1.85 + np.array([0, 0, 0.42]))
    at2 = tuple(WAIST + F_ * 0.60 + np.array([0, 0, -0.22]))
    def blend(camA, camB, n, W, furnV):
        for i in range(n):
            w = (i + 1) / n; w = w * w * (3 - 2 * w)
            eye = tuple((1 - w) * np.asarray(camA[0]) + w * np.asarray(camB[0]))
            at = tuple((1 - w) * np.asarray(camA[1]) + w * np.asarray(camB[1]))
            yield (W, furnV, (eye, at))
    # stage 1
    for t in range(0, 179):
        yield (model.fk(base_T(t), jang_gait(t)), V_settle if t == 0 else None, s1cam)
    # blend 1->2
    W178 = model.fk(base_T(178), jang_gait(178))
    yield from blend(s1cam, (eye2, at2), 22, W178, None)
    # stage 2
    for k in range(IDX_SETTLE, len(s2_times)):
        Vf = s2_verts[k]
        W = {}
        for lb in s2_link_proxy:
            if lb not in W0_vis:
                continue
            Rm, tt = kabsch(P0[lb], Vf[proxy_idx[lb]])
            M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = tt
            W[lb] = M @ W0_vis[lb]
        for ln in model.links:
            if ln not in W and ln in Wfrozen:
                W[ln] = Wfrozen[ln]
        yield (W, Vf, (eye2, at2))
    # blend 2->3
    b0 = bp[178]
    cam3 = ((b0[0] + 2.2, b0[1] - 3.2, 1.8), ((b0[0] + 2.77) / 2, b0[1] / 2, 0.85))
    Wend = model.fk(base_T(178), jang_gait(178, "zero"))
    yield from blend((eye2, at2), cam3, 22, Wend, V_final)
    # stage 3: right-foot back-step (reversed 140->119, rightward drift)
    RIGHTY = -L_
    BS0, BS1, DRIFT = 136, 123, 0.15
    OFFB = bp[178] - bp[BS0]
    jaA = jang_gait(178, "zero"); jaB = jang_gait(BS0, "zero")
    b0 = bp[178]
    cam3 = ((b0[0] + 2.2, b0[1] - 3.2, 1.8), ((b0[0] + 2.77) / 2, b0[1] / 2, 0.85))
    for i in range(8):
        w = (i + 1) / 8.0
        ja = {n: (1 - w) * jaA.get(n, 0.0) + w * jaB.get(n, 0.0) for n in set(jaA) | set(jaB)}
        w_, x_, y_, z_ = bq[BS0]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), bp[BS0] + OFFB)
        yield (model.fk(bT, ja), None, cam3)
    n_bs = BS0 - BS1 + 1
    for k, t in enumerate(range(BS0, BS1 - 1, -1)):
        drift = RIGHTY * (DRIFT * (k + 1) / n_bs)
        base = bp[t] + OFFB + drift
        w_, x_, y_, z_ = bq[t]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), base)
        yield (model.fk(bT, jang_gait(t, "zero")), None,
               ((base[0] + 2.2, base[1] - 3.2, 1.8), ((base[0] + 2.77) / 2, base[1] / 2, 0.85)))
    base_end = bp[BS1] + OFFB + RIGHTY * DRIFT
    jaA = jang_gait(BS1, "zero"); jaSTAND = jang_gait(178, "zero"); jaB = jang_gait(259, "zero")
    camB3 = ((base_end[0] + 2.2, base_end[1] - 3.2, 1.8), ((base_end[0] + 2.77) / 2, base_end[1] / 2, 0.85))
    for i in range(8):
        w = (i + 1) / 8.0
        ja = {n: (1 - w) * jaA.get(n, 0.0) + w * jaSTAND.get(n, 0.0) for n in set(jaA) | set(jaSTAND)}
        w_, x_, y_, z_ = bq[178]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), base_end)
        yield (model.fk(bT, ja), None, camB3)
    for i in range(8):
        w = (i + 1) / 8.0
        ja = {n: (1 - w) * jaSTAND.get(n, 0.0) + w * jaB.get(n, 0.0) for n in set(jaSTAND) | set(jaB)}
        w_, x_, y_, z_ = bq[259]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), base_end)
        yield (model.fk(bT, ja), None, camB3)
    cam4 = ((base_end[0] + 2.6, base_end[1] - 2.6, 1.6), (base_end[0], base_end[1], 0.9))
    yield from blend(camB3, cam4, 16, model.fk(T(R.from_quat([bq[259][1], bq[259][2], bq[259][3], bq[259][0]]).as_matrix(), base_end), jaB), None)
    # stage 4: turn right + walk away
    OFF = base_end - bp[259]
    for t in range(259, 419):
        ja = jang_gait(t, "swing")
        w_, x_, y_, z_ = bq[t]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), bp[t] + OFF)
        b = bp[t] + OFF
        yield (model.fk(bT, ja), None, ((b[0] + 2.6, b[1] - 2.6, 1.6), (b[0], b[1], 0.9)))

def gfmat(M):
    return Gf.Matrix4d(*[float(v) for v in np.asarray(M, float).T.ravel()])

def cam_matrix(eye, at):
    eye = np.asarray(eye, float); at = np.asarray(at, float)
    zc = eye - at; zc /= np.linalg.norm(zc)
    up = np.array([0.0, 0.0, 1.0])
    xc = np.cross(up, zc); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return Gf.Matrix4d(xc[0], xc[1], xc[2], 0, yc[0], yc[1], yc[2], 0,
                       zc[0], zc[1], zc[2], 0, eye[0], eye[1], eye[2], 1)

def export(path, links, flat):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(25); stage.SetFramesPerSecond(25)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    def mk_mesh(pathp, V, F, color):
        m = UsdGeom.Mesh.Define(stage, pathp)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.astype(np.int32).ravel()))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        return m
    link_prims = {}
    for ln, (V, F) in links.items():
        m = mk_mesh(f"/World/Robot/{san(ln)}", V, F, (0.72, 0.77, 0.86))
        if flat:
            link_prims[ln] = (m.GetPointsAttr(), V)
        else:
            xf = UsdGeom.Xformable(m.GetPrim()).AddTransformOp()
            link_prims[ln] = (xf, None)
    furn_m = mk_mesh("/World/Furniture", V_settle[FIDX], furnF_local, (0.62, 0.55, 0.48))
    doll_m = mk_mesh("/World/Doll", V_settle[ta:tb], dollF, (0.95, 0.45, 0.15))
    fp, dp = furn_m.GetPointsAttr(), doll_m.GetPointsAttr()
    camp = UsdGeom.Camera.Define(stage, "/World/Cam")
    camp.CreateFocalLengthAttr(18.5)
    cam_xf = UsdGeom.Xformable(camp.GetPrim()).AddTransformOp()
    first = True
    fi = 0
    for W, furnV, camAB in frame_program():
        tc = Usd.TimeCode(fi)
        for ln, (h, V0) in link_prims.items():
            if ln not in W:
                continue
            if flat:
                Vw = (W[ln][:3, :3] @ V0.T).T + W[ln][:3, 3]
                arr = Vt.Vec3fArray.FromNumpy(Vw.astype(np.float32))
                if first:
                    h.Set(arr)                                # default-time = frame-0 world pose
                h.Set(arr, tc)
            else:
                M = gfmat(W[ln])
                if first:
                    h.Set(M)
                h.Set(M, tc)
        if furnV is not None:
            fp.Set(Vt.Vec3fArray.FromNumpy(furnV[FIDX].astype(np.float32)), tc)
            dp.Set(Vt.Vec3fArray.FromNumpy(furnV[ta:tb].astype(np.float32)), tc)
        M = cam_matrix(*camAB)
        if first:
            cam_xf.Set(M)
        cam_xf.Set(M, tc)
        first = False
        fi += 1
        if fi % 300 == 0:
            print(f"  {os.path.basename(path)} f{fi}", flush=True)
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(fi - 1)
    stage.GetRootLayer().Save()
    print(f"[usd] {path} frames 0..{fi-1}", flush=True)

if MODE in ("lite", "both"):
    links = decimate_links(0.0)
    n = sum(len(V) for V, _ in links.values())
    print(f"[lite] robot verts: {n}", flush=True)
    export(os.path.join(WALK, "fd_4stage_animation.usdc"), links, flat=False)
if MODE in ("flat", "both"):
    links = decimate_links(0.96)
    n = sum(len(V) for V, _ in links.values())
    print(f"[flat] robot verts: {n}", flush=True)
    export(os.path.join(WALK, "fd_4stage_animation_flat.usdc"), links, flat=True)
