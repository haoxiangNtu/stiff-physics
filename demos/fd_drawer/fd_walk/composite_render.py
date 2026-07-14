#!/usr/bin/env python3
"""Final 3-stage composite render (full-res, Z-up).

Stage 1 (gait 0..178):     kinematic replay of gait_traj (walk -> settle stand)
Stage 2 (physics):         body/legs/head FROZEN at the frame-178 FK pose;
                           both arms + hands driven by Kabsch from stage2_states.npz;
                           cabinet + drawer + doll rendered from the engine states
Stage 3 (gait 259..418):   kinematic replay (turn + walk away), arms overridden to
                           a natural swing (story: hands are empty again)

Run:  test_stiff08/bin/python3.11 composite_render.py --out composite_frames [--stride 2]
      ffmpeg -r 25 -i composite_frames/f%05d.png ... final.mp4
"""
import sys, os, argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, "/home/ps/Downloads/FD-light/fd_walk")
from fd_walk_render import URDFModel, T, URDF

WALK = "/home/ps/Downloads/FD-light/fd_walk"
S2 = os.path.join(WALK, "stage2_out", "stage2_states.npz")
ARM_SWING = 0.45

def kabsch(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Rm, cQ - Rm @ cP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WALK, "composite_frames"))
    ap.add_argument("--stride", type=int, default=1, help="stage1/3 gait frame stride")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(WALK, "gait_traj.npz"), allow_pickle=True)
    names = list(d["joint_names"]); q = d["joint_angles"]
    bp = d["base_pos"]; bq = d["base_quat_wxyz"]
    model = URDFModel(URDF)

    def base_T(t):
        w, x, y, z = bq[t]
        return T(R.from_quat([x, y, z, w]).as_matrix(), bp[t])
    def jang_gait(t, arm_override=None):
        ja = {names[i]: float(q[t, i]) for i in range(len(names))}
        if arm_override in ("swing", "zero"):              # non-leg joints -> 0
            for n in list(ja):
                if not n.startswith("leg_"):
                    ja[n] = 0.0
        if arm_override == "swing":                        # natural swing, open hands
            ja["arm_r1_joint"] = ARM_SWING * ja.get("leg_l1_joint", 0.0)
            ja["arm_l1_joint"] = -ARM_SWING * ja.get("leg_r1_joint", 0.0)
        return ja

    # stage-2 states + record layout (rebuild scene topology WITHOUT engine: layout
    # is deterministic -> saved by stage2_run? fall back: derive via a fresh build)
    S2d = np.load(S2)
    s2_times = S2d["times"]; s2_verts = S2d["verts"]; s2_faces = S2d["faces"]
    ta, tb = int(S2d["ta"]), int(S2d["tb"])
    import json
    layout = json.load(open(os.path.join(WALK, "stage2_out", "layout.json")))

    # ---- polyscope ----
    import polyscope as ps
    try: ps.set_allow_headless_backends(True)
    except Exception: pass
    ps.set_program_name("composite"); ps.init()
    ps.set_up_dir("z_up"); ps.set_front_dir("neg_y_front"); ps.set_ground_plane_mode("shadow_only")

    # full-res robot links
    link_meshes = {}
    for ln, metas in model.links.items():
        Vs, Fs, off = [], [], 0
        for (Vl, Fl, _c) in metas:
            Vs.append(Vl); Fs.append(Fl + off); off += len(Vl)
        if not Vs:
            continue
        pm = ps.register_surface_mesh("r::" + ln, np.concatenate(Vs), np.concatenate(Fs),
                                      smooth_shade=True, color=(0.72, 0.77, 0.86))
        link_meshes[ln] = (pm, np.concatenate(Vs))

    # stage-2 scene meshes (cabinet+drawer from state verts; doll)
    isdoll = np.all((s2_faces >= ta) & (s2_faces < tb), axis=1)
    # non-robot, non-doll = cabinet+drawer: identified via layout ranges
    robot_rng = [(e["off"], e["off"] + e["cnt"]) for e in layout["robot"]]
    rmask = np.zeros(s2_verts.shape[1], bool)
    for lo_, hi_ in robot_rng:
        rmask[lo_:hi_] = True
    isrobot = np.all(rmask[s2_faces], axis=1)
    furnF = s2_faces[~isdoll & ~isrobot]
    dollF = s2_faces[isdoll]
    V00 = s2_verts[0]
    # settled state (doll at rest on the cabinet) — shown throughout stage 1
    IDX_SETTLE = int(np.searchsorted(s2_times, 2.0))
    V_settle = s2_verts[IDX_SETTLE]
    V_final = s2_verts[-1]                     # closed drawer, doll inside (stages 3/4)
    furn = ps.register_surface_mesh("furniture", V_settle, furnF, color=(0.62, 0.55, 0.48),
                                    smooth_shade=False)
    doll = ps.register_surface_mesh("doll", V_settle, dollF, color=(0.95, 0.45, 0.15),
                                    smooth_shade=True)

    def pose_robot_fk(t, arm_override=None):
        W = model.fk(base_T(t), jang_gait(t, arm_override))
        for ln, (pm, _V) in link_meshes.items():
            if ln in W:
                pm.set_transform(W[ln])

    # stage-2: frozen body FK + Kabsch arms/hands
    FROZEN_T = 178
    Wfrozen = model.fk(base_T(FROZEN_T), jang_gait(FROZEN_T))
    s2_link_proxy = {e["label"]: (e["off"], e["off"] + e["cnt"]) for e in layout["robot"]}
    rng = np.random.RandomState(0)
    proxy_idx = {}
    for lb, (lo_, hi_) in s2_link_proxy.items():
        n = hi_ - lo_
        sub = rng.choice(n, min(48, n), replace=False)
        proxy_idx[lb] = lo_ + sub
    P0 = {lb: s2_verts[0][idx] for lb, idx in proxy_idx.items()}

    # doll is REAL physics now (drop-in + tamp + ride the drawer) — no script layer
    def pose_stage2(fi):
        Vf = s2_verts[fi]
        for ln, (pm, _V) in link_meshes.items():
            if ln in s2_link_proxy:                        # arm/hand: Kabsch from engine
                Rm, tt = kabsch(P0[ln], Vf[proxy_idx[ln]])
                M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = tt
                base = W0_vis.get(ln)
                pm.set_transform(M @ base)
            elif ln in Wfrozen:                            # body/legs/head frozen
                pm.set_transform(Wfrozen[ln])
        furn.update_vertex_positions(Vf)
        doll.update_vertex_positions(Vf)

    # visual anchor: full-res link world pose matching s2 frame0 = FK of the dual-arm
    # zero pose on the frozen stand (arms hang at 0 like frame 178) -> Wfrozen itself.
    W0_vis = {lb: Wfrozen[lb] for lb in s2_link_proxy if lb in Wfrozen}

    def cam(eye, at):
        ps.look_at(tuple(eye), tuple(at)); ps.set_window_size(1280, 720)

    fi = 0
    def shoot():
        nonlocal fi
        ps.screenshot(os.path.join(args.out, f"f{fi:05d}.png")); fi += 1

    WAIST = np.array([2.3313, 0.0359, 1.05])
    F_ = np.array([np.cos(-0.101823), np.sin(-0.101823), 0])
    L_ = np.array([-np.sin(-0.101823), np.cos(-0.101823), 0])

    def furn_static(V):
        furn.update_vertex_positions(V); doll.update_vertex_positions(V)
    def cam_blend(camA, camB, n=20):
        """scene frozen, camera eases from A to B (sim does NOT step)"""
        for i in range(n):
            w = (i + 1) / n; w = w * w * (3 - 2 * w)
            eye = tuple((1 - w) * np.asarray(camA[0]) + w * np.asarray(camB[0]))
            at = tuple((1 - w) * np.asarray(camA[1]) + w * np.asarray(camB[1]))
            cam(eye, at); shoot()
    # ---- stage 1: walk in (static wide cam: path start + cabinet in frame) ----
    furn_static(V_settle)
    mid = np.array([(bp[0][0] + 2.97) / 2, (bp[0][1] + 0.0) / 2])
    span = abs(2.97 - bp[0][0]) + 1.2
    s1_eye = (mid[0] + 0.3, mid[1] - span * 1.35, 1.9)
    s1_at = (mid[0] + 0.1, mid[1], 0.85)
    for t in range(0, 179, args.stride):
        pose_robot_fk(t)
        cam(s1_eye, s1_at)
        shoot()
        if t % 60 == 0: print(f"stage1 {t}/178", flush=True)
    # ---- blend 1->2: robot frozen at stand, camera glides to the side view ----
    eye = WAIST + F_ * 0.30 - L_ * 1.85 + np.array([0, 0, 0.42])
    at = WAIST + F_ * 0.60 + np.array([0, 0, -0.22])
    pose_robot_fk(178)
    cam_blend((s1_eye, s1_at), (eye, at), 22)
    # ---- stage 2: pure-physics manipulation (start from the settled state) ----
    for k in range(IDX_SETTLE, len(s2_times)):
        pose_stage2(k)
        cam(eye, at)
        shoot()
        if k % 100 == 0: print(f"stage2 {k}/{len(s2_times)}", flush=True)
    # ---- blend 2->3: scene frozen at stage-2 end, camera glides to follow-cam ----
    furn_static(V_final)
    T_BACK = 110
    b0 = bp[178]
    cam3_0 = ((b0[0] + 2.2, b0[1] - 3.2, 1.8), ((b0[0] + 2.77) / 2, b0[1] / 2, 0.85))
    cam_blend((eye, at), cam3_0, 22)
    # ---- stage 3: right-foot back-step (reversed gait 140->119) + rightward drift ----
    furn_static(V_final)
    RIGHTY = -L_
    BS0, BS1, DRIFT = 136, 123, 0.15                          # 右脚摆动核心: 单步"向右后方立正"
    OFFB = bp[178] - bp[BS0]
    def follow(p):
        return ((p[0] + 2.2, p[1] - 3.2, 1.8), ((p[0] + 2.77) / 2, p[1] / 2, 0.85))
    jaA = jang_gait(178, "zero"); jaB = jang_gait(BS0, "zero")
    for i in range(6):
        w = (i + 1) / 6.0
        ja = {n: (1 - w) * jaA.get(n, 0.0) + w * jaB.get(n, 0.0) for n in set(jaA) | set(jaB)}
        w_, x_, y_, z_ = bq[BS0]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), bp[BS0] + OFFB)
        W = model.fk(bT, ja)
        for ln, (pm, _V) in link_meshes.items():
            if ln in W: pm.set_transform(W[ln])
        cam(*follow(bp[178])); shoot()
    n_bs = BS0 - BS1 + 1
    for k, t in enumerate(range(BS0, BS1 - 1, -1)):
        drift = RIGHTY * (DRIFT * (k + 1) / n_bs)
        base = bp[t] + OFFB + drift
        w_, x_, y_, z_ = bq[t]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), base)
        W = model.fk(bT, jang_gait(t, "zero"))
        for ln, (pm, _V) in link_meshes.items():
            if ln in W: pm.set_transform(W[ln])
        cam(*follow(base)); shoot()
    base_end = bp[BS1] + OFFB + RIGHTY * DRIFT
    jaA = jang_gait(BS1, "zero"); jaSTAND = jang_gait(178, "zero")
    for i in range(8):                                        # 落步 -> 立正站定
        w = (i + 1) / 8.0
        ja = {n: (1 - w) * jaA.get(n, 0.0) + w * jaSTAND.get(n, 0.0) for n in set(jaA) | set(jaSTAND)}
        w_, x_, y_, z_ = bq[178]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), base_end)
        W = model.fk(bT, ja)
        for ln, (pm, _V) in link_meshes.items():
            if ln in W: pm.set_transform(W[ln])
        cam(*follow(base_end)); shoot()
    jaB = jang_gait(259, "zero")
    for i in range(8):                                        # 立正 -> 转身起始
        w = (i + 1) / 8.0
        ja = {n: (1 - w) * jaSTAND.get(n, 0.0) + w * jaB.get(n, 0.0) for n in set(jaSTAND) | set(jaB)}
        w_, x_, y_, z_ = bq[259]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), base_end)
        W = model.fk(bT, ja)
        for ln, (pm, _V) in link_meshes.items():
            if ln in W: pm.set_transform(W[ln])
        cam(*follow(base_end)); shoot()
    _off4 = base_end - bp[259]
    _clear = min(2.81 - (bp[t] + _off4)[0] for t in range(259, 341)
                 if -0.52 < (bp[t] + _off4)[1] < 0.60)
    print(f"[composite] 转弯段与柜面最小净空 {_clear*100:.0f}cm (基座计)", flush=True)
    # ---- stage 4: turn right + walk away (clear of the desk) ----
    OFF4 = base_end - bp[259]
    for t in range(259, 419):
        w_, x_, y_, z_ = bq[t]
        bT = T(R.from_quat([x_, y_, z_, w_]).as_matrix(), bp[t] + OFF4)
        W = model.fk(bT, jang_gait(t, "swing"))
        for ln, (pm, _V) in link_meshes.items():
            if ln in W: pm.set_transform(W[ln])
        cam(*follow(bp[t] + OFF4)); shoot()
        if t % 60 == 0: print(f"s4 {t}", flush=True)
    print(f"[composite] {fi} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
