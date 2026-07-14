#!/usr/bin/env python3
"""Run the USER-recorded stage-2 timeline — pure physics end to end.

Everything from user_poses.json: proven grasp (right), proven pure-contact
drawer pull/close (left).  Drawer fully passive.  No servo, no carry fallback,
no drawer motor.  Milestones + screenshots + state npz (same format as before).
"""
import sys, os, json, time
import numpy as np
import functools
print = functools.partial(print, flush=True)
sys.path.insert(0, "/home/ps/Downloads/FD-light/fd_walk")
from stage2_scene import build, FING_R, FING_L, THUMB_R, THUMB_L, ARMJ_R, ARMJ_L

OUT = os.environ.get("S2_OUT", "/home/ps/Downloads/FD-light/fd_walk/stage2_out")
os.makedirs(OUT, exist_ok=True)
FRAME_EVERY = int(os.environ.get("S2_FRAME_EVERY", "4"))
TL = json.load(open("/home/ps/Downloads/FD-light/fd_walk/stage2_user_timeline.json"))
keys = TL["timeline_user"]; meta = TL["meta"]
DT = meta["dt"]; T_END = keys[-1][0]; HALF = meta["drawer_half"]

S = build()
eng = S["eng"]; rj = S["rj"]; recs = S["recs"]
ta, tb = S["ta"], S["tb"]
print(f"[urun] scene ready, timeline {T_END:.1f}s = {int(T_END/DT)} steps (drawer passive)")

def interp(t):
    for k in range(len(keys) - 1):
        t0, ja0, f0 = keys[k]; t1, ja1, f1 = keys[k + 1]
        if t0 <= t <= t1:
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            ja = {n: (1 - w) * ja0.get(n, 0.0) + w * ja1.get(n, 0.0)
                  for n in set(ja0) | set(ja1)}
            f = {n: (1 - w) * f0[n] + w * f1[n] for n in f1}
            return ja, f
    return keys[-1][1], keys[-1][2]

def apply(t):
    ja, f = interp(t)
    for n, v in ja.items():
        if n in rj: eng.native.set_revolute_target(rj[n], float(v))
    for fn in FING_R:
        if fn in rj: eng.native.set_revolute_target(rj[fn], f["grasp_r"] * -1.9)
    if THUMB_R[0] in rj: eng.native.set_revolute_target(rj[THUMB_R[0]], f["throot_r"])
    for fn in THUMB_R[1:]:
        if fn in rj: eng.native.set_revolute_target(rj[fn], f["thcurl_r"])
    for fn in FING_L:
        if fn not in rj: continue
        v = f["index_l"] if fn.startswith("if_") else f["fist_l"]
        eng.native.set_revolute_target(rj[fn], v * -1.9)
    if THUMB_L[0] in rj: eng.native.set_revolute_target(rj[THUMB_L[0]], f["throot_l"])
    for fn in THUMB_L[1:]:
        if fn in rj: eng.native.set_revolute_target(rj[fn], f["thcurl_l"])

def tips_c():
    V = np.asarray(eng.get_vertices())
    ps_ = []
    for fn in ("if_distal_link_r", "mf_distal_link_r", "rf_distal_link_r", "lf_distal_link_r"):
        r = recs[fn]
        ps_.append(V[r.vertex_offset:r.vertex_offset + r.vertex_count].mean(0))
    return np.mean(ps_, axis=0)

def run_servo2b():
    """closed-loop: measured fingertip cluster vs the intended pre-push point."""
    cur = tips_c()
    delta = np.asarray(TIPS_WANT, float) - cur
    print(f"[servo2B] tips实测={np.round(cur,3)} 目标={np.round(TIPS_WANT,3)} "
          f"Δ=({delta[0]*100:+.1f},{delta[1]*100:+.1f},{delta[2]*100:+.1f})cm", flush=True)
    if np.linalg.norm(delta) > 0.30:
        print("[servo2B] Δ>30cm 不修正", flush=True)
        return
    from fk_dual import DualFK
    from scipy.optimize import minimize
    fkk = DualFK()
    JR_ = [f"arm_r{i}_joint" for i in range(1, 8)]
    rows_fix = [SERVO2["rows"][2], SERVO2["push_row"] - 1, SERVO2["push_row"],
                SERVO2["push_row"] + 1, SERVO2.get("retract_row")]
    for ri in rows_fix:
        if ri is None:
            continue
        seed = {j: keys[ri][1].get(j, 0.0) for j in JR_}
        T0 = fkk.fk(seed)["base_link_r"]
        tgt = T0[:3, 3] + delta; R0 = T0[:3, :3]
        x0 = np.array([seed[j] for j in JR_])
        def cost(x):
            M = fkk.fk(dict(zip(JR_, x)))["base_link_r"]
            return (np.sum((M[:3, 3] - tgt) ** 2) + 0.05 * np.sum((M[:3, :3] - R0) ** 2)
                    + 0.001 * np.sum((x - x0) ** 2))
        r = minimize(cost, x0, method="Nelder-Mead",
                     options=dict(maxiter=3000, xatol=1e-4, fatol=1e-10))
        for j, v in zip(JR_, r.x):
            keys[ri][1][j] = float(np.clip(v, -1.57, 1.57))
    print("[servo2B] 推程已平移补偿", flush=True)

def palm(side):
    r = recs[f"base_link_{side}"]
    V = np.asarray(eng.get_vertices())
    return V[r.vertex_offset:r.vertex_offset + r.vertex_count].mean(0)

# ---- lump servo: measure the doll's on-table bulge, IK-shift grasp keys -----
SERVO = meta.get("servo")
SETTLE_XY = None            # doll centroid at servo time (M3 'moved' reference)
def measure_lump():
    """Centroid of the doll's on-table bulge (the graspable part)."""
    V = np.asarray(eng.get_vertices())[ta:tb]
    FACs = S["FACING"]; LEFs = S["LEFTY"]
    fc = S["WAIST"] + FACs * float(os.environ.get("FD_CAB_FRONT", "0.32"))
    d = V - fc
    on = (V[:, 2] > 0.962) & (np.abs(d @ FACs) < 0.34) & (np.abs(d @ LEFs) < 0.48)
    if on.sum() < 20:
        return None, 0
    Von = V[on]
    top = Von[Von[:, 2] > Von[:, 2].max() - 0.06]        # upper 6cm of the bulge
    return top.mean(0), int(on.sum())

def measure_curtain():
    """The doll fabric hanging over the cabinet FRONT face (user's grasp target)."""
    V = np.asarray(eng.get_vertices())[ta:tb]
    FACs = S["FACING"]; LEFs = S["LEFTY"]
    fc = S["WAIST"] + FACs * float(os.environ.get("FD_CAB_FRONT", "0.32"))
    d = V - fc
    a = d @ FACs
    cur = (a < 0.02) & (a > -0.30) & (V[:, 2] > 0.80) & (V[:, 2] < 1.08) & (np.abs(d @ LEFs) < 0.50)
    if cur.sum() < 20:
        return None, 0
    return V[cur].mean(0), int(cur.sum())

def run_servo():
    global SETTLE_XY
    SETTLE_XY = S["doll_c"]()[:2].copy()
    cc, nc = measure_curtain()
    if cc is not None:
        print(f"[servo] 前挂布帘 centroid=({cc[0]:.3f},{cc[1]:.3f},{cc[2]:.3f}) n={nc} "
              f"(期望球位 2.677,-0.180)", flush=True)
    else:
        print("[servo] 前脸区域无布帘顶点(<20)!", flush=True)
    lump, n = measure_lump()
    if lump is None:
        print("[servo] 桌面上找不到玩偶凸起 (on-table verts<20) — 不修正"); return
    dx, dy = lump[0] - SERVO["palm_xy"][0], lump[1] - SERVO["palm_xy"][1]
    print(f"[servo] lump=({lump[0]:.3f},{lump[1]:.3f},{lump[2]:.3f}) n={n} "
          f"shift=({dx*100:+.1f},{dy*100:+.1f})cm", flush=True)
    if abs(dx) > 0.12 or abs(dy) > 0.12:
        print("[servo] 偏移超过12cm — 超出手臂可达范围, 保持原姿态 (检查 FD_DOLL_SPOT)", flush=True)
        return
    from fk_dual import DualFK
    from scipy.optimize import minimize
    fkk = DualFK()
    JR = [f"arm_r{i}_joint" for i in range(1, 8)]
    hover_row = SERVO.get("hover_row")
    cache = {}
    for ri in SERVO["rows"]:
        t_r, ja, f = keys[ri]
        dz = 0.08 if ri == hover_row else 0.0            # hover key: straight above lump (低举)
        sig = (tuple(round(ja.get(j, 0.0), 6) for j in JR), dz)
        if sig in cache:
            ja.update(cache[sig]); continue
        seed = {j: ja.get(j, 0.0) for j in JR}
        T0 = fkk.fk(seed)["base_link_r"]
        tgt_p = T0[:3, 3] + np.array([dx, dy, dz])
        R0 = T0[:3, :3]
        x0 = np.array([seed[j] for j in JR])
        def cost(x):
            M = fkk.fk(dict(zip(JR, x)))["base_link_r"]
            return (np.sum((M[:3, 3] - tgt_p) ** 2)
                    + 0.05 * np.sum((M[:3, :3] - R0) ** 2)
                    + 0.001 * np.sum((x - x0) ** 2))
        r = minimize(cost, x0, method="Nelder-Mead",
                     options=dict(maxiter=3000, xatol=1e-4, fatol=1e-10))
        sol = {j: float(np.clip(v, -1.57, 1.57)) for j, v in zip(JR, r.x)}
        M = fkk.fk(sol)["base_link_r"]
        pe = np.linalg.norm(M[:3, 3] - tgt_p); re = np.linalg.norm(M[:3, :3] - R0)
        print(f"  row{ri}: dz={dz} pos_err={pe*100:.1f}cm rot_err={re:.3f}", flush=True)
        cache[sig] = sol; ja.update(sol)
    if SERVO["rows"]:
        retime_from(min(SERVO["rows"]))

SERVO2 = meta.get("servo2")
CAB_V3 = bool(os.environ.get("FD_CAB_MESH"))
def run_servo2():
    """Drawer-close servo: measure actual pull, aim approach/push at the REAL handle.
    v1/v2 = left arm; v3 (user mesh, right drawer) = RIGHT arm, mirrored corridors."""
    pull = S["drawer_pull"]()
    FACs = S["FACING"]; LEFs = S["LEFTY"]
    if CAB_V3:
        hp = np.asarray(S["HANDLE0"], float) - FACs * pull    # measured bar (build=half-open=pull0)
    else:
        fc = S["WAIST"] + FACs * float(os.environ.get("FD_CAB_FRONT", "0.32"))
        _v2 = os.environ.get("FD_CAB_V2", "0") == "1"
        dc0 = fc + FACs * 0.15 - FACs * 0.12 - LEFs * (0.0 if _v2 else 0.10)  # drawer center at pull=0 (half-open build)
        h0 = dc0 - FACs * (0.15 + 0.065)                      # handle bar at pull=0 (v2: centered)
        hp = h0 - FACs * pull                                 # actual handle bar now
        hp[2] = 0.84 if _v2 else 0.865                        # v2: 面板几何中心
    side = "r" if CAB_V3 else "l"
    print(f"[servo2{side.upper()}] pull={pull:+.3f} handle=({hp[0]:.3f},{hp[1]:.3f},{hp[2]:.3f})", flush=True)
    from fk_dual import DualFK
    from scipy.optimize import minimize
    fkk = DualFK()
    JL = [f"arm_{side}{i}_joint" for i in range(1, 8)]
    seed_row = keys[SERVO2["fire_row"]][1]                    # chain seed starts at the sweep pose
    seed = {j: seed_row.get(j, 0.0) for j in JL}
    def ik_l(target):
        wrist = {j: seed[j] for j in JL[4:]}
        x0 = np.array([seed[j] for j in JL[:4]])
        def cost(x):
            ja = dict(zip(JL[:4], x)); ja.update(wrist)
            return np.sum((fkk.fk(ja)[f"base_link_{side}"][:3, 3] - target) ** 2) + 0.002 * np.sum(x ** 2)
        r = minimize(cost, x0, method="Nelder-Mead",
                     options=dict(maxiter=2500, xatol=1e-4, fatol=1e-9))
        out = dict(seed)
        for j, v in zip(JL[:4], r.x):
            out[j] = float(np.clip(v, -1.57, 1.57))
        ja = {j: out[j] for j in JL[:4]}; ja.update({j: seed[j] for j in JL[4:]})
        err = np.linalg.norm(fkk.fk(ja)[f"base_link_{side}"][:3, 3] - target)
        return out, err
    if CAB_V3:
        # PROVEN fist corridor, relaxed half-curl hand: knuckle-backs push the bar face.
        # -0.03F = knuckle protrusion vs the closed fist.
        # ENGINE-MEASURED fist envelope: fist front = palm frame +0.11F (centroid+half).
        # Engage parks the fist face 1cm BEFORE the bar; push seats the drawer at rest.
        push_end = hp + FACs * (pull + 0.20 + 0.02 - 0.11)
        tgts = [hp - FACs * 0.13 + np.array([0, 0, 0.14]),    # HIGH far
                hp - FACs * 0.13,                             # DOWN far
                hp - FACs * 0.12,                             # ENGAGE: fist face 1cm off the bar
                hp - FACs * 0.12,                             # dwell (servo2B measures)
                push_end]                                     # ONE slow fist push
    else:
        LAT = LEFs
        push_end = hp + FACs * (pull + 0.175)                 # bar shoved past the stop
        tgts = [hp + np.array([0, 0, 0.14]) - FACs * 0.13 + LAT * 0.05,
                hp - FACs * 0.13 + LAT * 0.05,
                hp - FACs * 0.075 + LAT * 0.02,
                hp + FACs * 0.02,
                push_end]
    rows_all = list(SERVO2["rows"]) + [SERVO2["push_row"]]
    downfar_sol = None
    for ri, tgt in zip(rows_all, tgts):
        sol, err = ik_l(tgt)
        seed = {j: sol.get(j, seed[j]) for j in JL}           # chain: next row seeded from this one
        for k2, v2 in sol.items():
            keys[ri][1][k2] = v2
        if ri == SERVO2["rows"][1 if len(SERVO2["rows"]) > 1 else 0]:
            downfar_sol = sol                                 # retract reference pose
        if ri == SERVO2["push_row"] and SERVO2["push_row"] + 1 < len(keys):
            for k2, v2 in sol.items():                        # dwell row holds the same pose
                keys[SERVO2["push_row"] + 1][1][k2] = v2
        print(f"  [servo2] row{ri} -> ({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}) err={err*100:.1f}cm", flush=True)
    rr = SERVO2.get("retract_row")
    if rr and downfar_sol:
        for k2, v2 in downfar_sol.items():
            keys[rr][1][k2] = v2
    global SERVO2B_T, TIPS_WANT
    SERVO2B_T = keys[SERVO2["rows"][2]][0] + 1.6              # mid-dwell: arm settled
    TIPS_WANT = hp - FACs * 0.045 + np.array([0, 0, 0.0])     # fist knuckle cluster 1cm off the bar (tips≈palm+0.075F)
    retime_from(min(rows_all))

def retime_from(r0):
    """Re-apply the 0.45 rad/s cap from row r0 on (servo changed joint deltas)."""
    global T_END, CHECKS
    RATE, GRATE = 0.45, 0.9
    for i in range(max(r0, 1), len(keys)):
        t0, j0, f0 = keys[i - 1]; t1, j1, f1 = keys[i]
        nominal = t1 - t0 if t1 > t0 else 0.4
        dj = max(abs(j1.get(nm, 0) - j0.get(nm, 0)) for nm in set(j0) | set(j1))
        dg = max(abs(f1[kk] - f0[kk]) for kk in f1)
        keys[i][0] = round(keys[i - 1][0] + max(nominal, dj / RATE, dg / GRATE), 2)
    T_END = keys[-1][0]
    spec = meta.get("check_spec")
    if spec:
        CHECKS = [(float(keys[r][0] + off), nme) for nme, r, off in spec]
        print(f"[servo] retimed: T_END={T_END:.1f}s checks={[(n_, round(t_,1)) for t_, n_ in CHECKS]}",
              flush=True)

import polyscope as ps
try: ps.set_allow_headless_backends(True)
except Exception: pass
ps.set_program_name("s2urun"); ps.init()
ps.set_up_dir("z_up"); ps.set_front_dir("neg_y_front"); ps.set_ground_plane_mode("shadow_only")
F = np.asarray(eng.get_surface_faces()); V0 = np.asarray(eng.get_vertices())
dollF = F[np.all((F >= ta) & (F < tb), axis=1)]
restF = F[~np.all((F >= ta) & (F < tb), axis=1)]
sm = ps.register_surface_mesh("scene", V0, restF, color=(0.72, 0.78, 0.86), smooth_shade=True)
dm = ps.register_surface_mesh("doll", V0, dollF, color=(0.95, 0.45, 0.15), smooth_shade=True)
WAIST = S["WAIST"]; FAC = S["FACING"]; LEF = S["LEFTY"]
CAM_EYE = WAIST + FAC * 0.30 - LEF * 1.85 + np.array([0, 0, 0.42])
CAM_AT = WAIST + FAC * 0.50 + np.array([0, 0, -0.22])
def shot(name):
    Vn = np.asarray(eng.get_vertices())
    sm.update_vertex_positions(Vn); dm.update_vertex_positions(Vn)
    ps.look_at(tuple(CAM_EYE), tuple(CAM_AT)); ps.set_window_size(1000, 800)
    ps.screenshot(os.path.join(OUT, name))

mil = {}
def check(name, t):
    dc = S["doll_c"](); pull = S["drawer_pull"]()
    if name == "M1_grasp_lift":
        pr = palm("r")
        if CAB_V3:
            # v3编舞=舀住贴面拖: 判据=掌位对准+玩偶受控(可能仍在台面, 也可能已提早入斗)
            ok = dc[2] > 0.70 and dc[2] < 1.05 and np.linalg.norm((dc - pr)[:2]) < 0.32
        else:
            ok = dc[2] > 0.975 and np.linalg.norm((dc - pr)[:2]) < 0.32
        info = f"doll_z={dc[2]:.3f} (落稳0.972) distXY={np.linalg.norm((dc-pr)[:2]):.3f}"
    elif name == "M2_drawer_open":
        ok = pull > 0.09; info = f"pull={pull:+.3f} (full=+0.12)"
    elif name == "M3_doll_in_drawer":
        oc = np.asarray(meta["open_center"])
        d_in = dc[:2] - oc[:2]
        ref = SETTLE_XY if SETTLE_XY is not None else np.array([2.729, -0.199])
        moved = np.linalg.norm(dc[:2] - ref) > 0.04   # 必须真被拖动过
        zhi = 0.92 if CAB_V3 else 1.12                     # v3: 必须真正落进斗内(台面0.95)
        ok = moved and abs(d_in[0]) < 0.32 and abs(d_in[1]) < 0.38 and 0.70 < dc[2] < zhi
        info = f"doll={np.round(dc,3)} open_c={np.round(oc[:2],3)} (巨型玩偶: 落在抽屉区即可)"
    elif name == "M4_drawer_closed":
        thr = -0.17 if CAB_V3 else -0.09
        ok = pull < thr; info = f"pull={pull:+.3f} (closed={-0.20 if CAB_V3 else -0.12}) doll={np.round(dc,3)}"
    elif name == "M5_home":
        # palm positions, not joint array reads (rj indexing into
        # get_revolute_current_angles() proved unreliable — visual home read 1.57)
        pr_, pl_ = palm("r"), palm("l")
        ok = pr_[2] < 0.95 and pl_[2] < 0.95 and pr_[0] < 2.52 and pl_[0] < 2.52
        info = f"palmR={np.round(pr_,2)} palmL={np.round(pl_,2)} doll_z={dc[2]:.3f}"
    mil[name] = (ok, info)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} @t={t:.1f}s  {info}")
    shot(f"{name}.png")

CHECKS = [(float(t), n) for t, n in meta["checks"]]
state_log = []
ci = 0; t0 = time.time()
servo_done = SERVO is None
servo2_done = SERVO2 is None
SERVO2B_T = None; TIPS_WANT = None; servo2b_done = False
latched = False; seated = False; latch_t0 = 0.0; latch_p0 = 0.0                                              # v3: rail detent after the push
keeper_on = False                                            # v3: rail friction during the doll drop
# jam watchdog: 1 Newton iter ~= 65-85ms; wall > 4s/step ~= 50+ iters (near cap).
# N consecutive "high" steps => fingers pressing the table / arm wedged in the
# drawer — abort early with diagnostics instead of grinding the whole run.
JAM_WALL = float(os.environ.get("S2_JAM_WALL", "4.0"))
JAM_N = int(os.environ.get("S2_JAM_N", "12"))
jam_run = 0; jam_total = 0; jam_ref = None
# trend watchdog: sustained step-cost escalation = contact anomaly (ride-up, wedge,
# contact-pair explosion) long before the iteration cap. Baseline = settle phase.
ESC_MULT = float(os.environ.get("S2_ESC_MULT", "2.5"))
ESC_ABS = float(os.environ.get("S2_ESC_ABS", "0.30"))
ESC_HOLD = float(os.environ.get("S2_ESC_HOLD", "6.0"))
esc_base = None; esc_samples = []
esc_ema = None; esc_since = None; esc_warn_t = -9.0
def jam_state():
    return np.r_[palm("r"), palm("l"), S["doll_c"](), S["drawer_pull"]()]
k = 0
aborted = None
while k * DT < T_END:
    t = k * DT
    if not servo_done and t >= SERVO["t"]:
        run_servo(); servo_done = True
    if CAB_V3 and servo_done and keeper_on is False and t >= keys[9][0]:
        try:
            eng.native.set_prismatic_strength(S["pj"], 2.0)   # keeper: rail friction eats the drop impact
            eng.set_prismatic_target(S["pj"], float(S["drawer_pull"]()))
        except Exception:
            pass
        keeper_on = True
        print(f"[keeper] 导轨摩擦保持 @t={t:.1f}s", flush=True)
    _koff = (meta.get("user_push") or {}).get("keeper_off_row", SERVO2["fire_row"] if SERVO2 else None)
    if CAB_V3 and keeper_on is True and _koff is not None and t >= keys[_koff][0]:
        _free = bool(meta.get("user_push"))                   # user demo was on a FREE rail
        try:
            eng.native.set_prismatic_strength(S["pj"], 0.0 if _free else 0.35)
        except Exception:
            pass
        keeper_on = "released" if _free else "damper"
        print(f"[keeper] {'释放为自由导轨' if _free else '转阻尼模式'} (pull={S['drawer_pull']():+.3f}) @t={t:.1f}s", flush=True)
    if CAB_V3 and keeper_on == "damper" and not latched:
        try:
            eng.set_prismatic_target(S["pj"], float(S["drawer_pull"]()))   # target follows current = viscous drag
        except Exception:
            pass
    if not servo2_done and t >= keys[SERVO2["fire_row"]][0] + 0.1:
        run_servo2(); servo2_done = True
    if CAB_V3 and servo2_done and not servo2b_done and SERVO2B_T is not None and t >= SERVO2B_T:
        run_servo2b(); servo2b_done = True
    _dthr = (meta.get("user_push") or {}).get("detent_thr", -0.125)
    if CAB_V3 and servo2_done and not latched and S["drawer_pull"]() <= _dthr:
        try:
            eng.native.set_prismatic_strength(S["pj"], 0.6)
        except Exception:
            pass
        latched = True
        latch_t0 = t; latch_p0 = float(S["drawer_pull"]())
        print(f"[rail-detent] 锁定 pull={latch_p0:+.3f} @t={t:.1f}s (1s斜坡滑至-0.195)", flush=True)
    if CAB_V3 and latched and not seated:
        w_ = min(1.0, (t - latch_t0) / 1.0)                   # ramped target: true 1s glide
        try:
            eng.set_prismatic_target(S["pj"], float(latch_p0 + (-0.195 - latch_p0) * w_))
        except Exception:
            pass
    if CAB_V3 and latched and not seated and S["drawer_pull"]() <= -0.185:
        try:
            eng.native.set_prismatic_strength(S["pj"], 1.5)   # seat-freeze: no spring rebound
            eng.set_prismatic_target(S["pj"], float(S["drawer_pull"]()))
        except Exception:
            pass
        seated = True
        print(f"[rail-detent] 坐定冻结 pull={S['drawer_pull']():+.3f} @t={t:.1f}s", flush=True)
    apply(t)
    tw0 = time.time()
    eng.step()
    tw = time.time() - tw0
    esc_ema = tw if esc_ema is None else 0.97 * esc_ema + 0.03 * tw
    if esc_base is None:
        if 2.0 <= t <= 8.0:
            esc_samples.append(tw)
        elif t > 8.0:
            esc_base = float(np.median(esc_samples)) if esc_samples else 0.12
            print(f"[watchdog] 基线步耗={esc_base*1000:.0f}ms (趋势阈值={max(ESC_ABS, ESC_MULT*esc_base)*1000:.0f}ms)", flush=True)
    elif aborted is None:
        if esc_ema > max(ESC_ABS, ESC_MULT * esc_base):
            if esc_since is None:
                esc_since = t
            if t - esc_warn_t > 2.0:
                esc_warn_t = t
                print(f"[watchdog] 耗时爬升 ema={esc_ema*1000:.0f}ms (基线{esc_base*1000:.0f}ms) "
                      f"持续{t-esc_since:.1f}s | palmR={np.round(palm('r'),3)} "
                      f"doll={np.round(S['doll_c'](),3)} pull={S['drawer_pull']():+.3f}", flush=True)
            if keeper_on in ("released", "damper") and ((t - esc_since > ESC_HOLD) or (esc_ema > 1.0 and t - esc_since > 1.0)):
                aborted = t
                print(f"[watchdog] ABORT @t={t:.1f}s: 步耗持续爬升{t-esc_since:.0f}s "
                      f"(ema {esc_ema*1000:.0f}ms ≈ {esc_ema/0.075:.0f}次迭代/步, 基线 {esc_base*1000:.0f}ms) — 接触异常, 提前终止", flush=True)
                shot("ABORT_escalation.png")
                break
        else:
            esc_since = None
    if tw > JAM_WALL:
        jam_run += 1; jam_total += 1
        st = jam_state()
        if jam_ref is None:
            jam_ref = st
        moved = float(np.abs(st - jam_ref).max())
        if jam_run == 3:
            print(f"[watchdog] 3连续高耗步 (>{JAM_WALL}s/步) @t={t:.1f}s Δ={moved*1000:.1f}mm | 碰撞体检: "
                  f"palmR={np.round(palm('r'),3)} palmL={np.round(palm('l'),3)} "
                  f"doll={np.round(S['doll_c'](),3)} pull={S['drawer_pull']():+.3f}", flush=True)
        if jam_run >= JAM_N:
            if moved < 0.004:            # 高耗 + 世界冻结 = 真卡死
                aborted = t
                print(f"[watchdog] ABORT @t={t:.1f}s: 连续{jam_run}步逼近迭代上限且零进展 "
                      f"(Δ={moved*1000:.1f}mm) — 手指压桌/卡入抽屉, 提前终止 (总高耗步 {jam_total})", flush=True)
                shot("ABORT_jam.png")
                break
            print(f"[watchdog] 高耗但在推进 (Δ={moved*1000:.1f}mm/{jam_run}步) — 继续", flush=True)
            jam_run = 0; jam_ref = None
    else:
        jam_run = 0; jam_ref = None
    if k % FRAME_EVERY == 0:
        state_log.append((t, np.asarray(eng.get_vertices()).copy()))
        if len(state_log) % 50 == 0:                          # crash-safe checkpoint (observation only)
            np.savez(os.path.join(OUT, "stage2_states_ckpt.npz"),
                     times=np.array([tt for tt, _ in state_log]),
                     verts=np.stack([v for _, v in state_log]).astype(np.float32),
                     faces=F, ta=ta, tb=tb)
    if ci < len(CHECKS) and t >= CHECKS[ci][0]:
        check(CHECKS[ci][1], t); ci += 1
    if k % 200 == 0:
        print(f"  step {k}/{int(T_END/DT)} t={t:.1f}s ({(time.time()-t0)/max(k,1)*1000:.0f} ms/step) "
              f"doll={np.round(S['doll_c'](),3)} pull={S['drawer_pull']():+.3f} 高耗步={jam_total}")
    k += 1
if aborted is None:
    while ci < len(CHECKS):
        check(CHECKS[ci][1], T_END); ci += 1
np.savez_compressed(os.path.join(OUT, "stage2_states.npz"),
                    times=np.array([t for t, _ in state_log]),
                    verts=np.stack([v for _, v in state_log]).astype(np.float32),
                    faces=F, ta=ta, tb=tb)
npass = sum(1 for ok, _ in mil.values() if ok)
tag = f"ABORTED@t={aborted:.1f}s" if aborted is not None else "DONE"
print(f"[urun] {tag} {npass}/{len(mil)} | states saved ({len(state_log)} frames) | 高耗步总计 {jam_total}")
