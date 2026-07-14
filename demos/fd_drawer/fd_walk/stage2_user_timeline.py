#!/usr/bin/env python3
"""Stage-2 timeline v8 — user poses UNSHIFTED + runtime lump-servo metadata.

User's path rules (2026-07-10):
  1. interp must clear the tabletop (raise high -> travel -> vertical descend)
  2. no doll contact before the fingers close (hover 22cm above, then straight down)
  3. same on the way back
Grasp targeting: the giant doll (0.62x1.0x0.59m) can never rest fully on the
0.40x0.90m cabinet top — it drapes over the right edge; only a lump stays on
top and its position varies run-to-run.  So the timeline keeps the user's
recorded poses VERBATIM and marks hold/lift/over rows for the runner's
measure-then-IK-shift servo (fires at the hover key, before any contact).

Writes stage2_user_timeline.json (full, from standing) and
       stage2_user_timeline_quick.json (starts at R_RAISED, for iteration)
plus   init_pose_raised.json (FD_INIT_POSE for the quick variant).
"""
import sys, os, json
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/home/ps/Downloads/FD-light/fd_walk")
from fk_dual import DualFK

fk = DualFK()
P = json.load(open("/home/ps/Downloads/FD-light/fd_walk/user_poses.json"))
GH, LFT = P[28], P[30]                       # grasp_hold / lift (user demo)
YAW = -0.101823
FACING = np.array([np.cos(YAW), np.sin(YAW), 0.0])
LEFTY = np.array([-np.sin(YAW), np.cos(YAW), 0.0])
WAIST = np.array([2.3313, 0.0359, 1.05])
CAB_FRONT = float(os.environ.get("FD_CAB_FRONT", "0.32"))
SHIFT = CAB_FRONT - 0.32                     # cabinet pushed away from the robot
fc = WAIST + FACING * CAB_FRONT
CAB_V3 = bool(os.environ.get('FD_CAB_MESH'))
if CAB_V3:
    OPEN_C = (fc - FACING * 0.06 - LEFTY * 0.2375)         # right drawer A center
elif os.environ.get('FD_CAB_V2', '0') == '1':
    OPEN_C = (fc - FACING * 0.06)
else:
    OPEN_C = (fc - FACING * 0.06 - LEFTY * 0.10)

def arm(e, side="r"):
    return {k: float(np.radians(v)) for k, v in e[f"arm_{side}_deg"].items()}

PRE = P[27]                                  # user's arrive-open pose (wrist 35.5°)
R_PRE = arm(PRE)                             # -> #28 = the roll-tuck-close move
R_HOLD, R_LIFT = arm(GH), arm(LFT)           # verbatim user poses

JR7 = [f"arm_r{i}_joint" for i in range(1, 8)]
def ik_full(seed, dpos):
    """7-joint IK: shift the palm by dpos, keep the palm ORIENTATION."""
    T0 = fk.fk(seed)["base_link_r"]
    tgt = T0[:3, 3] + dpos; R0 = T0[:3, :3]
    x0 = np.array([seed.get(j, 0.0) for j in JR7])
    def cost(x):
        M = fk.fk(dict(zip(JR7, x)))["base_link_r"]
        return (np.sum((M[:3, 3] - tgt) ** 2) + 0.05 * np.sum((M[:3, :3] - R0) ** 2)
                + 0.001 * np.sum((x - x0) ** 2))
    r = minimize(cost, x0, method="Nelder-Mead",
                 options=dict(maxiter=4000, xatol=1e-4, fatol=1e-10))
    sol = dict(seed)
    for j, v in zip(JR7, r.x):
        sol[j] = float(np.clip(v, -1.57, 1.57))
    M = fk.fk(sol)["base_link_r"]
    print(f"  [shift-ik] dpos={np.round(dpos,3)} pos_err={np.linalg.norm(M[:3,3]-tgt)*100:.1f}cm "
          f"rot_err={np.linalg.norm(M[:3,:3]-R0):.3f}")
    return sol

if abs(SHIFT) > 1e-6:                        # bake the cabinet shift into the grasp keys
    R_PRE = ik_full(R_PRE, FACING * SHIFT)
    R_HOLD = ik_full(R_HOLD, FACING * SHIFT)
def ikw(target, seed):                       # position IK, wrist r5-r7 pinned
    Jv = [f"arm_r{i}_joint" for i in (1, 2, 3, 4)]
    Jw = [f"arm_r{i}_joint" for i in (5, 6, 7)]
    wrist = {j: seed.get(j, 0.0) for j in Jw}
    x0 = np.array([seed.get(j, 0.0) for j in Jv])
    def cost(x):
        ja = dict(zip(Jv, x)); ja.update(wrist)
        return np.sum((fk.fk(ja)["base_link_r"][:3, 3] - target) ** 2) + 0.002 * np.sum(x ** 2)
    r = minimize(cost, x0, method="Nelder-Mead",
                 options=dict(maxiter=2500, xatol=1e-4, fatol=1e-9))
    out = dict(seed)
    for j, v in zip(Jv, r.x):
        out[j] = float(np.clip(v, -1.57, 1.57))
    ja = {j: out[j] for j in Jv}; ja.update(wrist)
    return out, float(np.linalg.norm(fk.fk(ja)["base_link_r"][:3, 3] - target))

p_hold = fk.fk(R_HOLD)["base_link_r"][:3, 3]
RIGHTY = -LEFTY
R_ABOVE, eA = ikw(p_hold + np.array([0, 0, 0.22]), R_HOLD)   # hover, no contact
# v15 (user path spec): swing the right arm OUT TO THE RIGHT first, travel to the
# hover point from the table's right side; reverse the same way after the drop.
R_SIDE1, eS1 = ikw(WAIST + RIGHTY * 0.45 + FACING * 0.08 + np.array([0, 0, 0.03]), R_HOLD)
R_SIDE2, eS2 = ikw(p_hold + RIGHTY * 0.16 + np.array([0, 0, 0.17]), R_HOLD)
R_UPRIGHT, eUR = ikw(p_hold - FACING * 0.06 + RIGHTY * 0.20 + np.array([0, 0, 0.15]), R_HOLD)
# v13: drag the caged ball just past the table edge and STOP — it falls into the
# bay by itself; fingers never below the tabletop, never into the drawer.
DRAG_EXT = 0.095 if CAB_V3 else 0.0                        # v03台面前挑9.1cm: 拖到新台缘
R_DRAG1, eD1 = ikw(p_hold - FACING * (0.05 + DRAG_EXT * 0.6) + LEFTY * 0.03, R_HOLD)
R_DRAG2, eD2 = ikw(p_hold - FACING * (0.065 + DRAG_EXT) + LEFTY * 0.03, R_HOLD)
for nm, pose in (("SIDE1", R_SIDE1), ("SIDE2", R_SIDE2), ("ABOVE", R_ABOVE),
                 ("UPRIGHT", R_UPRIGHT), ("DRAG1", R_DRAG1), ("DRAG2", R_DRAG2)):
    pp = fk.fk(pose)["base_link_r"][:3, 3]
    print(f"[v15] {nm:8s} palm_actual={np.round(pp,3)}")

def ikw_l(target, seed):                     # left-arm position IK, wrist pinned
    Jv = [f"arm_l{i}_joint" for i in (1, 2, 3, 4)]
    Jw = [f"arm_l{i}_joint" for i in (5, 6, 7)]
    wrist = {j: seed.get(j, 0.0) for j in Jw}
    x0 = np.array([seed.get(j, 0.0) for j in Jv])
    def cost(x):
        ja = dict(zip(Jv, x)); ja.update(wrist)
        return np.sum((fk.fk(ja)["base_link_l"][:3, 3] - target) ** 2) + 0.002 * np.sum(x ** 2)
    r = minimize(cost, x0, method="Nelder-Mead",
                 options=dict(maxiter=2500, xatol=1e-4, fatol=1e-9))
    out = dict(seed)
    for j, v in zip(Jv, r.x):
        out[j] = float(np.clip(v, -1.57, 1.57))
    ja = {j: out[j] for j in Jv}; ja.update(wrist)
    return out, float(np.linalg.norm(fk.fk(ja)["base_link_l"][:3, 3] - target))

R0 = {f"arm_r{i}_joint": 0.0 for i in range(1, 8)}
L0 = {f"arm_l{i}_joint": 0.0 for i in range(1, 8)}
L_C = [arm(P[i], "l") for i in (18, 20, 21, 22, 23)]
def lf(i):
    e = P[i]
    return dict(index_l=e.get("index_l", 0.0), fist_l=e.get("fist_l", 0.0),
                throot_l=e.get("throot_l", -1.2), thcurl_l=e.get("thcurl_l", -0.8))
LF_C = [lf(i) for i in (18, 20, 21, 22, 23)]
L_SEED = L_C[3]                              # #22 push pose = wrist seed
L_SIDE1, eL1 = ikw_l(WAIST + LEFTY * 0.45 + FACING * 0.08 + np.array([0, 0, 0.03]), L_SEED)
L_SIDE2, eL2 = ikw_l(np.array([2.47, 0.30, 1.08]), L_SEED)   # clear of the torso hull
for nm, pose in (("L_SIDE1", L_SIDE1), ("L_SIDE2", L_SIDE2)):
    pp = fk.fk(pose)["base_link_l"][:3, 3]
    print(f"[v16] {nm:8s} palm_actual={np.round(pp,3)}")
LZ = dict(index_l=0.0, fist_l=0.0, throot_l=-1.2, thcurl_l=-0.8)
RF = lambda g, tr, tc: dict(grasp_r=g, throot_r=tr, thcurl_r=tc)
G, TR, TC = GH["grasp_r"], GH["throot_r"], GH["thcurl_r"]
GP, TRP, TCP = PRE["grasp_r"], PRE["throot_r"], PRE["thcurl_r"]   # arrive-open (#27)
GL = LFT["grasp_r"]

# v3 right-close: bar (half-open) at W0+0.200F-0.247L z0.84; wrist seed = palm-forward push
HP_V3 = WAIST * np.array([1, 1, 0]) + FACING * 0.182 + LEFTY * (-0.247) + np.array([0, 0, 0.84])   # bar @rest开20cm
seedW_R = {**{f"arm_r{i}_joint": 0.0 for i in range(1, 8)},
           "arm_r5_joint": float(np.pi / 2), "arm_r6_joint": 0.5}
# USER DESIGN: keep the sweep hand (fingers forward, tips +0.151F/-0.042z from palm)
# and push the flat panel strip ABOVE the handle (tips z 0.885, strip 0.851-0.911).
# Palm trails the tips by 15cm -> palm never reaches the counter edge (F<=0.31 vs 0.389).

RF_OPEN = lambda: RF(1.0, -1.2, -0.8)                      # fist (engine-measured envelope)

ROWS = [                                                   # (t, r_arm, rf, l_arm, lfing)
    (0.0,  R0, RF(0, TR, TC), L0, LZ),                     # 0 stand, doll settles
    (2.0,  R0, RF(0, TR, TC), L0, LZ),                     # 1
    (3.4,  R_SIDE1, RF(0, TR, TC), L0, LZ),                # 2 swing out to the RIGHT side
    (4.8,  R_SIDE2, RF(0, TR, TC), L0, LZ),                # 3 travel up along the right
    (6.0,  R_ABOVE, RF(GP, TRP, TCP), L0, LZ),             # 4 hover      <- SERVO fires
    (7.4,  R_PRE,   RF(GP, TRP, TCP), L0, LZ),             # 5 descend to #27 (open, 35.5°)
    (8.0,  R_PRE,   RF(GP, TRP, TCP), L0, LZ),             # 6 dwell
    (9.4,  R_HOLD,  RF(G, TR, TC), L0, LZ),                # 7 roll-tuck-close (#27->#28)
    (10.2, R_HOLD,  RF(G, TR, TC), L0, LZ),                # 8 dwell             [M1]
    (11.8, R_DRAG1, RF(0.22, TR, TC), L0, LZ),             # 9 cage + drag 5cm inward
    (13.2, R_DRAG2, RF(0.22, TR, TC), L0, LZ),             # 10 drag just past edge, then STOP
    (14.2, R_DRAG2, RF(-0.05, TR, TC), L0, LZ),            # 11 four fingers open, THUMB HELD [M3]
    (15.4, R_DRAG2, RF(-0.05, -0.5, 0.3), L0, LZ),         # 12 ball is down -> thumb follows
    (16.6, R_UPRIGHT, RF(0, -1.2, -0.8), L0, LZ),          # 13 immediately up-right
    (17.8, R_SIDE2, RF(0, -1.2, -0.8), L0, LZ),            # 14 reverse along the right side
    (19.0, R_SIDE1, RF(0, -1.2, -0.8), L0, LZ),            # 15
    (20.4, R0, RF(0, -1.2, -0.8), L0, LZ),                 # 16 hang
    # left arm swings out to the LEFT side first (mirror of the right-arm path),
    # rows 18-21 RUNTIME-REPLACED by servo2 (descend far -> slide in at bar
    # height); 22 = measured push end, 23 = dwell, 24 = retract (= DOWN-FAR pose).
    (21.6, R0, RF(0, -1.2, -0.8), L_SIDE1, LF_C[3]),       # 17 swing out LEFT
    (22.8, R0, RF(0, -1.2, -0.8), L_SIDE2, LF_C[3]),       # 18 travel along the left
    (24.0, R0, RF(0, -1.2, -0.8), L_C[3], LF_C[3]),        # 19 -> servo2 HIGH (far)
    (25.2, R0, RF(0, -1.2, -0.8), L_C[3], LF_C[3]),        # 20 -> servo2 DOWN-FAR
    (26.2, R0, RF(0, -1.2, -0.8), L_C[3], LF_C[3]),        # 21 -> servo2 FRONT
    (27.2, R0, RF(0, -1.2, -0.8), L_C[3], LF_C[3]),        # 22 -> servo2 ENGAGE
    (28.4, R0, RF(0, -1.2, -0.8), L_C[4], LF_C[4]),        # 23 push closed (measured)
    (29.4, R0, RF(0, -1.2, -0.8), L_C[4], LF_C[4]),        # 24 push dwell       [M4]
    (30.6, R0, RF(0, -1.2, -0.8), L_C[3], LF_C[3]),        # 25 retract (servo2=DOWN-FAR)
    (31.8, R0, RF(0, -1.2, -0.8), L_SIDE2, LZ),            # 26 back along the left
    (33.0, R0, RF(0, -1.2, -0.8), L_SIDE1, LZ),            # 27
    (34.4, R0, RF(0, -1.2, -0.8), L0, LZ),                 # 28 home             [M5]
    (35.2, R0, RF(0, -1.2, -0.8), L0, LZ),                 # 29
]

def build(rows, servo_row, servo_shift_rows, m_rows, path, s2_fire=12, s2_rows=(13, 14, 15), s2_push=16, s2_retract=None):
    TL = []
    for t, ra, rf_, la, lf_ in rows:
        ja = {}; ja.update(ra); ja.update(la)
        f = {}; f.update(rf_); f.update(lf_)
        TL.append((t, ja, f))
    RATE = 0.45
    new = [TL[0]]
    for i in range(1, len(TL)):
        t0, j0, f0 = TL[i - 1]; t1, j1, f1 = TL[i]
        dj = max(abs(j1.get(n, 0) - j0.get(n, 0)) for n in set(j0) | set(j1))
        dg = max(abs(f1[k] - f0[k]) for k in f1)
        new.append((round(new[-1][0] + max(t1 - t0, dj / RATE, dg / 0.9), 2), j1, f1))
    TL = new
    m1_off = 2.0 if CAB_V3 else 1.0
    SPEC = [["M1_grasp_lift", m_rows[0], m1_off], ["M3_doll_in_drawer", m_rows[1], 1.2],
            ["M4_drawer_closed", m_rows[2], 0.4], ["M5_home", -1, -0.1]]
    CHK = [[TL[r][0] + off, n] for n, r, off in SPEC]
    out = {"timeline_user": [[t, ja, f] for t, ja, f in TL],
           "meta": {"dt": 0.01, "open_center": OPEN_C.tolist(), "cab_top": 0.95,
                    "drawer_half": 0.12, "checks": CHK, "check_spec": SPEC,
                    "servo": {"t": TL[servo_row][0] + 0.15,      # after hover reached
                              "rows": servo_shift_rows,
                              "hover_row": servo_shift_rows[0],  # gets +16cm z target
                              # reference = expected doll position (user demo ball);
                              # measured lump - this = drift to shift the grasp keys by
                              "palm_xy": [2.677 + float(FACING[0]) * SHIFT, -0.180 + float(FACING[1]) * SHIFT]},
                    "servo2": {"fire_row": s2_fire, "rows": list(s2_rows),
                               "push_row": s2_push, "retract_row": s2_retract}}}
    json.dump(out, open(path, "w"), indent=1)
    print(f"[v8] {os.path.basename(path)}: {len(TL)} keys {TL[-1][0]:.1f}s "
          f"servo@t={out['meta']['servo']['t']:.2f} M1@{CHK[0][0]:.1f}s")
    return TL

RF_FIST = lambda: RF(1.0, -1.2, -0.8)
R_NUDGE, eN = ikw(p_hold - FACING * 0.21 + LEFTY * 0.03 - np.array([0, 0, 0.005]), R_DRAG2)
R_NUDGE2, eN2 = ikw(p_hold - FACING * 0.25 + LEFTY * 0.03 - np.array([0, 0, 0.005]), R_NUDGE)   # 平扫保险(不再下压)

seedS = {f"arm_r{i}_joint": 0.0 for i in range(1, 8)}
for _i in (5, 6, 7):
    seedS[f"arm_r{_i}_joint"] = R_NUDGE2[f"arm_r{_i}_joint"]
# USER PATH v4: lift OUT of the mouth first -> transit HIGH to beside-the-handle ->
# vertical descend (hanging fingers) -> straight slow push. All pose changes happen
# 20cm above the rim. HP_P4 = palm ref: tips (palm-0.078L,-0.155z) at (barF-0.011,
# bar_lat-0.137, 0.845) = panel band right of the bar.
sweep_palm = fk.fk({k: v for k, v in R_NUDGE2.items()})["base_link_r"][:3, 3]
R_LIFTX, eLX = ikw(sweep_palm + np.array([0, 0, 0.16]), R_NUDGE2)         # straight lift out
R_TRH, eTH = ikw(HP_V3 - FACING * 0.13 + np.array([0, 0, 0.14]), seedW_R)
R_TRD, eTD = ikw(HP_V3 - FACING * 0.13, R_TRH)
R_PM2, ePM = ikw(HP_V3 - FACING * 0.12, R_TRD)
R_PE2, ePE = ikw(HP_V3 + FACING * 0.11, R_PM2)
if CAB_V3:
    print(f"[v4] 占位 lift={eLX*100:.1f} errs={eTH*100:.1f}/{eTD*100:.1f}/{ePM*100:.1f}/{ePE*100:.1f}cm")
if CAB_V3:
    print(f"[v3] NUDGE err={eN*100:.1f}cm NUDGE2 err={eN2*100:.1f}cm")
ROWS_V3 = ROWS[:9] + [
    (11.8, R_DRAG1, RF(0.30, TR, TC), L0, LZ),             # 9 cage(加深) + drag
    (13.6, R_DRAG2, RF(0.30, TR, TC), L0, LZ),             # 10 long drag to the new edge
    (14.6, R_DRAG2, RF(-0.05, TR, TC), L0, LZ),            # 11 four fingers open (thumb held)
    (15.6, R_NUDGE, RF(-0.05, -0.5, 0.3), L0, LZ),         # 12 nudge: sweep ball over the edge
    (16.8, R_NUDGE2, RF(-0.05, -0.5, 0.3), L0, LZ),        # 13 press-through [M3]  <- servo2R fires
    (19.0, R_LIFTX, RF(-0.05, -0.5, 0.3), L0, LZ),         # 14 lift straight OUT of the mouth (+16cm)
    (21.6, R_TRH,   RF_OPEN(), L0, LZ),                    # 15 transit HIGH to beside-the-handle
    (24.2, R_TRD,   RF_OPEN(), L0, LZ),                    # 16 vertical descend (tips beside the bar)
    (27.0, R_TRD,   RF_OPEN(), L0, LZ),                    # 17 dwell: converge   <- servo2B measures
    (30.5, R_PM2,   RF_OPEN(), L0, LZ),                    # 18 slow push, first half
    (34.0, R_PE2,   RF_OPEN(), L0, LZ),                    # 19 slow push, second half
    (35.2, R_PE2,   RF_OPEN(), L0, LZ),                    # 20 dwell         [M4]
    (37.2, R_TRD,   RF_OPEN(), L0, LZ),                    # 21 retract straight back
    (39.4, R0,      RF(0, -1.2, -0.8), L0, LZ),            # 22 home          [M5]
    (40.2, R0,      RF(0, -1.2, -0.8), L0, LZ),            # 23
]
USER_PUSH = "/home/ps/Downloads/FD-light/fd_walk/fw_user_push.json"
if CAB_V3 and os.path.exists(USER_PUSH):
    # v5: the user's own recorded push (bench = real engine, zero model bias).
    # Five frames share r2-r7; only r1 sweeps -18..-43 deg — a single-joint arc.
    UP = json.load(open(USER_PUSH))
    def up_arm(e, r1_override=None):
        ja = {k: float(np.radians(v)) for k, v in e["arm_r_deg"].items()}
        if r1_override is not None:
            ja["arm_r1_joint"] = float(np.radians(r1_override))
        return ja
    def up_fn(e):
        return dict(grasp_r=float(e["grasp_r"]), throot_r=float(e["throot_r"]),
                    thcurl_r=float(e["thcurl_r"]), index_l=0.0, fist_l=0.0,
                    throot_l=-1.2, thcurl_l=-0.8)
    U_PRE = up_arm(UP[0], r1_override=5.0)                    # their family, hand by the thigh
    ROWS_V5 = ROWS[:9] + [
        (11.8, R_DRAG1, RF(0.40, TR, TC), L0, LZ),             # 9 cage(收紧) + drag
        (13.6, R_DRAG2, RF(0.40, TR, TC), L0, LZ),             # 10 long drag (玩偶握持到位)
        (15.2, R_DRAG2, RF(0.40, -0.5, 0.3), L0, LZ),          # 11 拇指先缓撤(笼仍扣,臂静止,不弹飞)
        (17.4, R_DRAG2, RF(0.15, -0.5, 0.3), L0, LZ),          # 12 笼缓松0.40->0.15: 减压落台,笼壁挡横挤
        (18.8, R_DRAG2, RF(-0.05, -0.5, 0.3), L0, LZ),         # 13 四指全开 -> 玩偶已静置
        (19.8, R_NUDGE, RF(-0.05, -0.5, 0.3), L0, LZ),         # 14 sweep
        (21.0, R_NUDGE2, RF(-0.05, -0.5, 0.3), L0, LZ),        # 15 press-through [M3]
        (22.8, R_UPRIGHT, RF(0, -1.2, -0.8), L0, LZ),          # 16 up-right (proven exit)
        (24.8, R_SIDE2,  RF(0, -1.2, -0.8), L0, LZ),           # 17 swing right-high
        (26.8, R_SIDE1,  RF(0, -1.2, -0.8), L0, LZ),           # 18 side-low
        (28.8, R0,       RF(0, -1.2, -0.8), L0, LZ),           # 19 hang
        (30.8, U_PRE,          up_fn(UP[0]), L0, LZ),          # 20 family r1=+5 (by the thigh)
        (32.6, up_arm(UP[0]),  up_fn(UP[0]), L0, LZ),          # 21 r1 -18: engage (contact)
        (33.8, up_arm(UP[1]),  up_fn(UP[1]), L0, LZ),          # 22 r1 -23
        (35.0, up_arm(UP[2]),  up_fn(UP[2]), L0, LZ),          # 23 r1 -28
        (36.2, up_arm(UP[3]),  up_fn(UP[3]), L0, LZ),          # 24 r1 -33
        (37.6, up_arm(UP[4]),  up_fn(UP[4]), L0, LZ),          # 25 r1 -43: push end
        (38.6, up_arm(UP[4]),  up_fn(UP[4]), L0, LZ),          # 26 dwell        [M4]
        (40.4, up_arm(UP[1]),  up_fn(UP[1]), L0, LZ),          # 27 retract along the arc
        (42.2, U_PRE,          up_fn(UP[0]), L0, LZ),          # 28 back to thigh
        (44.0, R0, RF(0, -1.2, -0.8), L0, LZ),                 # 29 home         [M5]
        (44.8, R0, RF(0, -1.2, -0.8), L0, LZ),                 # 30
    ]
    TLv5 = []
    for t, ra, rf_, la, lf_ in ROWS_V5:
        ja = {}; ja.update(ra); ja.update(la)
        f = {}; f.update(rf_); f.update(lf_)
        TLv5.append((t, ja, f))
    RATE = 0.45
    newt = [TLv5[0]]
    for i in range(1, len(TLv5)):
        t0, j0, f0 = TLv5[i - 1]; t1, j1, f1 = TLv5[i]
        dj = max(abs(j1.get(n, 0) - j0.get(n, 0)) for n in set(j0) | set(j1))
        dg = max(abs(f1[k] - f0[k]) for k in f1)
        newt.append((round(newt[-1][0] + max(t1 - t0, dj / RATE, dg / 0.9), 2), j1, f1))
    TLv5 = newt
    SPECv5 = [["M1_grasp_lift", 8, 2.0], ["M3_doll_in_drawer", 15, 1.2],
              ["M4_drawer_closed", 26, 0.6], ["M5_home", -1, -0.1]]
    CHKv5 = [[TLv5[r][0] + off, n] for n, r, off in SPECv5]
    out = {"timeline_user": [[t, ja, f] for t, ja, f in TLv5],
           "meta": {"dt": 0.01, "open_center": OPEN_C.tolist(), "cab_top": 0.95,
                    "drawer_half": 0.12, "checks": CHKv5, "check_spec": SPECv5,
                    "servo": {"t": TLv5[4][0] + 0.15, "rows": [5, 6, 7, 8, 9, 10, 11, 12],
                              "hover_row": 5,
                              "palm_xy": [2.677 + float(FACING[0]) * SHIFT, -0.180 + float(FACING[1]) * SHIFT]},
                    "user_push": {"keeper_off_row": 20, "detent_thr": -0.155}}}
    json.dump(out, open("/home/ps/Downloads/FD-light/fd_walk/stage2_user_timeline.json", "w"), indent=1)
    print(f"[v5] 用户推关时间线: {len(TLv5)} keys {TLv5[-1][0]:.1f}s M4@{CHKv5[2][0]:.1f}s")
elif CAB_V3:
    build(ROWS_V3, servo_row=4, servo_shift_rows=[5, 6, 7, 8, 9, 10, 11, 12], m_rows=(8, 13, 20),
          s2_fire=13, s2_rows=(15, 16, 17, 18), s2_push=19, s2_retract=21,
          path="/home/ps/Downloads/FD-light/fd_walk/stage2_user_timeline.json")
else:
    build(ROWS, servo_row=4, servo_shift_rows=[5, 6, 7, 8, 9, 10, 11, 12], m_rows=(8, 11, 24), s2_fire=18, s2_rows=(19, 20, 21, 22), s2_push=23, s2_retract=25,
          path="/home/ps/Downloads/FD-light/fd_walk/stage2_user_timeline.json")

# quick variant: start AT R_RAISED (skip stand->raised swing), same rows after
QROWS = [(0.0, R_SIDE2, RF(0, TR, TC), L0, LZ),
         (2.0, R_SIDE2, RF(0, TR, TC), L0, LZ)] + \
        [(t - 2.8, ra, rf_, la, lf_) for (t, ra, rf_, la, lf_) in ROWS[4:]]
build(QROWS, servo_row=2, servo_shift_rows=[3, 4, 5, 6, 7, 8, 9, 10], m_rows=(6, 9, 22), s2_fire=16, s2_rows=(17, 18, 19, 20), s2_push=21, s2_retract=23,
      path="/home/ps/Downloads/FD-light/fd_walk/stage2_user_timeline_quick.json")

ip = {k: float(v) for k, v in R_SIDE2.items()}
for i in range(1, 8):
    ip[f"arm_l{i}_joint"] = 0.0
json.dump(ip, open("/home/ps/Downloads/FD-light/fd_walk/init_pose_raised.json", "w"), indent=1)
print("[v8] init_pose_raised.json written")
