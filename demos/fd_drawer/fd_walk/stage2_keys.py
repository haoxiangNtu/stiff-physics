#!/usr/bin/env python3
"""Stage-2 keyframes v5: side-shifted doll + proven-fast sagittal reach.

Right arm: hang -> pre1 (sagittal swing, doll out of sweep) -> pre2 (small adduct
over the doll) -> down2 (descend, cage) -> close -> lift -> transport -> release.
Left arm: handle poses (unchanged, empirically fast).  Doll at the EMPIRICALLY
calibrated fist-cage xy (2.772,-0.341).  Timeline auto-retimed (joint-rate cap).
"""
import sys, os, json
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/home/ps/Downloads/FD-light/fd_walk")
from fk_dual import DualFK

fk = DualFK()
CAB_TOP = 0.95
FACING = np.array([np.cos(-0.101823), np.sin(-0.101823), 0.0])
LEFTY = np.array([np.sin(0.101823), np.cos(-0.101823), 0.0])
WAIST = np.array([2.3313, 0.0359, 1.0500])
fc = WAIST + FACING * 0.32
dc = fc + FACING * 0.15
HANDLE0 = np.r_[(dc - FACING * (0.15 + 0.035) + LEFTY * 0.04)[:2], 0.76 + 0.105]
DRAWER_MAX = 0.24
OPEN_C = dc - FACING * DRAWER_MAX
CAGE = np.array([2.7718, -0.3411])            # calibrated fist-cage xy
DOLL = np.r_[CAGE, CAB_TOP + 0.022]

def ik(side, target, joints=None, seed=None):
    joints = joints or [f"arm_{side}{i}_joint" for i in (1, 2, 3, 4)]
    palm = f"base_link_{side}"
    x0 = np.array([(seed or {}).get(jn, 0.0) for jn in joints])
    def cost(x):
        W = fk.fk(dict(zip(joints, x)))
        return np.sum((W[palm][:3, 3] - target) ** 2) + 0.003 * np.sum(x ** 2)
    r = minimize(cost, x0, method="Nelder-Mead",
                 options=dict(maxiter=2500, xatol=1e-4, fatol=1e-9))
    ja = {jn: float(np.clip(v, -1.57, 1.57)) for jn, v in zip(joints, r.x)}
    W = fk.fk(ja); p = W[palm][:3, 3]
    return ja, p, float(np.linalg.norm(p - target))

poses = {}
def solve(name, side, target, **kw):
    ja, p, err = ik(side, np.asarray(target, float), **kw)
    poses[name] = ja
    print(f"  {name:10} err={err*100:5.1f}cm palm={np.round(p,3)}")
    return ja

print("=== v5 keyposes ===")
SP = json.load(open("/home/ps/Downloads/FD-light/fd_walk/side_poses.json"))
poses["pre1"] = SP["pre1"]; poses["pre2"] = SP["pre2"]; poses["down2"] = SP["down2"]
print("  pre1/pre2/down2 从 side_poses.json 载入(净空已验证)")
solve("down3", "r", np.r_[CAGE, 1.005], seed=poses["down2"])   # 压到玩偶顶(压-卷式)
solve("r_over", "r", np.r_[OPEN_C[:2], 1.12], seed=poses["pre2"])
L6 = [f"arm_l{i}_joint" for i in (1, 2, 3, 4, 5, 6)]
solve("l_handle", "l", HANDLE0 - FACING * 0.045, joints=L6)
solve("l_mid",   "l", HANDLE0 - FACING * (DRAWER_MAX / 2 + 0.045), joints=L6, seed=poses["l_handle"])
solve("l_pulled","l", HANDLE0 - FACING * (DRAWER_MAX + 0.045), joints=L6, seed=poses["l_mid"])

Z = {f"arm_{s}{i}_joint": 0.0 for s in ("r", "l") for i in range(1, 8)}
def merged(*names):
    d = dict(Z)
    for n in names:
        d.update(poses[n])
    return d

TL = [
    (0.0, merged(), 0.0, 0.0, 0.0),
    (0.6, merged(), 0.0, 0.0, 0.0),
    (2.4, merged("pre1"), 0.0, 0.0, 0.0),
    (3.4, merged("pre2"), 0.0, 0.0, 0.0),
    (4.4, merged("down2"), 0.0, 0.0, 0.0),
    (5.2, merged("down3"), 0.0, 0.0, 0.0),   # 下压贴玩偶顶
    (6.8, merged("down3"), 1.0, 0.0, 0.0),   # 压着慢卷指(拖进掌心)
    (7.4, merged("down3"), 1.0, 0.0, 0.0),
    (9.8, merged("pre2"), 1.0, 0.0, 0.0),   # 慢提
    (9.2, merged("pre2", "l_handle"), 1.0, 0.0, 0.0),
    (10.0, merged("pre2", "l_handle"), 1.0, 0.5, 0.0),
    (10.8, merged("pre2", "l_mid"), 1.0, 0.5, DRAWER_MAX / 2),
    (11.6, merged("pre2", "l_pulled"), 1.0, 0.5, DRAWER_MAX),
    (13.2, merged("r_over", "l_pulled"), 1.0, 0.5, DRAWER_MAX),
    (14.2, merged("r_over", "l_pulled"), 0.0, 0.5, DRAWER_MAX),
    (15.6, merged("pre2", "l_pulled"), 0.0, 0.5, DRAWER_MAX),
    (16.4, merged("l_mid"), 0.0, 0.5, DRAWER_MAX / 2),
    (17.2, merged("l_handle"), 0.0, 0.5, 0.0),
    (18.0, merged("l_handle"), 0.0, 0.0, 0.0),
    (19.2, merged(), 0.0, 0.0, 0.0),
    (20.0, merged(), 0.0, 0.0, 0.0),
]

RATE_J, RATE_D, RATE_G = 0.45, 0.12, 0.9
newTL = [TL[0]]
for i in range(1, len(TL)):
    t0, ja0, g0r, g0l, d0 = TL[i - 1]; t1, ja1, g1r, g1l, d1 = TL[i]
    dj = max(abs(ja1.get(n, 0) - ja0.get(n, 0)) for n in set(ja0) | set(ja1))
    need = max(t1 - t0, dj / RATE_J, abs(d1 - d0) / RATE_D,
               abs(g1r - g0r) / RATE_G, abs(g1l - g0l) / RATE_G)
    newTL.append((round(newTL[-1][0] + need, 2), ja1, g1r, g1l, d1))
TL = newTL
print(f"[keys] 重排后总时长 {TL[-1][0]:.1f}s")
CHK = [[TL[8][0] + 0.8, "M1_grasp_lift"], [TL[12][0] + 0.5, "M2_drawer_open"],
       [TL[14][0] + 1.0, "M3_doll_in_drawer"], [TL[17][0] + 0.5, "M4_drawer_closed"],
       [TL[-1][0] - 0.1, "M5_home"]]
SERVO = {"hover_key": 4,            # 到 down2 悬停后测量
         "adjust_keys": [5, 6, 7, 8],  # down3/close/hold/lift 的右臂目标重瞄
         "palm_down3": [float(CAGE[0]), float(CAGE[1]), 1.005],
         "planned_doll": DOLL.tolist()}
out = {"timeline": [[t, ja, gr, gl, dr] for (t, ja, gr, gl, dr) in TL],
       "meta": {"dt": 0.01, "doll_spot": DOLL.tolist(), "open_center": OPEN_C.tolist(),
                "drawer_max": DRAWER_MAX, "cab_top": CAB_TOP, "checks": CHK,
                "servo": SERVO}}
with open("/home/ps/Downloads/FD-light/fd_walk/stage2_timeline.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"[keys] {len(TL)} keys -> stage2_timeline.json  (DOLL_SPOT={np.round(DOLL,3)})")
