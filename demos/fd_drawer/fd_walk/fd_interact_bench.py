#!/usr/bin/env python3
"""Interactive COARSE-MESH physics bench (fd_walk demo tooling, v2).

Real engine, live stepping: pose either arm, watch the drawer/doll react.
Loads keyposes designed in fd_pose_designer.py (fd_keyposes.json) and applies
them as motor targets. REC snapshots to fw_user_push.json.

You drive the RIGHT arm + fingers; REAL physics runs live — if your hand
presses the drawer face, you will see `pull` go negative in the readout.
REC buttons append snapshots to fw_user_push.json (arm angles, fingers,
palm world position, drawer pull).

Controls:
  * per-joint sliders r1-r7 (deg) + palm IK panel (world XYZ, solves r1-r4)
  * fingers: grasp (all four) + thumb root / thumb curl
  * readout: palm world position + drawer pull (0=build 20cm-open, -0.20=closed)
  * REC: engage / push_mid / push_end / custom
"""
import sys, os, json
import numpy as np
os.environ.setdefault("FD_CAB_MESH", "/home/ps/Downloads/cabinet_v2_with_drawer_v02.obj")
os.environ.setdefault("FD_CAB_FRONT", "0.48")
os.environ.setdefault("FD_DOLL_SPOT", "2.836,-0.196")
os.environ.setdefault("FD_TSCALE", "1")
os.environ.setdefault("FD_DENSITY", "150")
os.environ.setdefault("FD_JSR", "300")
os.environ.setdefault("FD_FSTR", "12")
os.environ.setdefault("FD_ASTR", "4")
os.environ.setdefault("FD_YOUNG", "1e5")
sys.path.insert(0, "/home/ps/Downloads/FD-light/fd_walk")
from stage2_scene import build, FING_R, THUMB_R, FING_L, THUMB_L
from fk_dual import DualFK
from scipy.optimize import minimize

POSES = "/home/ps/Downloads/FD-light/fd_walk/fw_user_push.json"
fkd = DualFK()
S = build()
eng = S["eng"]; rj = S["rj"]; recs = S["recs"]

st = {
    "arm_r": {f"arm_r{i}_joint": 0.0 for i in range(1, 8)},   # deg
    "arm_l": {f"arm_l{i}_joint": 0.0 for i in range(1, 8)},
    "grasp_r": 0.0, "throot_r": -1.2, "thcurl_r": -0.8,
    "grasp_l": 0.0, "throot_l": -1.2, "thcurl_l": -0.8,
    "ikx": 2.55, "iky": -0.22, "ikz": 0.90,
    "kp_idx": 0,
}
KEYPOSES = []
_kp = "/home/ps/Downloads/FD-light/fd_walk/fd_keyposes.json"
if os.path.exists(_kp):
    KEYPOSES = json.load(open(_kp))
    print(f"[bench] fd_keyposes.json: {len(KEYPOSES)} poses", flush=True)

def apply_targets():
    for j, v in list(st["arm_r"].items()) + list(st["arm_l"].items()):
        if j in rj: eng.native.set_revolute_target(rj[j], float(np.radians(v)))
    for fn in FING_R:
        if fn in rj: eng.native.set_revolute_target(rj[fn], st["grasp_r"] * -1.9)
    for fn in FING_L:
        if fn in rj: eng.native.set_revolute_target(rj[fn], st["grasp_l"] * -1.9)
    if THUMB_R[0] in rj: eng.native.set_revolute_target(rj[THUMB_R[0]], st["throot_r"])
    for fn in THUMB_R[1:]:
        if fn in rj: eng.native.set_revolute_target(rj[fn], st["thcurl_r"])
    if THUMB_L[0] in rj: eng.native.set_revolute_target(rj[THUMB_L[0]], st["throot_l"])
    for fn in THUMB_L[1:]:
        if fn in rj: eng.native.set_revolute_target(rj[fn], st["thcurl_l"])

def load_keypose(i):
    k = KEYPOSES[i]
    for j, v in k["joints"].items():
        if j in st["arm_r"]: st["arm_r"][j] = float(np.degrees(v))
        if j in st["arm_l"]: st["arm_l"][j] = float(np.degrees(v))
    for a in ("grasp_r", "throot_r", "thcurl_r", "grasp_l", "throot_l", "thcurl_l"):
        if a in k["fingers"]: st[a] = float(k["fingers"][a])
    print(f"[bench] 应用关键帧 {i}:{k.get('name','')}", flush=True)

def ik_move():
    tgt = np.array([st["ikx"], st["iky"], st["ikz"]])
    wrist = {f"arm_r{i}_joint": float(np.radians(st["arm_r"][f"arm_r{i}_joint"])) for i in (5, 6, 7)}
    Jv = [f"arm_r{i}_joint" for i in (1, 2, 3, 4)]
    x0 = np.array([np.radians(st["arm_r"][j]) for j in Jv])
    def cost(x):
        ja = dict(zip(Jv, x)); ja.update(wrist)
        return np.sum((fkd.fk(ja)["base_link_r"][:3, 3] - tgt) ** 2) + 0.002 * np.sum(x ** 2)
    r = minimize(cost, x0, method="Nelder-Mead", options=dict(maxiter=2500, xatol=1e-4, fatol=1e-9))
    for j, v in zip(Jv, r.x):
        st["arm_r"][j] = float(np.degrees(np.clip(v, -1.57, 1.57)))

def palm():
    r = recs["base_link_r"]
    V = np.asarray(eng.get_vertices())
    return V[r.vertex_offset:r.vertex_offset + r.vertex_count].mean(0)

def rec(tag):
    entry = dict(tag=tag,
                 arm_r_deg={k: float(v) for k, v in st["arm_r"].items()},
                 grasp_r=float(st["grasp_r"]),
                 throot_r=float(st["throot_r"]), thcurl_r=float(st["thcurl_r"]),
                 pull=float(S["drawer_pull"]()),
                 palm_r=[float(x) for x in palm()])
    data = []
    if os.path.exists(POSES):
        data = json.load(open(POSES))
    data.append(entry)
    json.dump(data, open(POSES, "w"), indent=1)
    print(f"[REC] #{len(data)-1} {tag} pull={entry['pull']:+.3f} palm={np.round(entry['palm_r'],3)}", flush=True)

import polyscope as ps
import polyscope.imgui as psim
ps.set_program_name("fw_push_bench")
ps.init()
ps.set_up_dir("z_up"); ps.set_front_dir("neg_y_front"); ps.set_ground_plane_mode("shadow_only")
F = np.asarray(eng.get_surface_faces()); V0 = np.asarray(eng.get_vertices())
ta, tb = S["ta"], S["tb"]
dollF = F[np.all((F >= ta) & (F < tb), axis=1)]
restF = F[~np.all((F >= ta) & (F < tb), axis=1)]
sm = ps.register_surface_mesh("scene", V0, restF, color=(0.72, 0.78, 0.86), smooth_shade=True)
dm = ps.register_surface_mesh("doll", V0, dollF, color=(0.95, 0.45, 0.15), smooth_shade=True)
WAIST = S["WAIST"]; FAC = S["FACING"]; LEF = S["LEFTY"]
ps.look_at(tuple(WAIST + FAC * 0.2 - LEF * 1.8 + np.array([0, 0, 0.3])),
           tuple(WAIST + FAC * 0.4 + np.array([0, 0, -0.25])))

def ui():
    if KEYPOSES:
        ch, st["kp_idx"] = psim.SliderInt("keypose#", st["kp_idx"], 0, len(KEYPOSES) - 1)
        psim.SameLine()
        if psim.Button("apply designer keypose"):
            load_keypose(st["kp_idx"])
    psim.Text("RIGHT arm (deg)")
    for i in range(1, 8):
        j = f"arm_r{i}_joint"
        _, st["arm_r"][j] = psim.SliderFloat(f"r{i}", st["arm_r"][j], -90.0, 90.0)
    psim.Text("LEFT arm (deg)")
    for i in range(1, 8):
        j = f"arm_l{i}_joint"
        _, st["arm_l"][j] = psim.SliderFloat(f"l{i}", st["arm_l"][j], -90.0, 90.0)
    psim.Separator(); psim.Text("Palm IK (world XYZ, solves r1-r4, wrist fixed)")
    _, st["ikx"] = psim.SliderFloat("ik x", st["ikx"], 2.2, 3.1)
    _, st["iky"] = psim.SliderFloat("ik y", st["iky"], -0.6, 0.3)
    _, st["ikz"] = psim.SliderFloat("ik z", st["ikz"], 0.6, 1.3)
    if psim.Button("IK move"):
        ik_move()
    psim.Separator(); psim.Text("Fingers")
    _, st["grasp_r"] = psim.SliderFloat("grasp (0..1)", st["grasp_r"], -0.2, 1.0)
    _, st["throot_r"] = psim.SliderFloat("thumb root", st["throot_r"], -1.9, 1.9)
    _, st["thcurl_r"] = psim.SliderFloat("thumb curl", st["thcurl_r"], -1.9, 1.9)
    pr = palm()
    psim.Separator()
    psim.Text(f"palm=({pr[0]:.3f},{pr[1]:.3f},{pr[2]:.3f})   drawer pull={S['drawer_pull']():+.3f}")
    psim.Text("(pull: 0 = built 20cm-open, -0.20 = fully closed; goes negative as you push)")
    if psim.Button("REC engage"):
        rec("engage")
    psim.SameLine()
    if psim.Button("REC push_mid"):
        rec("push_mid")
    psim.SameLine()
    if psim.Button("REC push_end"):
        rec("push_end")
    psim.SameLine()
    if psim.Button("REC custom"):
        rec("custom")
    apply_targets()
    eng.step()
    sm.update_vertex_positions(np.asarray(eng.get_vertices()))
    dm.update_vertex_positions(np.asarray(eng.get_vertices()))

ps.set_user_callback(ui)
print("[bench] ready — window open. REC appends to fw_user_push.json", flush=True)
ps.show()
