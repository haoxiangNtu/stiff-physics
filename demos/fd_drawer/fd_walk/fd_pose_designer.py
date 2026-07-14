#!/usr/bin/env python3
"""FD keypose designer — pure forward kinematics, NO physics engine.

Design / inspect / edit the robot's key poses for every demo stage (grasp,
place, push, walk, turn) with instant visual feedback:

  * full joint control: both arms (r1-7 / l1-7), grouped finger channels
    (grasp / thumb-root / thumb-curl per hand), legs, waist/head extras
  * base position (x, y, z) + yaw — place the robot anywhere
  * KEYFRAMES: add / update / delete / reorder / apply; FK-interpolated
    playback with the same 0.45 rad/s rate rule the demo timeline uses
  * import an existing stage2 timeline json (rows become keyframes) —
    tweak a row, save back out as a *_edited.json
  * STAGE scrubber: movie-accurate S1 walk-in / S3 back-step / S4 turn+out —
    rebuilt with the composite's base offsets (OFFB/drift/OFF4), so frames
    match the mp4 exactly (no cabinet penetration); capture any frame
  * static context meshes (cabinet OBJ + doll) for spatial reference

Run:  python3.11 fd_pose_designer.py            (needs a display)
Env:  FD_CAB_MESH=<obj>  cabinet context mesh (default: v02 two-drawer)
"""
import sys, os, json, copy
import numpy as np
sys.path.insert(0, "/home/ps/Downloads/FD-light/fd_walk")
from fd_walk_render import URDFModel, T, URDF
from scipy.spatial.transform import Rotation as Rot
import xml.etree.ElementTree as ET

WALK = "/home/ps/Downloads/FD-light/fd_walk"
CAB_MESH = os.environ.get("FD_CAB_MESH", "/home/ps/Downloads/cabinet_v2_with_drawer_v02.obj")
TOY = os.environ.get("FD_TOY", "/home/ps/Downloads/geo/Toy00.obj")
POSE_FILE = os.path.join(WALK, "fd_keyposes.json")
WAIST = np.array([2.3313, 0.0359, 1.0500])
YAW0 = -0.101823

# ---- URDF joints + limits --------------------------------------------------
_root = ET.parse(URDF).getroot()
LIMITS = {}
for j in _root.findall("joint"):
    if j.get("type") == "revolute":
        l = j.find("limit")
        LIMITS[j.get("name")] = (float(l.get("lower", -1.57)), float(l.get("upper", 1.57))) \
            if l is not None else (-1.57, 1.57)
ARM_R = [f"arm_r{i}_joint" for i in range(1, 8)]
ARM_L = [f"arm_l{i}_joint" for i in range(1, 8)]
LEG_R = sorted(j for j in LIMITS if j.startswith("leg_r"))
LEG_L = sorted(j for j in LIMITS if j.startswith("leg_l"))
MISC = sorted(j for j in LIMITS if not (j.startswith(("arm_", "leg_")) or "_link_" in j or j.startswith(("if_", "mf_", "rf_", "lf_", "th_"))))
FING_R = [f + s for f in ("if_", "mf_", "rf_", "lf_") for s in ("proximal_link_r", "distal_link_r")]
FING_L = [f + s for f in ("if_", "mf_", "rf_", "lf_") for s in ("proximal_link_l", "distal_link_l")]
THUMB_R = ["th_root_link_r", "th_proximal_link_r", "th_distal_link_r"]
THUMB_L = ["th_root_link_l", "th_proximal_link_l", "th_distal_link_l"]

model = URDFModel(URDF)

st = {
    "base": [float(WAIST[0]), float(WAIST[1]), float(WAIST[2]), float(YAW0)],
    "joints": {j: 0.0 for j in ARM_R + ARM_L + LEG_R + LEG_L + MISC},   # rad
    "fingers": {"grasp_r": 0.0, "throot_r": -1.2, "thcurl_r": -0.8,
                "grasp_l": 0.0, "throot_l": -1.2, "thcurl_l": -0.8},
    "keys": [],                 # [{name, base, joints, fingers}]
    "sel": -1,
    "play": False, "play_t": 0.0,
    "stage_sel": 0, "stage_t": 0, "stage_lab": "",
    "show_ctx": True,
    "tl_rows": None, "tl_meta": None, "tl_path": "", "tl_sel": 0,
    "tl_a": 0, "tl_b": 0, "tl_compact": False, "tl_msg": "",
}

def full_ja():
    ja = dict(st["joints"])
    f = st["fingers"]
    for fn in FING_R: ja[fn] = f["grasp_r"] * -1.9
    for fn in FING_L: ja[fn] = f["grasp_l"] * -1.9
    ja[THUMB_R[0]] = f["throot_r"]
    for fn in THUMB_R[1:]: ja[fn] = f["thcurl_r"]
    ja[THUMB_L[0]] = f["throot_l"]
    for fn in THUMB_L[1:]: ja[fn] = f["thcurl_l"]
    return ja

def base_T():
    x, y, z, yaw = st["base"]
    return T(Rot.from_euler("z", yaw).as_matrix(), np.array([x, y, z]))

# ---- gait ------------------------------------------------------------------
GAIT = None
try:
    d = np.load(os.path.join(WALK, "gait_traj.npz"), allow_pickle=True)
    GAIT = dict(names=list(d["joint_names"]), q=d["joint_angles"],
                bp=d["base_pos"], bq=d["base_quat_wxyz"])
except Exception as e:
    print("[designer] gait_traj.npz 不可用:", e)

# ---- movie-accurate stage sequences ----------------------------------------
# gait_traj.npz is the RAW gait path; the composite translates S3/S4 (OFFB /
# drift / OFF4) so the turn clears the cabinet. Rebuild the exact same frame
# sequences here so scrubbing == what the mp4 shows.
ARM_SWING = 0.45

def _b4(p, quat_wxyz):
    w, x, y, z = quat_wxyz
    return [float(p[0]), float(p[1]), float(p[2]),
            float(Rot.from_quat([x, y, z, w]).as_euler("zyx")[0])]

def jang_gait(t, arm_override=None):
    ja = {GAIT["names"][i]: float(GAIT["q"][t, i]) for i in range(len(GAIT["names"]))}
    if arm_override in ("swing", "zero"):              # non-leg joints -> 0
        for n in list(ja):
            if not n.startswith("leg_"): ja[n] = 0.0
    if arm_override == "swing":                        # natural swing
        ja["arm_r1_joint"] = ARM_SWING * ja.get("leg_l1_joint", 0.0)
        ja["arm_l1_joint"] = -ARM_SWING * ja.get("leg_r1_joint", 0.0)
    return ja

STAND = [float(WAIST[0]), float(WAIST[1]), 0.9116, float(YAW0)]
STAGES, STAGE_KEYS = {}, []
if GAIT is not None:
    bp, bq = GAIT["bp"], GAIT["bq"]
    STAND = _b4(bp[178], bq[178])          # standing at the cabinet = gait 178
    L_ = np.array([-np.sin(YAW0), np.cos(YAW0), 0.0])
    RIGHTY = -L_
    # S1: walk in — raw gait replay, arms from the gait itself
    STAGES["S1 walk-in"] = [(f"walk t{t}", _b4(bp[t], bq[t]), jang_gait(t))
                            for t in range(0, 179)]
    # S3: single right-foot back-step (composite stage 3, frame-exact)
    BS0, BS1, DRIFT = 136, 123, 0.15
    OFFB = bp[178] - bp[BS0]
    fr = []
    jaA, jaB = jang_gait(178, "zero"), jang_gait(BS0, "zero")
    for i in range(6):                                  # stand -> lift pose
        w = (i + 1) / 6.0
        ja = {n: (1 - w) * jaA.get(n, 0.0) + w * jaB.get(n, 0.0) for n in set(jaA) | set(jaB)}
        fr.append((f"lift {i+1}/6", _b4(bp[BS0] + OFFB, bq[BS0]), ja))
    n_bs = BS0 - BS1 + 1
    for k, t in enumerate(range(BS0, BS1 - 1, -1)):     # reversed swing + right drift
        drift = RIGHTY * (DRIFT * (k + 1) / n_bs)
        fr.append((f"back-swing t{t}", _b4(bp[t] + OFFB + drift, bq[t]), jang_gait(t, "zero")))
    base_end = bp[BS1] + OFFB + RIGHTY * DRIFT
    jaA, jaSTAND = jang_gait(BS1, "zero"), jang_gait(178, "zero")
    for i in range(8):                                  # foot down -> attention
        w = (i + 1) / 8.0
        ja = {n: (1 - w) * jaA.get(n, 0.0) + w * jaSTAND.get(n, 0.0) for n in set(jaA) | set(jaSTAND)}
        fr.append((f"settle {i+1}/8", _b4(base_end, bq[178]), ja))
    jaB = jang_gait(259, "zero")
    for i in range(8):                                  # attention -> turn start
        w = (i + 1) / 8.0
        ja = {n: (1 - w) * jaSTAND.get(n, 0.0) + w * jaB.get(n, 0.0) for n in set(jaSTAND) | set(jaB)}
        fr.append((f"to-turn {i+1}/8", _b4(base_end, bq[259]), ja))
    STAGES["S3 back-step"] = fr
    # S4: turn right + walk away, whole segment translated by OFF4
    OFF4 = base_end - bp[259]
    STAGES["S4 turn+out"] = [(f"turn t{t}", _b4(bp[t] + OFF4, bq[t]), jang_gait(t, "swing"))
                             for t in range(259, 419)]
    STAGE_KEYS = list(STAGES)

def apply_stage(si, fi):
    lab, b4, ja = STAGES[STAGE_KEYS[si]][fi]
    st["base"] = list(b4)
    for k in st["joints"]: st["joints"][k] = float(ja.get(k, 0.0))
    for k in st["fingers"]: st["fingers"][k] = 0.0      # film hands = URDF zero (open)
    return lab

st["base"] = list(STAND)

# ---- keyframes ---------------------------------------------------------------
def snap(name):
    return dict(name=name, base=list(st["base"]),
                joints={k: float(v) for k, v in st["joints"].items()},
                fingers={k: float(v) for k, v in st["fingers"].items()})

def apply_key(k):
    st["base"] = list(k["base"])
    st["joints"].update(k["joints"])
    st["fingers"].update(k["fingers"])

def save_keys(path=POSE_FILE):
    json.dump(st["keys"], open(path, "w"), indent=1)
    print(f"[designer] {len(st['keys'])} keyposes -> {path}", flush=True)

def load_keys(path=POSE_FILE):
    if os.path.exists(path):
        st["keys"] = json.load(open(path))
        print(f"[designer] loaded {len(st['keys'])} keyposes", flush=True)

# ---- timeline import/export --------------------------------------------------
def load_timeline(path):
    tl = json.load(open(path))
    st["tl_rows"] = tl["timeline_user"]; st["tl_meta"] = tl.get("meta", {})
    st["tl_path"] = path; st["tl_sel"] = 0
    print(f"[designer] timeline {len(st['tl_rows'])} rows loaded", flush=True)

def apply_tl_row(i):
    t, ja, f = st["tl_rows"][i]
    for k in st["joints"]:
        st["joints"][k] = float(ja.get(k, 0.0))
    fmap = {"grasp_r": "grasp_r", "throot_r": "throot_r", "thcurl_r": "thcurl_r",
            "throot_l": "throot_l", "thcurl_l": "thcurl_l"}
    for a, b in fmap.items():
        if b in f: st["fingers"][a] = float(f[b])
    if "fist_l" in f: st["fingers"]["grasp_l"] = float(f["fist_l"])
    st["base"] = list(STAND)

def store_tl_row(i):
    t, ja, f = st["tl_rows"][i]
    for k, v in st["joints"].items():
        if k.startswith("arm_"):
            ja[k] = float(v)
    f["grasp_r"] = st["fingers"]["grasp_r"]; f["throot_r"] = st["fingers"]["throot_r"]
    f["thcurl_r"] = st["fingers"]["thcurl_r"]
    f["fist_l"] = st["fingers"]["grasp_l"]; f["index_l"] = st["fingers"]["grasp_l"]
    f["throot_l"] = st["fingers"]["throot_l"]; f["thcurl_l"] = st["fingers"]["thcurl_l"]
    print(f"[designer] row {i} 已更新", flush=True)

def save_timeline():
    if not st["tl_rows"]: return
    out = {"timeline_user": st["tl_rows"], "meta": st["tl_meta"]}
    p = st["tl_path"].replace(".json", "_edited.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"[designer] timeline -> {p}", flush=True)

def tl_protected():
    """rows referenced by meta (milestones / servo / keeper) — deleting them
    would silently break the run, so they are load-bearing."""
    meta = st["tl_meta"] or {}
    n = len(st["tl_rows"])
    prot = {0: "start pose"}
    for nme, r, _off in meta.get("check_spec", []):
        prot[r % n] = f"milestone {nme}"
    sv = meta.get("servo") or {}
    for r in sv.get("rows", []): prot[r % n] = "servo scoop"
    if sv.get("hover_row") is not None: prot[sv["hover_row"] % n] = "servo hover"
    up = meta.get("user_push") or {}
    if up.get("keeper_off_row") is not None:
        prot[up["keeper_off_row"] % n] = "keeper release"
    return prot

def tl_delete(a, b, compact):
    """Delete rows a..b inclusive. Refuses load-bearing rows; remaps every
    row index stored in meta; optionally compacts the time gap to 2s (the
    executor's rate-limit retime stretches it back if the joint delta needs
    more, so 2s is a safe floor, not a hard duration)."""
    rows = st["tl_rows"]; n = len(rows)
    a, b = int(a), int(b)
    if not (0 <= a <= b < n):
        st["tl_msg"] = f"invalid range {a}..{b}"; return
    if b - a + 1 >= n:
        st["tl_msg"] = "refused: cannot delete every row"; return
    prot = tl_protected()
    hit = [f"row{r}({prot[r]})" for r in range(a, b + 1) if r in prot]
    if hit:
        st["tl_msg"] = "refused, load-bearing: " + ", ".join(hit)
        print("[designer]", st["tl_msg"], flush=True); return
    ndel = b - a + 1
    shift = 0.0
    if compact and a > 0 and b + 1 < n:
        shift = max((rows[b + 1][0] - rows[a - 1][0]) - 2.0, 0.0)
    del rows[a:b + 1]
    if shift:
        for i in range(a, len(rows)):
            rows[i][0] = round(rows[i][0] - shift, 2)

    def remap(r):
        rr = r % n                       # resolve negatives against OLD length
        nr = rr - ndel if rr > b else rr # rr inside [a,b] is impossible (blocked above)
        return nr if r >= 0 else nr - len(rows)   # originally-negative stays negative style
    meta = st["tl_meta"] or {}
    for cs in meta.get("check_spec", []):
        cs[1] = remap(cs[1])
    sv = meta.get("servo") or {}
    if sv.get("rows"): sv["rows"] = [remap(r) for r in sv["rows"]]
    if sv.get("hover_row") is not None: sv["hover_row"] = remap(sv["hover_row"])
    up = meta.get("user_push") or {}
    if up.get("keeper_off_row") is not None:
        up["keeper_off_row"] = remap(up["keeper_off_row"])
    if shift and sv.get("t") is not None and sv["t"] > rows[a - 1][0]:
        sv["t"] = round(sv["t"] - shift, 2)
    if meta.get("check_spec"):           # keep absolute-time checks self-consistent
        meta["checks"] = [[round(rows[cs[1] % len(rows)][0] + cs[2], 2), cs[0]]
                          for cs in meta["check_spec"]]
    st["tl_sel"] = min(max(a - 1, 0), len(rows) - 1)
    apply_tl_row(st["tl_sel"])
    st["tl_msg"] = f"deleted rows {a}..{b}" + \
        (f", later rows shifted -{shift:.2f}s" if shift else " (times kept: gap becomes one smooth transition)")
    print("[designer]", st["tl_msg"], flush=True)

# ---- playback ---------------------------------------------------------------
RATE = 0.45
def play_track():
    """(t, key) schedule with the demo's rate-limit rule"""
    if len(st["keys"]) < 2: return None
    ts = [0.0]
    for a, b in zip(st["keys"], st["keys"][1:]):
        dj = max(abs(b["joints"][k] - a["joints"][k]) for k in a["joints"])
        db = max(abs(np.array(b["base"][:3]) - np.array(a["base"][:3])).max() / 0.5,
                 abs(b["base"][3] - a["base"][3]) / 0.6)
        ts.append(ts[-1] + max(1.0, dj / RATE, db))
    return ts

def lerp_pose(a, b, w):
    st["base"] = [(1 - w) * x + w * y for x, y in zip(a["base"], b["base"])]
    for k in st["joints"]:
        st["joints"][k] = (1 - w) * a["joints"][k] + w * b["joints"][k]
    for k in st["fingers"]:
        st["fingers"][k] = (1 - w) * a["fingers"][k] + w * b["fingers"][k]

# ---- viewport ---------------------------------------------------------------
import polyscope as ps
import polyscope.imgui as psim
ps.set_program_name("fd_pose_designer")
ps.init()
ps.set_up_dir("z_up"); ps.set_front_dir("neg_y_front"); ps.set_ground_plane_mode("shadow_only")

link_meshes = {}
for ln, metas in model.links.items():
    Vs, Fs, off = [], [], 0
    for (Vl, Fl, _c) in metas:
        Vs.append(Vl); Fs.append(Fl + off); off += len(Vl)
    if Vs:
        pm = ps.register_surface_mesh("r::" + ln, np.concatenate(Vs), np.concatenate(Fs),
                                      smooth_shade=True, color=(0.72, 0.77, 0.86))
        link_meshes[ln] = pm

ctx_meshes = []
try:
    import trimesh
    m = trimesh.load(CAB_MESH, process=False)
    if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(list(m.geometry.values()))
    ctx_meshes.append(ps.register_surface_mesh("ctx::cabinet", np.asarray(m.vertices),
                      np.asarray(m.faces), color=(0.62, 0.55, 0.48), smooth_shade=False))
    toy = trimesh.load(TOY, process=False)
    tv = np.asarray(toy.vertices, float)
    tv = tv - (tv.min(0) + tv.max(0)) / 2
    spot = np.array([2.836, -0.196, 0.95 - tv[:, 2].min() + 0.003])
    ctx_meshes.append(ps.register_surface_mesh("ctx::doll", tv + spot,
                      np.asarray(toy.faces), color=(0.95, 0.45, 0.15), smooth_shade=True))
except Exception as e:
    print("[designer] 上下文网格加载失败:", e)

ps.look_at((WAIST[0] - 1.2, WAIST[1] - 2.6, 1.8), (WAIST[0] + 0.4, WAIST[1], 0.8))

def refresh():
    W = model.fk(base_T(), full_ja())
    for ln, pm in link_meshes.items():
        if ln in W: pm.set_transform(W[ln])
    for cm in ctx_meshes:
        cm.set_enabled(st["show_ctx"])

def slider_group(title, joints):
    if psim.TreeNode(title):
        for j in joints:
            lo, hi = LIMITS.get(j, (-1.57, 1.57))
            ch, v = psim.SliderFloat(j.replace("_joint", ""), float(np.degrees(st["joints"][j])),
                                     float(np.degrees(lo)), float(np.degrees(hi)))
            if ch: st["joints"][j] = float(np.radians(v))
        psim.TreePop()

def ui():
    psim.TextColored((0.4, 0.9, 0.5, 1), "FD keypose designer (pure FK, no physics)")
    _, st["show_ctx"] = psim.Checkbox("show cabinet/doll context", st["show_ctx"])

    if psim.TreeNode("base (pos + yaw)"):
        for i, nm in enumerate(("x", "y", "z")):
            ch, v = psim.SliderFloat("base " + nm, st["base"][i],
                                     st["base"][i] - 2.0, st["base"][i] + 2.0)
            if ch: st["base"][i] = v
        ch, v = psim.SliderFloat("yaw(deg)", float(np.degrees(st["base"][3])), -180.0, 180.0)
        if ch: st["base"][3] = float(np.radians(v))
        psim.TreePop()

    slider_group("right arm r1-r7", ARM_R)
    slider_group("left arm l1-l7", ARM_L)
    if psim.TreeNode("fingers (grouped channels)"):
        for k in ("grasp_r", "throot_r", "thcurl_r", "grasp_l", "throot_l", "thcurl_l"):
            lo, hi = (-0.2, 1.0) if k.startswith("grasp") else (-1.9, 1.9)
            ch, v = psim.SliderFloat(k, st["fingers"][k], lo, hi)
            if ch: st["fingers"][k] = v
        psim.TreePop()
    slider_group("right leg", LEG_R)
    slider_group("left leg", LEG_L)
    if MISC: slider_group("waist/head/misc", MISC)

    psim.Separator(); psim.TextColored((0.9, 0.8, 0.3, 1), f"KEYFRAMES ({len(st['keys'])})")
    if psim.Button("add current pose"):
        st["keys"].append(snap(f"pose{len(st['keys'])}")); st["sel"] = len(st["keys"]) - 1
    psim.SameLine()
    if psim.Button("save JSON"): save_keys()
    psim.SameLine()
    if psim.Button("load JSON"): load_keys()
    for i, k in enumerate(st["keys"]):
        tag = ">> " if i == st["sel"] else "   "
        if psim.Button(f"{tag}{i}:{k['name']}##sel{i}"):
            st["sel"] = i; apply_key(k)
        psim.SameLine()
        if psim.Button(f"upd##u{i}"): st["keys"][i] = snap(k["name"])
        psim.SameLine()
        if psim.Button(f"del##d{i}"): st["keys"].pop(i); st["sel"] = -1; break
        psim.SameLine()
        if psim.Button(f"up##m{i}") and i > 0:
            st["keys"][i - 1], st["keys"][i] = st["keys"][i], st["keys"][i - 1]; break
    if len(st["keys"]) >= 2:
        _, st["play"] = psim.Checkbox("preview play (0.45rad/s rate-limited)", st["play"])
        if st["play"]:
            ts = play_track()
            st["play_t"] += 1.0 / 60.0
            if st["play_t"] >= ts[-1]: st["play_t"] = 0.0
            seg = int(np.searchsorted(ts, st["play_t"], "right")) - 1
            seg = min(seg, len(st["keys"]) - 2)
            w = (st["play_t"] - ts[seg]) / max(ts[seg + 1] - ts[seg], 1e-6)
            lerp_pose(st["keys"][seg], st["keys"][seg + 1], float(np.clip(w, 0, 1)))

    psim.Separator(); psim.TextColored((0.5, 0.8, 1.0, 1), "STAGE scrub (movie-accurate)")
    if STAGES:
        for i, k in enumerate(STAGE_KEYS):
            if i: psim.SameLine()
            tag = ">> " if st["stage_sel"] == i else ""
            if psim.Button(f"{tag}{k}##stg{i}"):
                st["stage_sel"] = i; st["stage_t"] = 0
                st["stage_lab"] = apply_stage(i, 0)
        n = len(STAGES[STAGE_KEYS[st["stage_sel"]]])
        ch, v = psim.SliderInt("stage frame", min(st["stage_t"], n - 1), 0, n - 1)
        if ch:
            st["stage_t"] = v
            st["stage_lab"] = apply_stage(st["stage_sel"], v)
        if st["stage_lab"]:
            psim.SameLine(); psim.Text(st["stage_lab"])
        if psim.Button("capture stage frame as keyframe"):
            st["stage_lab"] = apply_stage(st["stage_sel"], st["stage_t"])
            st["keys"].append(snap(f"{STAGE_KEYS[st['stage_sel']].split()[0]}f{st['stage_t']}"))
        psim.Text("S2 (manipulation) = TIMELINE below")

    psim.Separator(); psim.TextColored((1.0, 0.7, 0.9, 1), "S2 TIMELINE edit (stage2 json)")
    if psim.Button("load stage2_user_timeline.json"):
        load_timeline(os.path.join(WALK, "stage2_user_timeline.json"))
    if st["tl_rows"]:
        ch, v = psim.SliderInt("row", st["tl_sel"], 0, len(st["tl_rows"]) - 1)
        if ch:
            st["tl_sel"] = v; apply_tl_row(v)
        psim.Text(f"row t={st['tl_rows'][st['tl_sel']][0]:.2f}s")
        if psim.Button("write pose back to row"):
            store_tl_row(st["tl_sel"])
        psim.SameLine()
        if psim.Button("save as *_edited.json"):
            save_timeline()
        nrow = len(st["tl_rows"])
        _, st["tl_a"] = psim.SliderInt("del from", min(st["tl_a"], nrow - 1), 0, nrow - 1)
        _, st["tl_b"] = psim.SliderInt("del to", min(st["tl_b"], nrow - 1), 0, nrow - 1)
        _, st["tl_compact"] = psim.Checkbox("compact gap to 2s", st["tl_compact"])
        if psim.Button("delete current row"):
            tl_delete(st["tl_sel"], st["tl_sel"], st["tl_compact"])
        psim.SameLine()
        if psim.Button("delete rows from..to"):
            tl_delete(st["tl_a"], st["tl_b"], st["tl_compact"])
        if st["tl_msg"]:
            psim.TextColored((1.0, 0.65, 0.4, 1), st["tl_msg"])
    refresh()

ps.set_user_callback(ui)
load_keys()
refresh()
print("[designer] ready", flush=True)
ps.show()
