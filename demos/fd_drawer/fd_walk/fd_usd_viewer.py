#!/usr/bin/env python3
"""FD demo USD viewer — scrub / play the delivered animation USD.

Loads any of our exported .usdc stages (transform-animated robot links +
point-animated furniture/doll + /World/Cam) and plays it interactively:
frame slider, play/pause + speed, free orbit or the movie camera.

Run:  python3.11 fd_usd_viewer.py [path/to.usdc]     (needs a display)
Default file: output_v3/fd_4stage_animation_v3_full.usdc
"""
import sys, os
import numpy as np
from pxr import Usd, UsdGeom

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ps/Downloads/FD-light/fd_walk/output_v3/fd_4stage_animation_v3_full.usdc"
st_usd = Usd.Stage.Open(PATH)
T0, T1 = int(st_usd.GetStartTimeCode()), int(st_usd.GetEndTimeCode())
FPS = st_usd.GetTimeCodesPerSecond() or 25.0
print(f"[viewer] {os.path.basename(PATH)}  frames {T0}-{T1} @{FPS:.0f}fps")

meshes = []
for prim in st_usd.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        F = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        pts = m.GetPointsAttr()
        P0 = np.array(pts.Get(Usd.TimeCode(T0)))
        animated_pts = pts.GetNumTimeSamples() > 2
        meshes.append(dict(prim=prim, path=prim.GetPath().pathString,
                           pts=pts, P0=P0, F=F, anim=animated_pts))
print(f"[viewer] {len(meshes)} meshes")
cam_prim = st_usd.GetPrimAtPath("/World/Cam")
cam_xf = UsdGeom.Xformable(cam_prim) if cam_prim else None

import polyscope as ps
import polyscope.imgui as psim
ps.set_program_name("fd_usd_viewer")
ps.init()
ps.set_up_dir("z_up"); ps.set_front_dir("neg_y_front"); ps.set_ground_plane_mode("shadow_only")
xc = UsdGeom.XformCache()
regs = []
for mi in meshes:
    nm = mi["path"]
    col = (0.95, 0.45, 0.15) if "Doll" in nm else \
          ((0.62, 0.55, 0.48) if "Furniture" in nm else (0.72, 0.77, 0.86))
    sm = ps.register_surface_mesh(nm.replace("/", "_"), mi["P0"], mi["F"],
                                  color=col, smooth_shade=True)
    regs.append((sm, mi))

ui_st = {"frame": T0, "play": False, "speed": 1.0, "use_cam": True, "last": -1}

def set_frame(fr):
    t = Usd.TimeCode(fr)
    xc.SetTime(t); xc.Clear()
    for sm, mi in regs:
        if mi["anim"]:
            sm.update_vertex_positions(np.array(mi["pts"].Get(t)))
            sm.set_transform(np.eye(4))
        else:
            M = np.array(xc.GetLocalToWorldTransform(mi["prim"]))   # row-vector convention
            sm.set_transform(M.T)                                    # polyscope column convention
    if ui_st["use_cam"] and cam_xf is not None:
        Mc = np.array(cam_xf.GetLocalTransformation(t))
        eye = Mc[3][:3]; fwd = -Mc[2][:3]
        ps.look_at(tuple(eye), tuple(eye + fwd))

def ui():
    psim.TextColored((0.4, 0.9, 0.5, 1), f"{os.path.basename(PATH)}   {T0}-{T1} @{FPS:.0f}fps")
    ch, ui_st["frame"] = psim.SliderInt("frame", ui_st["frame"], T0, T1)
    _, ui_st["play"] = psim.Checkbox("play", ui_st["play"])
    psim.SameLine()
    _, ui_st["speed"] = psim.SliderFloat("speed", ui_st["speed"], 0.1, 4.0)
    _, ui_st["use_cam"] = psim.Checkbox("movie cam (/World/Cam)", ui_st["use_cam"])
    psim.Text(f"t = {ui_st['frame']/FPS:.2f}s")
    if ui_st["play"]:
        ui_st["frame"] = T0 + int((ui_st["frame"] - T0 + max(1, round(ui_st["speed"] * 60 / FPS)))
                                  % (T1 - T0 + 1))
    if ui_st["frame"] != ui_st["last"]:
        set_frame(ui_st["frame"]); ui_st["last"] = ui_st["frame"]

ps.set_user_callback(ui)
set_frame(T0)
print("[viewer] ready", flush=True)
ps.show()
