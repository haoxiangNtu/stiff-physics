#!/usr/bin/env python3
"""Dump stage-2 record layout (label -> vertex range) for the compositor."""
import sys, json, numpy as np
import functools; print=functools.partial(print,flush=True)
sys.path.insert(0,"/home/ps/Downloads/FD-light/fd_walk")
from stage2_scene import build
S=build()
robot=[]
for lb,r in S["recs"].items():
    robot.append({"label":lb,"off":int(r.vertex_offset),"cnt":int(r.vertex_count)})
allr=S["eng"].get_load_records()
cab,drw=allr[-3],allr[-2]     # 加载顺序: URDF..., cabinet, drawer, doll
out={"robot":robot,"ta":int(S["ta"]),"tb":int(S["tb"]),
     "cab":{"off":int(cab.vertex_offset),"cnt":int(cab.vertex_count)},
     "drw":{"off":int(drw.vertex_offset),"cnt":int(drw.vertex_count)}}
json.dump(out,open("/home/ps/Downloads/FD-light/fd_walk/stage2_out/layout.json","w"),indent=1)
print("[layout] %d robot records -> stage2_out/layout.json"%len(robot))
