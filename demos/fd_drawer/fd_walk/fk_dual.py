"""Tiny FK for fd_dual_arm.urdf (no engine, instant). Z-up world coords."""
import os, numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
D="/home/ps/Downloads/FD-light/fd-urdf-full/FD-URDF"
URDF=os.path.join(D,"fd_dual_arm.urdf")
WAIST=np.array([2.3313,0.0359,1.0500])   # 站立姿态 waist 世界位置 (mount 平移)
class DualFK:
    def __init__(self):
        root=ET.parse(URDF).getroot()
        self.joints={}
        self.children={}
        childs=set()
        for j in root.findall("joint"):
            p=j.find("parent").get("link"); c=j.find("child").get("link")
            o=j.find("origin")
            xyz=[float(x) for x in (o.get("xyz","0 0 0").split() if o is not None else [0]*3)]
            rpy=[float(x) for x in (o.get("rpy","0 0 0").split() if o is not None else [0]*3)]
            a=j.find("axis")
            axis=[float(x) for x in (a.get("xyz").split() if a is not None else [1,0,0])]
            T=np.eye(4); T[:3,:3]=R.from_euler("xyz",rpy).as_matrix(); T[:3,3]=xyz
            self.joints[j.get("name")]=dict(parent=p,child=c,type=j.get("type"),T=T,axis=np.array(axis))
            self.children.setdefault(p,[]).append(j.get("name")); childs.add(c)
        links={l.get("name") for l in root.findall("link")}
        self.root=list(links-childs)[0]
    def fk(self, ja=None):
        ja=ja or {}
        W={self.root:np.eye(4)}
        W[self.root][:3,3]=WAIST
        stack=[self.root]
        while stack:
            p=stack.pop()
            for jn in self.children.get(p,[]):
                jd=self.joints[jn]
                Tj=np.eye(4)
                if jd["type"]=="revolute":
                    ang=ja.get(jn,0.0)
                    if abs(ang)>1e-12:
                        Tj[:3,:3]=R.from_rotvec(jd["axis"]/np.linalg.norm(jd["axis"])*ang).as_matrix()
                W[jd["child"]]=W[p]@jd["T"]@Tj
                stack.append(jd["child"])
        return W
