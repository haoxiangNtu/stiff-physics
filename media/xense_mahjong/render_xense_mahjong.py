"""Tactile readout + video for a ``xense_mahjong.py`` recording.

From the recorded gel vertices this rebuilds what each Xense pad's camera sees, frame by frame, with the same
sensor model as the rest of ``stiff_physics.sensors``:

* depth   -- orthographic z-buffer of the deformed coat surface (pad-local), 0.079375 mm pixels;
* deformation map -- depth minus the untouched depth (frame 0);
* RGB     -- Taccel's published normal->RGB network (``n2rgb.pth`` + ``6k_ref.png``, MIT) on the metric normals;
* markers -- the 7 x 7 printed dots, displacement against frame 0.

Glyph readout (Taccel Fig. 1): at the frame of peak squeeze the left pad's deformation map is thresholded
(Otsu) inside the region where the pad covers the tile face; the engraved strokes are where the gel is NOT
pressed.  It is scored against the ground-truth glyph projected into the pad's pixels through the recorded pad
and tile poses (IoU / precision / recall).  Images are shown as seen by a person facing the engraved face
(rows = world up -> down, columns = world +x); the physical camera sits behind the gel and sees it mirrored.

Outputs in the recording's folder:  readout.png, readout.json, tactile_seq.npz (per-frame maps, decimated),
xense_mahjong.mp4 (3D view + both pads' RGB images + deformation map + force/lift plot, faststart).

    PYTHONPATH=. python examples/xense_mahjong/render_xense_mahjong.py out/xense_mahjong/v1/xense_mahjong.npz \
        --taccel-data /path/to/Taccel/taccel/data
needs torch (RGB network), pyrender (EGL) and ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from xense_assets import FINGERS, HAND_LINKS, XenseGripper, load_pad_asset  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def otsu(x: np.ndarray) -> float:
    h, e = np.histogram(x, 256)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


class PadView:
    """Per-frame tactile images of one pad."""

    def __init__(self, asset, side, z, renderer=None):
        from stiff_physics.sensors.observation import pixel_grid_local
        self.a, self.side, self.z, self.R = asset, side, z, renderer
        self.cam = asset.camera
        self.tf0 = z[f"pad_tf0_{side}"]
        self.xs, self.ys = pixel_grid_local(self.cam)
        self.rest = None
        self.rest_markers = None

    def pad_tf(self, fr):
        return self.z[self.side][fr] @ self.tf0

    def local(self, fr):
        W = self.pad_tf(fr)
        return (self.z[f"gel_{self.side}"][fr].astype(np.float64) - W[:3, 3]) @ W[:3, :3]

    def frame(self, fr, rgb=True):
        from stiff_physics.sensors.observation import (depth_from_coat_surface, depth_to_normal,
                                                       markers_to_pixels, smooth_depth)
        loc = self.local(fr)
        d = smooth_depth(depth_from_coat_surface(loc, self.a.coat_faces, self.cam), 1.0)
        if self.rest is None:
            self.rest = d.copy()
            self.rest_markers = markers_to_pixels(self.a.markers_from_verts(loc), self.cam)
        out = dict(depth=d, defo=np.clip(d - self.rest, 0.0, None))
        mk = markers_to_pixels(self.a.markers_from_verts(loc), self.cam)
        out["markers"], out["flow"] = mk, mk - self.rest_markers
        if rgb and self.R is not None:
            n = depth_to_normal(d, self.cam.pixel_size_m)
            out["rgb"] = self.R.render(d[None], n[None])[0]
        return out

    def view(self, img):
        """Pad image -> as seen by a person facing the pad's object (left pad: the engraved face)."""
        # rows: pad x = gripper-body z = world down (row 0 = top); columns: left pad y = world +x (already),
        # right pad y = world -x -> flip so both read as seen from their own side of the tile.
        return img[:, ::-1] if self.side == "right" else img

    def tile_pixels(self, fr, tile_d, tile_T):
        """Ground-truth glyph and tile-face masks in this pad's pixels at frame fr."""
        W = self.pad_tf(fr)
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        p = np.stack([X, Y, np.zeros_like(X)], -1) @ W[:3, :3].T + W[:3, 3]
        q = (p - tile_T[:3, 3]) @ tile_T[:3, :3]                     # tile-local (mesh frame)
        gx, gy, m = tile_d["gx"], tile_d["gy"], tile_d["mask"]
        on = (q[..., 0] > gx[0]) & (q[..., 0] < gx[-1]) & (q[..., 1] > gy[0]) & (q[..., 1] < gy[-1])
        i = np.clip(np.round((q[..., 0] - gx[0]) / (gx[1] - gx[0])).astype(int), 0, len(gx) - 1)
        j = np.clip(np.round((q[..., 1] - gy[0]) / (gy[1] - gy[0])).astype(int), 0, len(gy) - 1)
        return m[i, j] & on, on


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--taccel-data", default=os.environ.get("TACCEL_DATA", ""),
                    help="folder with n2rgb.pth and 6k_ref.png from Taccel (MIT): taccel/data in "
                         "https://github.com/Taccel-Simulator/Taccel (or set TACCEL_DATA)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    if not (a.taccel_data and osp.exists(osp.join(a.taccel_data, "n2rgb.pth"))):
        sys.exit("--taccel-data: need the folder holding Taccel's n2rgb.pth and 6k_ref.png")
    z = dict(np.load(a.npz, allow_pickle=False))
    st = json.loads(str(z["stats"]))
    outdir = osp.dirname(osp.abspath(a.npz))
    n_fr = len(z["t"])
    print(f"[render] {n_fr} frames, status {st['status']}, grip {st['grip_force']} N, gel E {st.get('gel_E')}")

    from stiff_physics.sensors.render import TactileRenderer
    asset = load_pad_asset(st.get("gel_E", 2e5))
    R = TactileRenderer(osp.join(a.taccel_data, "n2rgb.pth"), osp.join(a.taccel_data, "6k_ref.png"),
                        device=a.device, resolution=(asset.camera.height, asset.camera.width))
    pads = {s: PadView(asset, s, z, R) for s in FINGERS}
    for p in pads.values():
        p.frame(0, rgb=False)                                         # rest reference

    # ---------------------------------------------------------------- readout at peak squeeze
    from xense_assets import tile
    td = tile(st["char"])
    stages = [lab for _, lab in st["stages"]]
    stage = z["stage"]
    cand = np.nonzero(np.isin(stage, [stages.index(s) for s in ("squeeze", "lift", "hold") if s in stages]))[0]
    peak = int(cand[np.argmax(z["fn_left"][cand])])
    obs = {s: pads[s].frame(peak) for s in pads}
    tile_T = z["tile"][peak] @ z["tile_T0"]
    gt, face = pads["left"].tile_pixels(peak, td, tile_T)
    defo = obs["left"]["defo"]
    region = face & (defo > 0) | face & gt
    thr = otsu(defo[face]) if face.any() else 0.0
    read = face & (defo < thr)
    inter, union = (read & gt).sum(), (read | gt).sum()
    score = dict(frame=peak, t=float(z["t"][peak]), fn_left=float(z["fn_left"][peak]),
                 fn_right=float(z["fn_right"][peak]), defo_max_mm=float(defo.max() * 1e3),
                 defo_face_median_mm=float(np.median(defo[face & ~gt]) * 1e3) if face.any() else 0.0,
                 defo_glyph_median_mm=float(np.median(defo[gt]) * 1e3) if gt.any() else 0.0,
                 otsu_mm=thr * 1e3, iou=float(inter / max(union, 1)),
                 precision=float(inter / max(read.sum(), 1)), recall=float(inter / max(gt.sum(), 1)),
                 face_px=int(face.sum()), glyph_px=int(gt.sum()))
    # blank control: the right pad presses the flat back face -- same threshold, should read nothing
    face_r = pads["right"].tile_pixels(peak, td, tile_T)[1]
    dr = obs["right"]["defo"]
    score["control_right_false_px"] = int((face_r & (dr < thr)).sum())
    score["control_right_face_px"] = int(face_r.sum())
    i_lift = int(np.argmax(stage == stages.index("lift")))
    tile_dy = (z["tile"][:, 1, 3] - z["tile"][i_lift, 1, 3]) * 1e3          # from the start of the lift
    lift = (z["lift"] - z["lift"][i_lift]) * 1e3
    held = stage == stages.index("hold")
    score["tile_lift_mm"] = float(tile_dy.max())
    score["hold_slip_mm"] = float(np.max(np.abs(lift[held] - tile_dy[held]))) if held.any() else None
    with open(osp.join(outdir, "readout.json"), "w") as f:
        json.dump(score, f, indent=1)
    print("[render] readout", json.dumps(score))

    _readout_png(osp.join(outdir, "readout.png"), pads, obs, gt, face, read, score, st)

    # ---------------------------------------------------------------- per-frame maps + video
    if a.no_video:
        return 0
    frames = list(range(0, n_fr, a.stride))
    seq = {s: [] for s in pads}
    _video(osp.join(outdir, "xense_mahjong.mp4"), z, st, pads, frames, seq, a.fps, score)
    np.savez_compressed(osp.join(outdir, "tactile_seq.npz"), frames=np.array(frames),
                        **{f"defo_{s}": np.array(v, np.float16) for s, v in seq.items()})
    return 0


def _readout_png(path, pads, obs, gt, face, read, score, st):
    from PIL import Image, ImageDraw, ImageFont
    L, Rt = pads["left"], pads["right"]
    def rgb8(x):
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)
    def heat(d, vmax):
        u = np.clip(d / max(vmax, 1e-9), 0, 1)
        return rgb8(np.stack([u ** 0.6, u ** 1.5, 0.35 * (1 - u) + 0.2 * u], -1))
    vmax = max(np.percentile(obs["left"]["defo"][face], 99), 1e-5) if face.any() else 1e-5
    ov = np.zeros(gt.shape + (3,), np.uint8)
    ov[face] = (60, 60, 64)
    ov[read & ~gt] = (230, 80, 60)
    ov[gt & ~read] = (70, 120, 230)
    ov[read & gt] = (245, 245, 245)
    panels = [(L.view(rgb8(obs["left"]["rgb"])), "left pad RGB (engraved face)"),
              (L.view(heat(obs["left"]["defo"], vmax)), f"deformation map (max {vmax*1e3:.2f} mm)"),
              (L.view(ov), f"readout vs truth  IoU {score['iou']:.2f}"),
              (Rt.view(rgb8(obs["right"]["rgb"])), "right pad RGB (blank back face)")]
    s = 2
    h, w = gt.shape[0] * s // 1, gt.shape[1] * s // 1
    W = Image.new("RGB", (len(panels) * (w + 16) + 16, h + 90), (250, 250, 252))
    try:
        f = ImageFont.truetype(FONT, 20); fs = ImageFont.truetype(FONT, 16)
    except Exception:
        f = fs = None
    d = ImageDraw.Draw(W)
    for k, (img, lab) in enumerate(panels):
        W.paste(Image.fromarray(img).resize((w, h), Image.NEAREST), (16 + k * (w + 16), 50))
        d.text((16 + k * (w + 16), 18), lab, fill=(20, 20, 20), font=fs)
    d.text((16, h + 58), f"Xense gripper pinching '{st['char']}': grip {score['fn_left']:.1f} / {score['fn_right']:.1f} N, "
           f"precision {score['precision']:.2f}, recall {score['recall']:.2f}, white = correct, red = extra, blue = missed; "
           f"right pad (control) false pixels {score['control_right_false_px']} / {score['control_right_face_px']}",
           fill=(40, 40, 40), font=fs)
    W.save(path)
    print(f"[render] -> {path}")


def _video(path, z, st, pads, frames, seq, fps, score):
    from PIL import Image, ImageDraw, ImageFont
    from stiff_physics.sensors.offscreen import SceneRenderer
    g = XenseGripper()
    T00 = g.fk(0.0)
    W0 = dict(zip([str(n) for n in z["link_names"]], z["link_W0"]))
    body_T0 = {"hand": W0[g.root]}
    for s, link in FINGERS.items():
        body_T0[s] = W0[link] @ np.linalg.inv(T00[link])
    VW, VH, PW = 1280, 720, 520
    Rr = SceneRenderer(VW - PW, VH, bg=(0.94, 0.95, 0.97))
    col = {"tool_adapter_link": (0.20, 0.20, 0.22), "gripper_mount_plate": (0.30, 0.31, 0.35),
           "wrist_d435i_link": (0.08, 0.08, 0.09), "gripper_base_link": (0.74, 0.76, 0.78),
           "finger_left": (0.58, 0.61, 0.66), "finger_right": (0.58, 0.61, 0.66)}
    parts = []
    for link in (*HAND_LINKS, *FINGERS.values()):
        m = g.meshes[link]
        v = np.asarray(m.vertices) @ T00[link][:3, :3].T + T00[link][:3, 3]          # root frame, q = 0
        grp = "hand" if link in HAND_LINKS else ("left" if link == FINGERS["left"] else "right")
        Rr.add_mesh(link, v, np.asarray(m.faces), color=col[link], merge=False)
        parts.append((link, grp))
    Rr.add_mesh("tile", z["tile_rest_v"], z["tile_rest_f"], color=(0.96, 0.93, 0.80), merge=False)
    gx = np.array([[x, -0.0002, zz] for x in (-0.3, 0.3) for zz in (-0.3, 0.3)])
    Rr.add_mesh("table", gx, np.array([[0, 1, 3], [0, 3, 2], [0, 3, 1], [0, 2, 3]], np.int32),
                color=(0.50, 0.38, 0.27), merge=False)
    a0 = pads["left"].a
    surf = a0.surface_faces
    # from the open side of the jaws (+x, fingers close along z), eye just above the table so the lift reads against
    # the table edge (pyrender draws no shadows); the upper half of the gripper is out of frame
    Rr.look_at((0.22, 0.035, 0.10), (0.0, 0.050, 0.0), up=(0, 1, 0), yfov=0.50)
    Rr._nodes["key"].light.intensity = 2.2                  # the default rig (3.2 + 1.2) washes the grey parts out
    Rr._nodes["fill"].light.intensity = 0.8
    Rr.scene.ambient_light = np.array([0.22, 0.22, 0.24])
    try:
        f = ImageFont.truetype(FONT, 20); fs = ImageFont.truetype(FONT, 15)
    except Exception:
        f = fs = None
    tmp = osp.join(osp.dirname(path), "_frames")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    stages = [lab for _, lab in st["stages"]]
    t = z["t"]; fnl, fnr = z["fn_left"], z["fn_right"]
    tile_dy = (z["tile"][:, 1, 3] - z["tile"][0, 1, 3]) * 1e3
    gel_nodes = []
    vmax = 0.25e-3
    for k, fr in enumerate(frames):
        for link, grp in parts:
            Rr.set_pose(link, z[grp][fr] @ body_T0[grp])
        Rr.set_pose("tile", z["tile"][fr])
        for nd in gel_nodes:
            Rr.scene.remove_node(nd)
        gel_nodes = [Rr.add_mesh(f"gel_{s}", z[f"gel_{s}"][fr], surf, color=(0.20, 0.22, 0.26), merge=False)
                     for s in pads]
        img = Image.new("RGB", (VW, VH), (250, 250, 252))
        img.paste(Image.fromarray(Rr.render()), (0, 0))
        d = ImageDraw.Draw(img)
        d.text((18, 14), "StiffGIPC · Xense gripper + visuotactile pads · mahjong pinch", fill=(25, 25, 30), font=f)
        d.text((18, 44), f"t = {t[fr]:.2f} s   {stages[z['stage'][fr]]}", fill=(60, 60, 70), font=fs)
        ob = {s: pads[s].frame(fr) for s in pads}
        for s in pads:
            seq[s].append(ob[s]["defo"][::2, ::2])
        # tactile column: left RGB, left deformation, right RGB  (each pad image 302 x 202 -> 230 x 154)
        ph, pw = 230, 154
        x0 = VW - PW + 14
        def rgb8(x):
            return (np.clip(x, 0, 1) * 255).astype(np.uint8)
        u = np.clip(ob["left"]["defo"] / vmax, 0, 1)
        heat = rgb8(np.stack([u ** 0.6, u ** 1.5, 0.35 * (1 - u) + 0.2 * u], -1))
        tiles = [(pads["left"].view(rgb8(ob["left"]["rgb"])), "left pad · RGB"),
                 (pads["left"].view(heat), "left pad · deformation"),
                 (pads["right"].view(rgb8(ob["right"]["rgb"])), "right pad · RGB")]
        for i, (im, lab) in enumerate(tiles):
            xx = x0 + i * (pw + 12)
            img.paste(Image.fromarray(im).resize((pw, ph), Image.BILINEAR), (xx, 40))
            d.text((xx, 16), lab, fill=(40, 40, 50), font=fs)
        # marker arrows on the left RGB tile
        mk, fl = ob["left"]["markers"], ob["left"]["flow"]
        for (r, c), (dr, dc) in zip(mk, fl):
            cy, cx = 40 + r * ph / pads["left"].cam.height, x0 + c * pw / pads["left"].cam.width
            d.line([cx, cy, cx + 8 * dc * pw / pads["left"].cam.width, cy + 8 * dr * ph / pads["left"].cam.height],
                   fill=(255, 40, 40), width=2)
        # force + lift plot
        py0, py1, px0, px1 = 330, 600, x0, VW - 20
        d.rectangle([px0, py0, px1, py1], outline=(180, 180, 185))
        tm = t[-1]
        fmax = max(1.0, float(max(fnl.max(), fnr.max()))) * 1.15
        def P(tt, v, vm):
            return px0 + (px1 - px0) * tt / tm, py1 - (py1 - py0) * v / vm
        for arr, c, vm in ((fnl, (200, 60, 50), fmax), (fnr, (60, 90, 200), fmax),
                           (tile_dy, (40, 150, 70), max(1.0, tile_dy.max() * 1.15))):
            pts = [P(t[i], arr[i], vm) for i in range(0, fr + 1)]
            if len(pts) > 1:
                d.line(pts, fill=c, width=2)
        d.text((px0, py0 - 22), f"pad force L {fnl[fr]:5.2f} N   R {fnr[fr]:5.2f} N   tile lift {tile_dy[fr]:5.1f} mm",
               fill=(40, 40, 50), font=fs)
        d.text((px0 + 4, py0 + 4), f"red / blue: pad normal force (0..{fmax:.0f} N)   green: tile height", fill=(90, 90, 100),
               font=fs)
        img.save(osp.join(tmp, f"{k:05d}.png"))
        if k % 50 == 0:
            print(f"[render] video frame {k}/{len(frames)}", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", osp.join(tmp, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", path], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[render] -> {path}")


if __name__ == "__main__":
    sys.exit(main())
