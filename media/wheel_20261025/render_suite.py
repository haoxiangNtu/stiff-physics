"""Validate returned suite recordings and render time-accurate browser MP4s."""
from pathlib import Path
import argparse, csv, hashlib, json, re, subprocess, zipfile
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "site_outputs"
TITLES = {
    "provided_example": "Python 示例：液体龙、布料与方块塔",
    "builtin_00": "内置 case1：混合方块",
    "builtin_01": "内置 case2：双兔子",
    "builtin_02": "内置 case3：撞球场景",
    "builtin_03": "内置 case4：高分辨率布料",
    "builtin_04": "内置 case5：柔性软垫",
    "builtin_05": "内置 case6：147 粒子 MPM 场景",
    "builtin_11": "内置 case12：MPM、布料与方块塔",
    "builtin_12": "内置 case13：HyperMPM 双龙碰撞",
    "builtin_13": "内置 case14：implicit-gpu-mpm 双龙",
    "builtin_14": "内置 case15：128 万粒子溃坝",
}
PALETTE = np.array([[54,132,178],[219,126,64],[104,157,118],[148,108,165],
                    [198,169,64],[83,162,163]], dtype=np.uint8)


def prepare(archive, allow_partial=False):
    expected = archive.with_suffix(".zip.sha256").read_text().split()[0]
    received = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert received == expected or allow_partial
    run = archive.with_suffix("")
    damaged=[]
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            assert (run/n).resolve().is_relative_to(run.resolve())
        if received == expected:z.extractall(run)
        else:
            for n in z.namelist():
                try:data=z.read(n)
                except Exception as exc:
                    damaged.append({'file':n,'error':str(exc)})
                    continue
                p=run/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            assert damaged and all(x['file'].startswith('frames/') for x in damaged)
            (run/'transfer_integrity.json').write_text(json.dumps({'expected_sha256':expected,
                'received_sha256':received,'damaged_entries':damaged},indent=2))
    status = json.loads((run/"status.json").read_text())
    assert status["status"] == "completed", status
    assert status["simulation_seconds"] + 1e-7 >= status["target_seconds"]
    metrics = list(csv.DictReader((run/"metrics.csv").open()))
    assert len(metrics) == status["completed_steps"]+1
    assert [int(x["step"]) for x in metrics] == list(range(len(metrics)))
    paths = sorted((run/"frames").glob("*.npz"))
    assert len(paths)+len(damaged) == status["saved_samples"]
    lo, hi = np.full(3,np.inf), np.full(3,-np.inf)
    mesh_lo, mesh_hi = lo.copy(), hi.copy()
    times = []
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            v, m = d["vertices"], d["mpm_particles"]
            assert v.shape == (status["vertices"],3)
            assert m.shape == (status["recorded_mpm_particles"],3)
            assert np.isfinite(v).all() and np.isfinite(m).all()
            times.append(float(d["simulation_seconds"]))
            for array in (v,m):
                if len(array):
                    lo=np.minimum(lo,array.min(axis=0));hi=np.maximum(hi,array.max(axis=0))
            if len(v):
                mesh_lo=np.minimum(mesh_lo,v.min(axis=0));mesh_hi=np.maximum(mesh_hi,v.max(axis=0))
    assert np.all(np.diff(times)>0) and abs(times[-1]-status["simulation_seconds"])<1e-6
    for name in ("initial_state_full", "final_state_full"):
        with np.load(run/(name+".npz"), allow_pickle=False) as d:
            assert d["mpm_particles"].shape == (status["mpm_parents"],3)
            assert np.isfinite(d["vertices"]).all() and np.isfinite(d["mpm_particles"]).all()
    status.update(local_validation="partial_transfer" if damaged else "passed", original_archive_sha256=expected,
                  bounds=[lo.tolist(),hi.tolist()])
    if damaged:
        status.update(status="partial_transfer",remote_simulation_status="completed",
                      received_archive_sha256=received,missing_samples=damaged,
                      available_samples=len(paths),preview_gap_policy="nearest valid sample; gaps marked in video")
    return run,status,paths,np.asarray(times),lo,hi,mesh_lo,mesh_hi


def render(archive, allow_partial=False, video_only=False):
    run,s,paths,times,lo,hi,mesh_lo,mesh_hi = prepare(archive,allow_partial)
    case=s["case"]; dest=OUT/case;dest.mkdir(parents=True,exist_ok=True)
    with np.load(run/"topology.npz",allow_pickle=False) as t:
        faces=t["faces"]; body_ids=t["point_body_ids"];mpm_ids=t["mpm_object_ids"]
    with np.load(paths[0],allow_pickle=False) as d:
        vertices=d["vertices"].copy();particles=d["mpm_particles"].copy()
    mesh=None;cloud=None
    if len(faces):
        assert faces.min()>=0 and faces.max()<len(vertices)
        mesh=pv.PolyData(vertices,np.column_stack((np.full(len(faces),3),faces)).ravel())
        face_ids=body_ids[faces[:,0]].astype(int)
        mesh.cell_data["rgb"]=PALETTE[face_ids%len(PALETTE)]
        if case in ("provided_example", "builtin_11"):
            counts=np.bincount(face_ids)
            rgb=np.tile(PALETTE[1],(len(faces),1))
            rgb[face_ids==int(counts.argmax())]=PALETTE[0]
            mesh.cell_data["rgb"]=rgb
    if len(particles):
        cloud=pv.PolyData(particles)
        if len(mpm_ids)==len(particles) and len(np.unique(mpm_ids))>1:
            cloud.point_data["rgb"]=PALETTE[mpm_ids.astype(int)%len(PALETTE)]
        else:
            cloud.point_data["rgb"]=np.tile(np.array([28,57,83],dtype=np.uint8),(len(particles),1))
    width,height,view_h=1920,1080,850
    plotter=pv.Plotter(shape=(1,2),off_screen=True,border=False,window_size=(width,view_h))
    for col in (0,1):
        plotter.subplot(0,col);plotter.set_background("#f6f8fb")
        if mesh is not None:
            plotter.add_mesh(mesh,scalars="rgb",rgb=True,show_scalar_bar=False,
                             smooth_shading=False,ambient=.38,diffuse=.65)
        if cloud is not None:
            plotter.add_mesh(cloud,style="points",scalars="rgb",rgb=True,show_scalar_bar=False,
                             point_size=6 if len(particles)<1000 else (3.4 if len(particles)<30000 else 2.2),
                             render_points_as_spheres=True,ambient=.4)
        if col==0:
            a,b=(vertices.min(axis=0),vertices.max(axis=0)) if case in ('provided_example','builtin_11','builtin_02','builtin_04') else (lo,hi)
            center=(a+b)/2;extent=np.maximum(b-a,.05)
            direction=np.array([.6,.35,.75]);direction/=np.linalg.norm(direction)
            plotter.camera_position=[center+direction*np.linalg.norm(extent)*2,center,(0,1,0)]
            plotter.enable_parallel_projection();plotter.camera.parallel_scale=np.linalg.norm(extent)*.60
            if case in ('builtin_00','builtin_01'):
                right=np.cross(np.array([0.,1.,0.]),direction);right/=np.linalg.norm(right)
                up=np.cross(direction,right)
                horizontal=np.abs(right)@extent;vertical=np.abs(up)@extent
                plotter.camera.parallel_scale=max(vertical,horizontal/(960/view_h))*.55
            if case in ("provided_example","builtin_11"):
                center=np.array([0.,.9,0.])
                plotter.camera_position=[center+direction*10,center,(0,1,0)]
                plotter.camera.parallel_scale=2.8
        else:
            center=(lo+hi)/2;span=np.maximum(hi-lo,.05)
            plotter.camera_position=[center+[0,np.linalg.norm(span)*3,0],center,(0,0,-1)]
            plotter.enable_parallel_projection();plotter.camera.parallel_scale=max(span[2],span[0]/(960/view_h))*.57
        plotter.reset_camera_clipping_range()
    plotter.enable_anti_aliasing("fxaa");plotter.show(auto_close=False,interactive=False)
    fontpath="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fonts={n:ImageFont.truetype(fontpath,n) for n in (22,25,30,40)}
    movie=dest/(case+".mp4")
    encoder_log=(dest/"encoding.log").open("wb")
    proc=subprocess.Popen(["ffmpeg","-hide_banner","-y","-f","rawvideo","-pix_fmt","rgb24",
                           "-s","1920x1080","-r","30","-i","-","-an","-c:v","libx264",
                           "-preset","fast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(movie)],
                          stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=encoder_log)
    try:
        # Use the closest available physics sample at each 30 Hz video time, without geometry interpolation.
        video_times=np.arange(round(s["simulation_seconds"]*30)+1)/30
        for j,vt in enumerate(video_times):
            k=int(np.argmin(np.abs(times-vt)))
            with np.load(paths[k],allow_pickle=False) as d:
                if mesh is not None:mesh.points=d["vertices"]
                if cloud is not None:cloud.points=d["mpm_particles"]
            for renderer in plotter.renderers:
                renderer.reset_camera_clipping_range()
            plotter.render();img=plotter.screenshot(return_img=True,transparent_background=False)
            canvas=Image.new("RGB",(width,height),"#f6f8fb");canvas.paste(Image.fromarray(img[:,:,:3]),(0,155))
            draw=ImageDraw.Draw(canvas)
            draw.text((48,22),TITLES.get(case,case),font=fonts[40],fill="#172b42")
            draw.text((50,82),f"原始 Windows wheel · RTX 5090 · {s['completed_steps']:,} 步 · 实时速度回放",font=fonts[22],fill="#536479")
            if s.get('missing_samples'):
                message=f"预览：{s['available_samples']}/{s['saved_samples']} 采样帧可用，{len(s['missing_samples'])} 帧待重传"
                if abs(times[k]-vt)>s['dt']+1e-8:message="缺少此处采样，显示最近有效帧"
                draw.text((1100,82),message,font=fonts[22],fill="#b83b30")
            draw.text((1470,28),f"t = {times[k]:.2f} / {s['simulation_seconds']:.2f} s",font=fonts[30],fill="#172b42")
            draw.text((48,126),"三维视角（场景近景）" if case in ('provided_example','builtin_11','builtin_02','builtin_04') else "三维视角",font=fonts[25],fill="#334b65")
            draw.text((1008,126),"全域俯视图",font=fonts[25],fill="#334b65")
            draw.line((960,164,960,985),fill="#dfe5ed",width=2)
            label=f"{s['vertices']:,} 网格顶点 · {s['mpm_parents']:,} MPM 粒子 · dt = {s['dt']:.6g} s"
            draw.text((48,1007),label,font=fonts[22],fill="#536479")
            if s['mpm_parents']>s['recorded_mpm_particles']:
                draw.text((1300,1007),f"显示采样 {s['recorded_mpm_particles']:,} 点，仿真全量",font=fonts[22],fill="#536479")
            draw.line((48,1056,1872,1056),fill="#dce4ed",width=7)
            draw.line((48,1056,48+1824*j/max(len(video_times)-1,1),1056),fill="#458db6",width=7)
            proc.stdin.write(canvas.tobytes())
            if j==len(video_times)//3:canvas.resize((960,540)).save(dest/(case+".jpg"),quality=90)
            if j%50==0:print(case,j,len(video_times),flush=True)
        proc.stdin.close();assert proc.wait()==0
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        encoder_log.close();plotter.close()
    if video_only:
        print('VIDEO READY',case,flush=True)
        return
    public_status={k:v for k,v in s.items() if k not in ("python_scene_state",)}
    public_status.update(title=TITLES.get(case,case),video_fps=30,video_playback_speed=1.0,
                         source_wheel_sha256="6db76a8416c9d4e9bd48eef4d0fa1029409a1154b3765507f4c58aceaf0488b7")
    (dest/"metadata.json").write_text(json.dumps(public_status,indent=2,ensure_ascii=False))
    data_suffix="_partial_data.zip" if s.get('missing_samples') else "_data.zip"
    with zipfile.ZipFile(dest/(case+data_suffix),"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(run.rglob("*")):
            if not p.is_file() or "native_logs" in p.relative_to(run).parts:continue
            if p.name in ("status.json","simulation.log","error.txt"):continue
            z.write(p,p.relative_to(run))
        z.write(dest/"metadata.json","metadata.json")
        z.write(ROOT/"capture_suite.py","capture_suite.py")
        z.writestr("README.txt","Original pygipc wheel results. Coordinates: simulation units, Y up.\nframes/: float32 visualization samples at about 30 Hz; large MPM scenes use the stored particle_sample_indices.\ninitial_state_full.npz and final_state_full.npz: ALL particles and vertices at float64.\nmetrics.csv covers EVERY physics step. The simulation itself was not downsampled.\nSource: user-provided pygipc 0.0.1 Windows CPython 3.14 wheel; StiffGIPC / Kemeng Huang et al.\n")
    print("READY",case,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("archives",nargs="+",type=Path)
    parser.add_argument('--allow-partial',action='store_true',help='Explicitly render only ZIP entries passing CRC, marking missing samples')
    parser.add_argument('--video-only',action='store_true',help='Re-render camera/labels without changing the data archive')
    args=parser.parse_args()
    for archive in args.archives:render(archive.resolve(),args.allow_partial,args.video_only)
