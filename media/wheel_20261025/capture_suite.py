"""Inspect and record every built-in scene using the supplied Windows wheel."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def child(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for key, path in {"GIPC_PROJECT_ROOT": ROOT / "source", "GIPC_ASSETS_DIR": ROOT / "source/Assets",
                      "GIPC_OUTPUT_DIR": out / "native_output", "GIPC_LOG_AND_TEMP_DIR": out / "native_logs"}.items():
        os.environ[key] = str(path)
    record = {"case": args.child, "status": "initializing", "completed_steps": 0,
              "simulation_seconds": 0.0, "target_seconds": args.seconds,
              "record_fps": args.record_fps, "initial_repair_requested": args.repair}
    write_json(out / "status.json", record)
    began = time.perf_counter()
    try:
        import numpy as np
        import pygipc as gipc
        counter = getattr(gipc, "available_scene_count", None)
        record["available_scene_count"] = int(counter()) if counter is not None else None
        if args.child == "provided_example":
            spec = importlib.util.spec_from_file_location("original_example", ROOT / "source/13_mpm_ipcabd.py")
            example = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(example)
            scene, cloth = example.build_scene()
            state = {}
            pre_init = example.make_pre_init(cloth, state)
        else:
            scene = gipc.Scene.from_builtin(int(args.child.split("_")[1]), config={"max_frames": 1_000_000})
            state = {}
            pre_init = None
        world = gipc.World(gipc.Engine("cuda"))
        world.init(scene, on_pre_init=pre_init)
        record["initial_intersections"] = world.has_initial_intersections()
        if args.repair and record["initial_intersections"]:
            record["initial_repair_succeeded"] = bool(world.resolve_initial_intersections())
        world.retrieve()
        dt = float(world.dt)
        if not math.isfinite(dt) or dt <= 0:
            raise RuntimeError(f"Invalid dt: {dt}")
        steps = 1 if args.probe else math.ceil(args.seconds / dt - 1e-9)
        record.update(dt=dt, target_steps=steps, vertices=int(world.vertex_count()),
                      faces=int(world.face_count()), mpm_parents=int(len(world.mpm_particles)),
                      ready=bool(world.simulation_ready()), initialization_seconds=time.perf_counter()-began,
                      python_scene_state=state)
        (out / "effective_config.json").write_text(world.effective_config.to_json(), encoding="utf-8")
        count = len(world.mpm_particles)
        indices = (np.linspace(0, count-1, min(count, 150000), dtype=np.int64)
                   if count else np.empty(0, dtype=np.int64))
        ids = world.mpm_object_ids
        record.update(recorded_mpm_particles=len(indices), recording_dtype="float32",
                      full_position_snapshots_dtype="float64", visualization_subsample_only=True)
        np.savez_compressed(out / "topology.npz", faces=world.faces,
                            point_body_ids=world.point_body_ids(),
                            mpm_object_ids=ids[indices] if len(ids) else ids,
                            particle_sample_indices=indices)
        np.savez_compressed(out / "initial_state_full.npz", vertices=world.vertices,
                            mpm_particles=world.mpm_particles, frame=int(world.frame()))
        write_json(out / "status.json", record)
        if not world.simulation_ready():
            raise RuntimeError("Initial intersections require explicit repair; original geometry retained")
        frame_dir = out / "frames"
        frame_dir.mkdir(exist_ok=True)
        next_save = 0.0
        sample_count = 0
        seconds = 0.0
        initial_shapes = (world.vertices.shape, world.mpm_particles.shape)
        advance_total = 0.0
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            metrics = csv.writer(f)
            metrics.writerow(["step", "frame", "simulation_seconds", "advance_retrieve_seconds",
                              "mpm_y_min", "mpm_y_max", "saved"])
            for step in range(steps + 1):
                elapsed = 0.0
                if step:
                    old_frame = int(world.frame())
                    start = time.perf_counter()
                    world.advance()
                    world.retrieve()
                    elapsed = time.perf_counter() - start
                    advance_total += elapsed
                    if int(world.frame()) != old_frame + 1:
                        raise RuntimeError(f"Frame did not advance at step {step}")
                    seconds += dt
                vertices, particles = world.vertices, world.mpm_particles
                if (vertices.shape, particles.shape) != initial_shapes:
                    raise RuntimeError("Position array shape changed")
                if not np.isfinite(vertices).all() or not np.isfinite(particles).all():
                    raise RuntimeError(f"Nonfinite positions at step {step}")
                saved = seconds + 1e-10 >= next_save or step == steps
                if saved:
                    np.savez_compressed(frame_dir / f"sample_{sample_count:06d}.npz",
                                        vertices=vertices.astype(np.float32),
                                        mpm_particles=particles[indices].astype(np.float32),
                                        frame=int(world.frame()), simulation_seconds=seconds)
                    sample_count += 1
                    next_save = (math.floor(seconds * args.record_fps + 1e-8) + 1) / args.record_fps
                metrics.writerow([step, int(world.frame()), seconds, elapsed,
                                  float(particles[:,1].min()) if len(particles) else "",
                                  float(particles[:,1].max()) if len(particles) else "", int(saved)])
                record.update(status="running", completed_steps=step, simulation_seconds=seconds,
                              saved_samples=sample_count, advance_retrieve_seconds=advance_total)
                if step % 25 == 0 or step == steps:
                    f.flush()
                    write_json(out / "status.json", record)
                    print(f"{args.child}: {step}/{steps}, t={seconds:.4f}, samples={sample_count}", flush=True)
        np.savez_compressed(out / "final_state_full.npz", vertices=world.vertices,
                            mpm_particles=world.mpm_particles, frame=int(world.frame()),
                            simulation_seconds=seconds)
        record["status"] = "completed"
        return_code = 0
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return_code = 1
    record["elapsed_seconds"] = time.perf_counter() - began
    write_json(out / "status.json", record)
    return return_code


def parent(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = args.cases or [f"builtin_{i:02d}" for i in range(15)] + ["provided_example"]
    report = {"mode": "probe" if args.probe else "run", "target_seconds": args.seconds,
              "record_fps": args.record_fps, "cases": [], "status": "running"}
    env = os.environ.copy()
    env.pop("GIPC_PYGIPC_EXTENSION_PATH", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    for case in cases:
        folder = out / case
        folder.mkdir(exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", case,
                   "--output", str(folder), "--seconds", str(args.seconds), "--record-fps", str(args.record_fps)]
        if args.probe:
            command.append("--probe")
        if args.repair:
            command.append("--repair")
        print(f"START {case}", flush=True)
        with (folder / "simulation.log").open("wb") as log:
            proc = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                code = "timeout"
        status_file = folder / "status.json"
        record = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {"case": case}
        record["exit_code"] = code
        if code != 0 or record.get("status") != "completed":
            record["status"] = "failed"
            record.setdefault("error", f"Subprocess exited with {code}")
        write_json(status_file, record)
        case_archive = out / f"{case}.zip"
        with zipfile.ZipFile(case_archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and "native_logs" not in path.relative_to(folder).parts:
                    z.write(path, path.relative_to(folder))
        digest = hashlib.sha256(case_archive.read_bytes()).hexdigest()
        case_archive.with_suffix(".zip.sha256").write_text(f"{digest}  {case_archive.name}\n", encoding="ascii")
        record.update(archive=case_archive.name, archive_bytes=case_archive.stat().st_size, sha256=digest)
        report["cases"].append(record)
        write_json(out / "suite_report.json", report)
        print(f"END {case}: {record['status']} {record.get('error', '')}", flush=True)
    report["status"] = "finished"
    write_json(out / "suite_report.json", report)
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(out.rglob("*")):
            if (path.is_file() and path.name in ("status.json", "suite_report.json", "effective_config.json",
                                                "error.txt", "metrics.csv")
                    and "native_logs" not in path.relative_to(out).parts):
                z.write(path, path.relative_to(out))
        z.write(Path(__file__), "capture_suite.py")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(f"FINISHED: {archive}\nSHA256: {digest}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--timeout", type=int, default=7200)
    arguments = parser.parse_args()
    if arguments.seconds <= 0 or arguments.record_fps <= 0:
        parser.error("seconds and record-fps must be positive")
    raise SystemExit(child(arguments) if arguments.child else parent(arguments))
