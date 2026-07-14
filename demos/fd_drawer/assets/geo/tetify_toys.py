#!/usr/bin/env python3
"""Tetrahedralize toy surface OBJs into FEM tet meshes via pytetwild (fTetWild).

Caches result next to the OBJ as <name>.tet.npz  {verts (N,3) f64, tets (M,4) i32}.
Re-run with FORCE=1 to rebuild.  EDGE_FAC overrides edge_length_fac (bb-diag frac;
smaller = more tets = softer/slower). Run standalone to (re)build + report.
"""
import os, sys, time, numpy as np
# pytetwild imported lazily inside tetify(): cache hits (<name>.tet.npz) work in
# environments without it (e.g. the v0.8.x conda env).


def load_obj(path):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(V, np.float64), np.asarray(F, np.int32)


def tetify(obj_path, edge_fac=0.15, epsilon=0.01, coarsen=True, force=False):
    """Return (verts f64 (N,3), tets i32 (M,4)) for obj_path, cached to .tet.npz.

    Defaults (edge_fac 0.15 / epsilon 0.01 / coarsen) give ~1k tets on the toys:
    enough resolution to deform around fingers, cheap enough for interactive FEM.
    The tight default fTetWild envelope (epsilon 0.001) otherwise pins ~20k tets
    to the surface regardless of edge_fac — raising epsilon + coarsen is what
    actually controls tet count here.
    """
    cache = os.path.splitext(obj_path)[0] + ".tet.npz"
    if os.path.exists(cache) and not force:
        d = np.load(cache)
        return d["verts"].astype(np.float64), d["tets"].astype(np.int32)
    import pytetwild   # lazy: only needed on cache miss / FORCE=1
    V, F = load_obj(obj_path)
    t0 = time.time()
    tv, tt = pytetwild.tetrahedralize(V.astype(np.float64), F.astype(np.int32),
                                      edge_length_fac=edge_fac, epsilon=epsilon,
                                      coarsen=coarsen, optimize=True)
    tv = np.ascontiguousarray(tv, np.float64)
    tt = np.ascontiguousarray(tt, np.int32)
    # fTetWild emits tets with the opposite winding to the engine's convention
    # (all signed volumes come out negative).  Flip node order 2<->3 so every
    # tet has positive signed volume -> valid rest-shape Dm for the FEM energy.
    p = tv[tt]
    vol = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                    p[:, 3] - p[:, 0]) / 6.0
    if np.median(vol) < 0:
        tt = np.ascontiguousarray(tt[:, [0, 1, 3, 2]], np.int32)
    np.savez(cache, verts=tv, tets=tt)
    dt = time.time() - t0
    # quality after orientation fix
    p = tv[tt]
    vol = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                    p[:, 3] - p[:, 0]) / 6.0
    print(f"  {os.path.basename(obj_path)}: surf {len(V)}v/{len(F)}f -> "
          f"tet {len(tv)}v/{len(tt)}t in {dt:.1f}s | "
          f"vol_sum={vol.sum():.3e} m^3 | neg={(vol <= 0).sum()} | "
          f"min_vol={vol.min():.2e}")
    return tv, tt


if __name__ == "__main__":
    edge = float(os.environ.get("EDGE_FAC", "0.15"))
    eps = float(os.environ.get("EPSILON", "0.01"))
    force = os.environ.get("FORCE", "0") == "1"
    geo = os.path.dirname(os.path.abspath(__file__))
    toys = sys.argv[1:] or ["Toy00.obj", "Toy02.obj"]
    print(f"[tetify] edge_length_fac={edge} force={force}")
    for t in toys:
        tetify(os.path.join(geo, t), edge_fac=edge, epsilon=eps, force=force)
