"""TokaMaker solver backend for the Streamlit app.

Replicates the MAIN_AOE_tokamaker.ipynb workflow (mesh -> Grad-Shafranov solve ->
stats -> coil currents -> vertical-stability feedback parameter).

Usage:  python tokamak_solver.py design.json outdir

OpenFUSIONToolkit is imported lazily inside run(), so the pure helpers below
(DEFAULT_DESIGN, resize_polygon, plasma_boundary, profile_shape, validate_design)
can be imported by the GUI without ever touching the Fortran library.
"""
import contextlib
import io
import json
import math
import os
import sys
import time
import traceback
import warnings

import numpy as np

DEFAULT_DESIGN = {
    "major_radius": 2.9, "minor_radius": 1.4, "elongation": 1.8, "triangularity": 0.5,
    "vv_boundary": [[1.0, -3.0], [5.0, -3.0], [5.0, 3.0], [1.0, 3.0]],
    "coil_locs": [[1.7, 3.5], [1.7, -3.5], [4.0, 4.0], [4.0, -4.0],
                  [5.5, 1.5], [5.5, -1.5], [0.5, 0.75], [0.5, -0.75]],
    "B0": 1.0, "Ip_target": 3.0e6, "p_core_target": 8.0e4,
    "ffp_alpha": 4.0, "ffp_gamma": 1.0, "pp_alpha": 1.5, "pp_gamma": 2.0,
}

# Fixed by the notebook ("DO NOT CHANGE")
PLASMA_DX, COIL_DX, VV_DX, VAC_DX = 0.15, 0.15, 0.15, 0.25
VV_ETA = 6.9e-7
COIL_SIZE = 0.3
WALL_THICKNESS = 0.04
N_ISOFLUX, ISOFLUX_WEIGHT = 30, 5.0

_LABELS = {
    "major_radius": "Major radius", "minor_radius": "Minor radius", "elongation": "Elongation",
    "triangularity": "Triangularity", "B0": "Toroidal field B0", "Ip_target": "Plasma current target",
    "p_core_target": "Core pressure target", "ffp_alpha": "FF' alpha", "ffp_gamma": "FF' gamma",
    "pp_alpha": "P' alpha", "pp_gamma": "P' gamma",
}
_MUST_BE_POSITIVE = ["major_radius", "minor_radius", "elongation", "B0", "Ip_target",
                     "p_core_target", "ffp_alpha", "ffp_gamma", "pp_alpha", "pp_gamma"]

# gs_err_reason() strings from OFT grad_shaf.F90 -> (error code, student explanation)
_SOLVE_ERRORS = {
    'Exceeded "maxits"': (-1, "The solver gave up without settling on a stable plasma shape. "
                              "Try a less extreme shape or move the coils closer to the plasma."),
    "Total poloidal flux is zero": (-2, "The magnetic flux came out as zero, so there is no plasma to confine."),
    "Closed flux volume lost": (-3, "The plasma lost its closed magnetic surfaces: the coils cannot hold this "
                                    "shape. Try moving coils closer to the plasma or a less extreme shape."),
    'Axis dropped below "rmin"': (-4, "The centre of the plasma drifted into the machine's central axis."),
    "Toroidal current droppped too low": (-5, "The plasma current collapsed during the solve."),
    "Matrix solve failed for targets": (-6, "The solver could not match all your targets at once. Try a less "
                                            "extreme shape, pressure or profile, or a different coil layout."),
    "Isoflux fitting failed": (-7, "The coils could not be set to produce the requested plasma shape."),
    "Wall eigenmode flux loop fitting failed": (-8, "Internal wall-mode fitting failed."),
}


# ---------------------------------------------------------------- pure helpers
def resize_polygon(points, dx):
    """Offset a polygon by dx (outward for counter-clockwise points). Copied from the notebook."""
    points = np.array(points, dtype=float)
    new_points = np.empty(np.shape(points))
    n = np.shape(points)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            last = points[i - 1, :]
            nxt = points[(i + 1) % n, :]
            par = points[i, :] - last
            par /= np.linalg.norm(par)
            perp = np.array([par[1], -par[0]])
            temp = points[i, :] + perp * dx
            par_2 = nxt - points[i, :]
            par_2 /= np.linalg.norm(par_2)
            perp_2 = [par_2[1], -par_2[0]]
            new_points[i, :] = temp + dx / np.dot(perp_2, par) * par + par * dx / np.dot(par_2, perp) * np.dot(par_2, par)
    return new_points


def plasma_boundary(npts, R0, Z0, a, kappa, delta):
    """Same as OpenFUSIONToolkit.TokaMaker.util.create_isoflux (up-down symmetric)."""
    theta = np.arange(npts) * 2.0 * np.pi / npts
    with np.errstate(invalid="ignore"):
        R = R0 + a * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    return np.column_stack([R, Z0 + a * (kappa * np.sin(theta))])


def profile_shape(alpha, gamma, npts=40):
    """x, y of OpenFUSIONToolkit's create_power_flux_fun: y = (1 - x^alpha)^gamma."""
    x = np.linspace(0.0, 1.0, npts)
    return x, np.power(1.0 - np.power(x, alpha), gamma)


def _ccw(poly):
    """Return poly in counter-clockwise order (the orientation resize_polygon offsets outward)."""
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if area > 0 else poly[::-1].copy()


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segs_intersect(p1, p2, q1, q2, eps=1e-9):
    """True if closed segments p1-p2 and q1-q2 touch or cross."""
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True

    def on_seg(a, b, c, d):  # c collinear with a-b and inside its bounding box
        return abs(d) <= eps and min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps \
            and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    return on_seg(q1, q2, p1, d1) or on_seg(q1, q2, p2, d2) or on_seg(p1, p2, q1, d3) or on_seg(p1, p2, q2, d4)


def _edges(poly):
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def _point_in_poly(pt, poly):
    inside = False
    for (x1, y1), (x2, y2) in _edges(poly):
        if (y1 > pt[1]) != (y2 > pt[1]) and pt[0] < x1 + (pt[1] - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _edges_cross(A, B):
    return any(_segs_intersect(a1, a2, b1, b2) for a1, a2 in _edges(A) for b1, b2 in _edges(B))


def _polys_overlap(A, B):
    return _edges_cross(A, B) or _point_in_poly(A[0], B) or _point_in_poly(B[0], A)


def _self_intersections(poly):
    n, edges, hits = len(poly), _edges(poly), []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segs_intersect(*edges[i], *edges[j]):
                hits.append((i, j))
    return hits


def _dist_to_poly(pt, poly):
    best = np.inf
    for a, b in _edges(poly):
        ab = b - a
        t = np.clip(np.dot(pt - a, ab) / max(np.dot(ab, ab), 1e-300), 0.0, 1.0)
        best = min(best, np.linalg.norm(pt - (a + t * ab)))
    return best


def _coil_square(c):
    h = COIL_SIZE / 2
    return np.array([[c[0] - h, c[1] - h], [c[0] + h, c[1] - h], [c[0] + h, c[1] + h], [c[0] - h, c[1] + h]])


def validate_design(design):
    """Return a list of {"level": "error"|"warning", "message": str} for a design dict."""
    out = []
    err = lambda m: out.append({"level": "error", "message": m})
    warn = lambda m: out.append({"level": "warning", "message": m})
    p = {}
    for key, label in _LABELS.items():
        try:
            p[key] = float(design[key])
            if not math.isfinite(p[key]):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            err(f"{label} is missing or is not a number.")
    for key in _MUST_BE_POSITIVE:
        if key in p and p[key] <= 0:
            err(f"{_LABELS[key]} must be greater than zero (it is {p[key]:g}).")
    if "triangularity" in p and abs(p["triangularity"]) >= 1:
        err(f"Triangularity must be between -1 and 1 (it is {p['triangularity']:g}).")
    try:
        vv = np.array(design["vv_boundary"], dtype=float).reshape(-1, 2)
        coils = np.array(design["coil_locs"], dtype=float).reshape(-1, 2)
        assert np.all(np.isfinite(vv)) and np.all(np.isfinite(coils))
    except Exception:
        err("The vessel points or coil locations are not (R, Z) number pairs.")
        return out

    # --- vacuum vessel
    vv_ok = False
    if len(vv) < 3:
        err("The vessel needs at least 3 points.")
    else:
        n = len(vv)
        dupes = [(i, j) for i in range(n) for j in range(i + 1, n) if np.linalg.norm(vv[i] - vv[j]) < 0.01]
        for i, j in dupes:
            err(f"Vessel points {i + 1} and {j + 1} are in the same place.")
        crossings = [] if dupes else _self_intersections(vv)
        for i, j in crossings:
            err(f"The vessel outline crosses itself (side {i + 1}-{i + 2} hits side {j + 1}-{(j + 1) % n + 1}).")
        if np.any(vv[:, 0] <= 0):
            err("Every vessel point needs R > 0.")
        elif not dupes and not crossings:
            outer = resize_polygon(_ccw(vv), WALL_THICKNESS)
            bad = np.where(~np.all(np.isfinite(outer), axis=1))[0]
            if len(bad):
                for k in bad:
                    r, z = _ccw(vv)[k]
                    err(f"The vessel point at R={r:.2f}, Z={z:.2f} is not a corner (it is in line with its neighbours).")
            elif _self_intersections(outer) or _edges_cross(outer, vv):
                err("The vessel has a corner too sharp, or a gap too narrow, for its 4 cm wall.")
            else:
                vv_ok = True

    # --- plasma boundary
    if vv_ok and all(k in p for k in ("major_radius", "minor_radius", "elongation", "triangularity")) \
            and abs(p["triangularity"]) < 1:
        bpts = plasma_boundary(N_ISOFLUX, p["major_radius"], 0.0, p["minor_radius"], p["elongation"], p["triangularity"])
        outside = [pt for pt in bpts if not _point_in_poly(pt, vv)]
        if outside:
            err("The target plasma does not fit inside the vessel.")
        elif _edges_cross(bpts, vv):
            err("The target plasma does not fit inside the vessel.")
        else:
            gap = min(_dist_to_poly(pt, vv) for pt in bpts)
            if gap < 0.05:
                warn(f"The target plasma is only {100 * gap:.0f} cm from the vessel wall.")

    # --- coils
    if len(coils) < 1:
        err("Add at least one PF coil.")
    squares = [_coil_square(c) for c in coils]
    for i, (c, sq) in enumerate(zip(coils, squares)):
        name = f"PF_{i + 1}"
        if sq[:, 0].min() <= 0:
            err(f"Coil {name} must have R > {COIL_SIZE / 2:.2f} m.")
        if vv_ok and _polys_overlap(sq, outer):
            where = "is inside the vessel" if _point_in_poly(c, vv) else "overlaps the vessel wall"
            err(f"Coil {name} {where}.")
        for j in range(i + 1, len(coils)):
            if np.all(np.abs(coils[i] - coils[j]) < COIL_SIZE + 1e-6):
                err(f"Coils {name} and PF_{j + 1} overlap.")
    return out


# ---------------------------------------------------------------- solver
def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.number)):
        v = float(v)
        return v if math.isfinite(v) else None
    return v


def _solve(design, outdir, res, stage):
    """Run the notebook workflow. `stage(name)` records progress. Fills `res` in place."""
    root = os.environ.get("OFT_ROOTPATH") or os.path.expanduser("~/OpenFUSIONToolkit/install_release")
    sys.path.append(os.path.join(root, "python"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (6, 6), "font.weight": "bold", "axes.labelweight": "bold",
                         "lines.linewidth": 2, "lines.markeredgewidth": 2})

    R0, a = float(design["major_radius"]), float(design["minor_radius"])
    kappa, delta = float(design["elongation"]), float(design["triangularity"])
    vv = _ccw(np.array(design["vv_boundary"], dtype=float))  # resize_polygon offsets outward only for CCW
    vv_outer = resize_polygon(vv, WALL_THICKNESS)
    coils = np.array(design["coil_locs"], dtype=float)
    bpts = plasma_boundary(N_ISOFLUX, R0, 0.0, a, kappa, delta)

    stage("mesh")
    from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
    mesh = gs_Domain()
    mesh.define_region("air", VAC_DX, "boundary")
    mesh.define_region("vv", VV_DX, "conductor", eta=VV_ETA)
    mesh.define_region("plasma", PLASMA_DX, "plasma")
    for i in range(len(coils)):
        mesh.define_region(f"PF_{i + 1}", COIL_DX, "coil")
    mesh.add_annulus(vv, "plasma", vv_outer, "vv", parent_name="air")
    for i, (r, z) in enumerate(coils):
        mesh.add_rectangle(r, z, COIL_SIZE, COIL_SIZE, f"PF_{i + 1}", parent_name="air")
    mesh_pts, mesh_lc, mesh_reg = mesh.build_mesh()
    coil_dict, cond_dict = mesh.get_coils(), mesh.get_conductors()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    mesh.plot_mesh(fig, ax)
    fig.savefig(os.path.join(outdir, "mesh.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    res["images"]["mesh"] = "mesh.png"

    stage("setup")
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    tk = TokaMaker(OFT_env(nthreads=1))
    tk.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    tk.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    tk.setup(order=2, F0=float(design["B0"]) * R0)
    tk.set_targets(Ip=float(design["Ip_target"]), pax=float(design["p_core_target"]))
    profs = {}
    for name in ("ffp", "pp"):
        x, y = profile_shape(float(design[f"{name}_alpha"]), float(design[f"{name}_gamma"]), 40)
        profs[name] = {"type": "linterp", "x": x, "y": y}
    tk.set_profiles(ffp_prof=profs["ffp"], pp_prof=profs["pp"])
    tk.set_isoflux_constraints(bpts, weights=ISOFLUX_WEIGHT * np.ones(len(bpts)))
    nc = tk.ncoils + 1
    tk.set_coil_reg(np.eye(nc, dtype=np.float64), np.zeros(nc), 0.1 * np.ones(nc))

    stage("solve")
    tk.init_psi(R0, 0.0, a, kappa, delta)
    try:
        tk.solve()  # OFT 26.6 raises ValueError("Error in solve: <reason>") instead of returning a flag
        res["err_flag"] = 0
    except ValueError as e:
        reason = str(e).replace("Error in solve:", "").strip()
        code, msg = _SOLVE_ERRORS.get(reason, (None, f"The equilibrium solver failed ({reason})."))
        res["err_flag"], res["error"] = code, msg
        return

    stage("stats")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tk.print_info()
    res["log"] = buf.getvalue()
    stats = tk.get_stats()
    res["stats"] = _jsonable(stats)
    currents, _ = tk.get_coil_currents()
    res["coil_currents"] = {k: float(v) for k, v in currents.items()}
    res["coil_current_limit"] = 2.0 * float(stats["Ip"])
    for psi_n in (0.999, 0.99):
        try:
            lcfs = tk.trace_surf(psi_n)
        except Exception:
            lcfs = None
        if lcfs is not None:  # traces come back with thousands of points; ~500 is plenty to draw
            res["lcfs"] = _jsonable(np.array(lcfs)[::max(1, len(lcfs) // 500)])
            break
    fig, ax = plt.subplots()
    tk.plot_machine(fig, ax)
    tk.plot_psi(fig, ax)
    tk.plot_constraints(fig, ax)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    fig.savefig(os.path.join(outdir, "equilibrium.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    res["images"]["equilibrium"] = "equilibrium.png"
    res["ok"] = True

    stage("stability")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            eig_vals, _ = tk.eig_td(omega=-1e4, neigs=10)
            growth_rate = -eig_vals[0, 0]
            eigval_wall, _ = tk.eig_wall()
            wall_time = 1.0 / eigval_wall[1, 0]
        res["stability"] = _jsonable({"growth_rate": growth_rate, "wall_time": wall_time,
                                      "feedback_capability": growth_rate * wall_time})
    except Exception as e:
        res["stability_error"] = f"The vertical stability calculation failed ({e})."


def run(design, outdir):
    """Validate and solve `design`, writing images, progress.txt and results.json into outdir."""
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    res = {"ok": False, "stage": "validate", "error": None, "err_flag": None, "validation": [],
           "stats": None, "coil_currents": None, "coil_current_limit": None, "stability": None,
           "stability_error": None, "lcfs": None, "target_boundary": [], "images": {}, "log": "",
           "elapsed_s": 0.0, "timings": []}

    def stage(name):
        res["stage"] = name
        res["timings"].append([name, round(time.time() - t0, 2)])  # seconds since start when stage began
        with open(os.path.join(outdir, "progress.txt"), "w") as f:
            f.write(name + "\n")

    stage("validate")
    res["validation"] = validate_design(design)
    try:
        res["target_boundary"] = _jsonable(plasma_boundary(
            N_ISOFLUX, float(design["major_radius"]), 0.0, float(design["minor_radius"]),
            float(design["elongation"]), float(design["triangularity"])))
    except Exception:
        pass
    errors = [v["message"] for v in res["validation"] if v["level"] == "error"]
    if errors:
        more = f" (and {len(errors) - 1} more problem(s))" if len(errors) > 1 else ""
        res["error"] = errors[0] + more
    else:
        # Send Fortran + Python output to solver.log (unbuffered so it survives a hard abort)
        os.environ["GFORTRAN_UNBUFFERED_PRECONNECTED"] = "y"
        log_path = os.path.join(outdir, "solver.log")
        sys.stdout.flush(); sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        with open(log_path, "w") as logf:
            os.dup2(logf.fileno(), 1)
            os.dup2(logf.fileno(), 2)
            sys.stdout.reconfigure(line_buffering=True)
            try:
                _solve(design, outdir, res, stage)
                if res["ok"]:
                    stage("done")
            except Exception as e:
                res["error"] = f"Something went wrong during the '{res['stage']}' step: {e}"
                traceback.print_exc()
            finally:
                sys.stdout.flush(); sys.stderr.flush()
                os.dup2(saved[0], 1); os.dup2(saved[1], 2)
                os.close(saved[0]); os.close(saved[1])
        with open(log_path, errors="replace") as f:
            tail = f.read()[-6000:]
        res["log"] = (res["log"] + "\n--- solver output (tail) ---\n" + tail).strip()
    res["elapsed_s"] = round(time.time() - t0, 3)
    res["log"] += "\n--- stage start times (s) ---\n" + ", ".join(f"{n} {t}" for n, t in res["timings"]) \
        + f", finished {res['elapsed_s']}"
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    if len(sys.argv) != 3:
        print("usage: python tokamak_solver.py design.json outdir")
        return 2
    outdir = sys.argv[2]
    try:
        with open(sys.argv[1]) as f:
            design = json.load(f)
    except Exception as e:
        design = {}
        os.makedirs(outdir, exist_ok=True)
        print(f"Could not read design file: {e}")
    res = run(design, outdir)
    print(f"ok={res['ok']} stage={res['stage']} error={res['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
