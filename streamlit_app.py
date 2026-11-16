"""Tokamaker Demo: a no-code version of the MAIN_AOE_tokamaker notebook.

Students set the same quantities they would type into the notebook, watch their machine
update live, and press "Solve equilibrium" to run TokaMaker. Each solve runs
tokamak_solver.py in its own process, so OpenFUSIONToolkit is never imported here.

Run with:  streamlit run streamlit_app.py
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tokamak_solver import DEFAULT_DESIGN, plasma_boundary, profile_shape, resize_polygon, validate_design

SOLVER = Path(__file__).with_name("tokamak_solver.py")
LOGO = Path(__file__).with_name("tokamaker_logo.png")
COIL_SIZE = 0.3  # m, coils are squares (fixed in the notebook)
WALL = 0.04  # m, vessel wall thickness (fixed in the notebook)
SCALAR_KEYS = ["major_radius", "minor_radius", "elongation", "triangularity",
               "B0", "ffp_alpha", "ffp_gamma", "pp_alpha", "pp_gamma"]
STAGES = {"start": "start-up", "validate": "design check", "mesh": "mesh building", "setup": "solver setup",
          "solve": "equilibrium solve", "stats": "statistics", "stability": "vertical-stability check",
          "done": "final step"}
OK, WARN, INFO = ":green[:material/check_circle:]", ":orange[:material/warning:]", ":blue[:material/info:]"


# ------------------------------------------------------------------ design state
def reset_design():
    """Put every control back to the notebook's values (also used to initialise the app)."""
    s = st.session_state
    for key in SCALAR_KEYS:
        s[key] = float(DEFAULT_DESIGN[key])
    s.Ip_MA = DEFAULT_DESIGN["Ip_target"] / 1e6
    s.p_kPa = DEFAULT_DESIGN["p_core_target"] / 1e3
    s.vv_base = pd.DataFrame(DEFAULT_DESIGN["vv_boundary"], columns=["R", "Z"], dtype=float)
    s.coil_base = pd.DataFrame(DEFAULT_DESIGN["coil_locs"], columns=["R", "Z"], dtype=float)
    # New editor keys throw away the tables' pending edits so the reset values show.
    s.table_version = s.get("table_version", 0) + 1


def points(df):
    """(R, Z) rows of an edited table as a list of float pairs, skipping rows with blank cells."""
    return df[["R", "Z"]].dropna().astype(float).values.tolist()


def read_design(vv_df, coil_df):
    s = st.session_state
    design = {key: s[key] for key in SCALAR_KEYS}
    design.update(Ip_target=s.Ip_MA * 1e6, p_core_target=s.p_kPa * 1e3,
                  vv_boundary=points(vv_df), coil_locs=points(coil_df))
    return design


# ------------------------------------------------------------------ sidebar controls
def controls():
    """Draw every design control in the sidebar; return the edited vessel and coil tables."""
    s = st.session_state
    sb = st.sidebar
    sb.header("Your tokamak design")
    sb.button("Reset to default values", on_click=reset_design, width="stretch")

    sb.subheader("1. Plasma shape targets")
    sb.caption("The cross-section you want the plasma to have. TokaMaker will try to match it.")
    sb.slider("Major radius R (m)", 0.5, 8.0, step=0.05, key="major_radius",
              help="Distance from the machine's central axis to the centre of the plasma.")
    sb.slider("Minor radius a (m)", 0.2, 2.5, step=0.05, key="minor_radius",
              help="Half the width of the plasma.")
    sb.slider("Elongation κ", 1.0, 2.5, step=0.05, key="elongation",
              help="Height ÷ width. 1 is a circle; bigger stretches the plasma vertically.")
    sb.slider("Triangularity δ", -0.8, 0.8, step=0.05, key="triangularity",
              help="Positive: top and bottom tips lean toward the central axis (a 'D' shape). "
                   "Negative: they lean outward.")

    sb.subheader("2. Vacuum vessel")
    sb.caption("The airtight steel chamber around the plasma, drawn with a 4 cm wall. Each row is a "
               "corner (R, Z); corners are joined **in table order** and the last joins back to the "
               "first (row 1 is point 1 in the picture). Add rows at the bottom or delete rows to "
               "change the outline. A wall close to the plasma helps hold it still vertically.")
    number = {"R": st.column_config.NumberColumn("R (m)", format="%.2f", step=0.01, min_value=0.0),
              "Z": st.column_config.NumberColumn("Z (m)", format="%.2f", step=0.01)}
    vv_df = sb.data_editor(s.vv_base, key=f"vv_table_{s.table_version}", num_rows="dynamic",
                           hide_index=True, column_config=number, width="stretch")

    sb.subheader("3. Poloidal-field (PF) coils")
    sb.caption("30 cm × 30 cm coils whose magnetic fields push and pull the plasma into shape. "
               "TokaMaker picks their currents for you; you choose where they sit. Keep them outside "
               "the vessel and move them to fix the plasma shape. Each row is one coil (row 1 is PF_1): "
               "add rows at the bottom or delete rows to change how many there are.")
    coil_df = sb.data_editor(s.coil_base, key=f"coil_table_{s.table_version}", num_rows="dynamic",
                             hide_index=True, column_config=number, width="stretch")

    sb.subheader("4. Field, current and pressure")
    sb.caption("Raising the plasma current lowers q95; raising the core pressure raises β_pol.")
    sb.slider("Toroidal field B0 (T)", 0.5, 13.0, step=0.1, key="B0",
              help="Strength of the main magnetic field that runs the long way around the torus, "
                   "measured at the major radius R.")
    sb.slider("Plasma current target Ip (MA)", 0.1, 20.0, step=0.1, key="Ip_MA",
              help="Current flowing through the plasma itself. Its magnetic field wraps around the "
                   "plasma and squeezes it (1 MA = one million amps).")
    sb.number_input("Core pressure target (kPa)", 1.0, 5000.0, step=5.0, format="%.1f", key="p_kPa",
                    help="Pressure at the plasma's centre (1 atmosphere ≈ 101 kPa). Hotter or denser "
                         "plasma means higher pressure.")

    sb.subheader("5. Profile shapes")
    sb.caption("How current (FF′) and pressure (P′) are spread from the plasma's centre (ψ̂ = 0) to "
               "its edge (ψ̂ = 1). Each follows $(1-\\hat\\psi^{\\alpha})^{\\gamma}$. A smaller α or a larger γ piles more "
               "current near the centre (a 'peaked' profile), which raises l_i.")
    for name, what in (("ffp", "FF′ (current)"), ("pp", "P′ (pressure)")):
        sb.slider(f"{what} α", 0.5, 6.0, step=0.05, key=f"{name}_alpha",
                  help="Smaller α: the profile drops off sooner, so it is more peaked at the centre.")
        sb.slider(f"{what} γ", 0.5, 6.0, step=0.05, key=f"{name}_gamma",
                  help="Larger γ: the profile falls faster toward the edge, so it is more peaked.")
    return vv_df, coil_df


# ------------------------------------------------------------------ live plots
def flagged_objects(messages):
    """Which drawn objects an error message is about: coil names, 'plasma' or 'vessel'."""
    flagged = set()
    for m in messages:
        if m["level"] != "error":
            continue
        coils = re.findall(r"\bPF_\d+\b", m["message"])
        if coils:
            flagged.update(coils)
        elif "plasma" in m["message"]:
            flagged.add("plasma")
        elif "vessel" in m["message"]:
            flagged.add("vessel")
    return flagged


def closed(poly):
    poly = np.asarray(poly, dtype=float)
    return np.vstack([poly, poly[:1]])


def add_wall(fig, vv, bad):
    """Draw the vessel as the solver builds it: the outline plus a 4 cm wall offset outward."""
    color = "#d62728" if bad else "#555555"
    if len(vv) < 3:
        return
    signed_area = np.sum(vv[:, 0] * np.roll(vv[:, 1], -1) - np.roll(vv[:, 0], -1) * vv[:, 1])
    inner = vv if signed_area > 0 else vv[::-1]  # resize_polygon offsets outward for counter-clockwise points
    outer = resize_polygon(inner, WALL)
    if np.all(np.isfinite(outer)):
        # One "keyhole" path (outer loop, then inner loop backwards) fills just the wall, not the inside.
        wall = np.vstack([closed(outer), inner[:1], inner[:0:-1], inner[:1]])
        fig.add_scatter(x=wall[:, 0], y=wall[:, 1], fill="toself", fillcolor=color, mode="lines",
                        line=dict(color=color, width=1), name="Vacuum vessel wall", hoverinfo="skip")
    else:
        ring = closed(vv)
        fig.add_scatter(x=ring[:, 0], y=ring[:, 1], mode="lines", line=dict(color=color, width=3),
                        name="Vacuum vessel wall", hoverinfo="skip")
    fig.add_scatter(x=vv[:, 0], y=vv[:, 1], mode="markers+text", text=[str(i + 1) for i in range(len(vv))],
                    marker=dict(size=17, color="white", line=dict(color=color, width=2)),
                    textfont=dict(size=10, color="black"), name="Vessel points (table rows)",
                    hovertemplate="Vessel point %{text}<br>R=%{x:.2f} m, Z=%{y:.2f} m<extra></extra>")


def add_coils(fig, coil_locs, flagged):
    """Draw the coils as squares named PF_1, PF_2, ... in table order."""
    h = COIL_SIZE / 2
    named = [(f"PF_{i + 1}", r, z) for i, (r, z) in enumerate(coil_locs)]
    for bad in (False, True):
        rows = [c for c in named if (c[0] in flagged) == bad]
        if not rows:
            continue
        color = "#d62728" if bad else "#c87533"
        xs, ys = [], []
        for _, r, z in rows:
            xs += [r - h, r + h, r + h, r - h, r - h, None]
            ys += [z - h, z - h, z + h, z + h, z - h, None]
        fig.add_scatter(x=xs, y=ys, fill="toself", fillcolor=color, mode="lines", line=dict(color=color),
                        name="PF coils (problem)" if bad else "PF coils", hoverinfo="skip")
        fig.add_scatter(x=[r for _, r, _ in rows], y=[z + h for _, _, z in rows], mode="text",
                        text=[name for name, _, _ in rows], textposition="top center", showlegend=False,
                        textfont=dict(color=color), customdata=[z for _, _, z in rows],
                        hovertemplate="%{text}<br>R=%{x:.2f} m, Z=%{customdata:.2f} m<extra></extra>")


def cross_section(design, flagged, lcfs):
    fig = go.Figure()
    add_wall(fig, np.array(design["vv_boundary"], dtype=float).reshape(-1, 2), "vessel" in flagged)
    shape = (design["major_radius"], 0.0, design["minor_radius"], design["elongation"], design["triangularity"])
    color = "#d62728" if "plasma" in flagged else "#1f77b4"
    smooth, targets = closed(plasma_boundary(200, *shape)), plasma_boundary(30, *shape)
    fig.add_scatter(x=smooth[:, 0], y=smooth[:, 1], fill="toself", fillcolor="rgba(31,119,180,0.15)",
                    mode="lines", line=dict(color=color, width=2), name="Target plasma shape", hoverinfo="skip")
    fig.add_scatter(x=targets[:, 0], y=targets[:, 1], mode="markers", marker=dict(color=color, size=5),
                    name="30 shape-target points", hovertemplate="R=%{x:.2f} m, Z=%{y:.2f} m<extra></extra>")
    add_coils(fig, design["coil_locs"], flagged)
    if lcfs:
        edge = closed(lcfs)
        fig.add_scatter(x=edge[:, 0], y=edge[:, 1], mode="lines", line=dict(color="#2ca02c", width=3),
                        name="Solved plasma edge (LCFS)", hoverinfo="skip")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="R (m)",
                      yaxis_title="Z (m)", yaxis=dict(scaleanchor="x", scaleratio=1, fixedrange=True),
                      xaxis_fixedrange=True, dragmode=False,  # hover only: no zooming or panning
                      legend=dict(orientation="h", yanchor="bottom", y=1.01))
    return fig


def profile_figure(design):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("FF′ (current)", "P′ (pressure)"), shared_yaxes=True)
    for col, name in ((1, "ffp"), (2, "pp")):
        x, y = profile_shape(design[f"{name}_alpha"], design[f"{name}_gamma"])
        fig.add_scatter(x=x, y=y, mode="lines", row=1, col=col, showlegend=False,
                        hovertemplate="ψ̂=%{x:.2f}: %{y:.2f}<extra></extra>")
        fig.update_xaxes(title_text="ψ̂ (0 = centre, 1 = edge)", row=1, col=col)
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="relative size", dragmode=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# ------------------------------------------------------------------ solving
@st.cache_resource
def solve_slots():
    """Shared by every visitor: at most 2 solves run at once (each needs ~250 MB), the rest wait their turn."""
    return threading.Semaphore(2)


def run_solver(design):
    """Run tokamak_solver.py on `design` in a throwaway folder; return everything the results need."""
    workdir = Path(tempfile.mkdtemp(prefix="tokamak_"))
    try:
        design_file = workdir / "design.json"
        design_file.write_text(json.dumps(design))
        try:
            proc = subprocess.run([sys.executable, str(SOLVER), str(design_file), str(workdir)],
                                  capture_output=True, text=True, timeout=600)
            stderr = proc.stderr
        except subprocess.TimeoutExpired:
            stderr = "The solver was stopped after 10 minutes."
        if (workdir / "results.json").exists():
            res = json.loads((workdir / "results.json").read_text())
        else:  # the Fortran library crashed hard before results.json could be written
            progress = workdir / "progress.txt"
            stages = progress.read_text().split() if progress.exists() else []
            log = workdir / "solver.log"
            res = {"ok": False, "stage": stages[-1] if stages else "start", "crashed": True,
                   "error": "The solver program crashed without finishing.",
                   "log": log.read_text(errors="replace")[-6000:] if log.exists() else ""}
        names = res.get("images") or {"mesh": "mesh.png", "equilibrium": "equilibrium.png"}
        images = {k: (workdir / v).read_bytes() for k, v in names.items() if (workdir / v).exists()}
        return {"results": res, "images": images, "stderr": stderr[-4000:], "design": design}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ------------------------------------------------------------------ results
def fmt(value, digits=2, scale=1.0):
    return "—" if value is None else f"{value * scale:.{digits}f}"


def show_convergence(solved):
    res = solved["results"]
    if res.get("ok"):
        st.success(f"Converged: TokaMaker found an equilibrium (error flag 0) in {res.get('elapsed_s', 0):.1f} s.")
        return
    stage = STAGES.get(res.get("stage"), res.get("stage"))
    flag = f" (error flag {res['err_flag']})" if res.get("err_flag") is not None else ""
    st.error(f"No equilibrium: the solve stopped during the **{stage}** step{flag}.  \n{res.get('error') or ''}")
    st.markdown("**Things to try:**\n"
                "- Leave more room between the target plasma and the vessel wall.\n"
                "- Move the PF coils closer to the plasma, and spread them around it so each has a job.\n"
                "- Ask for less: a smaller elongation κ or triangularity |δ|, or a lower Ip or core pressure.")
    if res.get("crashed"):
        with st.expander("Crash details (for your teacher)"):
            st.caption("Error output of the solver program (its full output is in the solver log below).")
            st.code(solved["stderr"].strip() or "(no error output)")


def show_equilibrium(images):
    if "equilibrium" in images:
        st.image(images["equilibrium"], width=520,
                 caption="Coloured lines are flux surfaces inside the plasma (surfaces the field lines stay "
                         "on); dashed grey lines are the field outside it. Red crosses are your target "
                         "boundary: if the plasma edge misses them, move the coils.")
    if "mesh" in images:
        with st.expander("Computational mesh"):
            st.image(images["mesh"], width=480,
                     caption="TokaMaker splits the machine into small triangles and solves on them.")


def checklist(stats, design):
    """The notebook's six things to check (cell 48), each with an icon and a rule-of-thumb hint."""
    def shape(name, target, got, tol, pct):
        if got is None:
            return INFO, f"**{name}**: not available."
        off = abs(got - target) > tol
        within = f"±{pct}" if pct else f"±{tol:.2f}"
        return (WARN if off else OK), f"**{name}** {got:.2f} vs target {target:.2f} ({within}). " + (
            "Too far off: move the PF coils to pull the plasma toward the target shape." if off else "Close enough.")

    centroid = (stats.get("Ip_centroid") or [None])[0]
    items = [shape("Current centroid R (m)", design["major_radius"], centroid, 0.03 * design["major_radius"], "3 %"),
             shape("Elongation κ", design["elongation"], stats.get("kappa"), 0.05 * design["elongation"], "5 %"),
             shape("Triangularity δ", design["triangularity"], stats.get("delta"), 0.05, None)]
    q95 = stats.get("q_95")
    if q95 is None:
        items.append((INFO, "**q95**: not available."))
    elif q95 < 2.5:
        items.append((WARN, f"**q95** = {q95:.2f} is below 2.5, which risks disruptions. Lower Ip to raise it."))
    elif q95 > 4:
        items.append((WARN, f"**q95** = {q95:.2f} is well above 4. You could raise Ip to bring it down."))
    else:
        items.append((OK, f"**q95** = {q95:.2f} is in the usual 2.5–4 range."))
    items.append((INFO, f"**β_pol** = {fmt(stats.get('beta_pol'))} %. Compare with your target: if too low, "
                        "raise the core pressure; if too high, lower it."))
    items.append((INFO, f"**l_i** = {fmt(stats.get('l_i'))}. Compare with your target: to raise it, make the "
                        "profiles more peaked (smaller α or larger γ); to lower it, make them flatter."))
    return items


def show_stats(res, design):
    stats = res["stats"]
    centroid = (stats.get("Ip_centroid") or [None])[0]
    rows = [("Major radius R (m), current centroid", fmt(design["major_radius"]), fmt(centroid)),
            ("Major radius R (m), geometric centre", fmt(design["major_radius"]), fmt(stats.get("R_geo"))),
            ("Minor radius a (m)", fmt(design["minor_radius"]), fmt(stats.get("a_geo"))),
            ("Elongation κ", fmt(design["elongation"]), fmt(stats.get("kappa"))),
            ("Triangularity δ", fmt(design["triangularity"]), fmt(stats.get("delta"))),
            ("Plasma current Ip (MA)", fmt(design["Ip_target"], scale=1e-6), fmt(stats.get("Ip"), scale=1e-6)),
            ("Core pressure (kPa)", fmt(design["p_core_target"], 1, 1e-3), fmt(stats.get("P_ax"), 1, 1e-3)),
            ("q95, safety factor near the edge", "2.5–4", fmt(stats.get("q_95"))),
            ("q0, safety factor at the centre", "", fmt(stats.get("q_0"))),
            ("β_pol, poloidal beta (%)", "your target", fmt(stats.get("beta_pol"))),
            ("β_tor, toroidal beta (%)", "", fmt(stats.get("beta_tor"))),
            ("β_N, normalized beta", "", fmt(stats.get("beta_n"))),
            ("l_i, internal inductance", "your target", fmt(stats.get("l_i")))]
    left, right = st.columns([1, 1])
    left.dataframe(pd.DataFrame(rows, columns=["Quantity", "Target", "Achieved"]), hide_index=True,
                   width="stretch", height=35 * len(rows) + 38)
    with right:
        st.markdown("**Checklist**")
        for icon, text in checklist(stats, design):
            st.markdown(f"{icon} {text}")
    with st.expander("What do these numbers mean?"):
        st.markdown(
            "- **q (safety factor)**: how many times a magnetic field line goes the long way around the "
            "torus for each trip the short way around. Too low (q95 < 2.5) and the plasma tends to disrupt.\n"
            "- **β (beta)**: plasma pressure divided by magnetic pressure: how hard the magnet is working. "
            "β_pol compares with the field made by the plasma current; β_tor with the toroidal field B0; "
            "β_N = β_tor (%) × a (m) × B0 (T) ÷ Ip (MA) is a 'how close to the limit' number (about 3–4).\n"
            "- **l_i (internal inductance)**: how concentrated the current is near the centre. Peaked "
            "profiles give a higher l_i.")


def show_coils(res):
    currents = res["coil_currents"]
    limit = (res.get("coil_current_limit") or 2 * res["stats"]["Ip"]) / 1e6
    amps = [abs(v) / 1e6 for v in currents.values()]
    fig = go.Figure(go.Bar(x=list(currents), y=amps, marker_color=["#d62728" if a > limit else "#1f77b4" for a in amps],
                           hovertemplate="%{x}: %{y:.2f} MA<extra></extra>"))
    fig.add_hline(y=limit, line_dash="dash", line_color="#d62728", annotation_text=f"limit 2 × Ip = {limit:.2f} MA")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="|coil current| (MA)",
                      dragmode=False, xaxis_fixedrange=True, yaxis_fixedrange=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    over = [name for name, a in zip(currents, amps) if a > limit]
    if over:
        st.warning(f"{', '.join(over)} {'is' if len(over) == 1 else 'are'} above 2 × Ip. Try:\n"
                   "- moving those coils closer to the plasma (you may need to reshape the vessel);\n"
                   "- moving coils apart from one another, so they don't have to fight each other.")
    else:
        st.success("Every coil current is below 2 × Ip.")


def show_stability(res):
    stab = res.get("stability")
    if not stab:
        st.info(f"The vertical-stability check could not be completed, but the equilibrium above is still "
                f"valid. {res.get('stability_error') or ''}")
        return
    value = stab.get("feedback_capability")
    c1, c2, c3 = st.columns(3)
    c1.metric("Feedback capability parameter", fmt(value))
    c2.metric("Growth rate (1/s)", fmt(stab.get("growth_rate"), 0))
    c3.metric("Wall time (ms)", fmt(stab.get("wall_time"), 1, 1e3))
    st.caption("Growth rate: how fast the plasma would drift up or down. Wall time: how long currents in the "
               "vessel wall can slow it. Their product must be below 2 for a control system to keep up.")
    if value is not None and value <= 0:
        st.success("No growing vertical motion was found: this plasma is vertically stable on its own.")
    elif value is not None and value < 2:
        st.success("Below 2: a feedback control system can hold this plasma in place.")
    elif value is not None:
        st.warning("2 or more: the plasma would slip away too fast. Bring the vacuum vessel closer to the "
                   "plasma; currents induced in the wall resist the plasma's motion.")


def show_results(solved, stale):
    st.divider()
    st.header("Results")
    if stale:
        st.warning("Your design changed since the last solve. The results below are for the old design; "
                   "press **Solve equilibrium** to update them.", icon=":material/history:")
    res = solved["results"]
    st.subheader("1. Did it converge?")
    show_convergence(solved)
    if solved["images"]:
        st.subheader("2. Equilibrium" if "equilibrium" in solved["images"] else "2. Mesh")
        show_equilibrium(solved["images"])
    if res.get("stats"):
        st.subheader("3. Equilibrium statistics")
        show_stats(res, solved["design"])
    if res.get("coil_currents"):
        st.subheader("4. Coil currents")
        show_coils(res)
    if res.get("ok"):
        st.subheader("5. Vertical stability")
        show_stability(res)
    if res.get("log"):
        with st.expander("Solver log"):
            st.code(res["log"])


# ------------------------------------------------------------------ page
st.set_page_config(page_title="Tokamaker Demo", layout="wide")
# Streamlit leaves a tall blank band above the title; trim it so the cross-section fits on screen,
# and lift the sidebar so its heading lines up with the page title.
st.markdown("<style>.block-container {padding-top: 1.5rem;}"
            '[data-testid="stSidebarUserContent"] {margin-top: -1.2rem;}</style>', unsafe_allow_html=True)
if "vv_base" not in st.session_state:
    reset_design()
vv_df, coil_df = controls()
design = read_design(vv_df, coil_df)
messages = validate_design(design)
solved = st.session_state.get("solved")
stale = solved is not None and solved["design"] != design

logo, heading = st.columns([1, 12], vertical_alignment="center")
logo.image(str(LOGO), width=90)
heading.title("Tokamaker Demo")
st.markdown(
    "Design a tokamak and find its **equilibrium**: the state where the plasma's outward push is exactly "
    "balanced by magnetic forces (the Grad–Shafranov equation, solved by TokaMaker).")
with st.expander("How to use this page"):
    st.markdown(
        "1. **Shape**: set the target plasma shape in the sidebar.\n"
        "2. **Build**: choose the points that define the vacuum vessel that surrounds the plasma and place the PF coils outside it. The picture updates live.\n"
        "3. **Physics**: choose the field, plasma current, pressure and profile shapes.\n"
        "4. **Solve and improve**: press *Solve equilibrium*, check the results against your device targets "
        "(shape, q95, β_pol, l_i, coil currents below 2 × Ip, feedback parameter below 2), adjust, and solve again.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Cross-section")
    lcfs = None if stale or solved is None else solved["results"].get("lcfs")
    st.plotly_chart(cross_section(design, flagged_objects(messages), lcfs), width="stretch",
                    config={"displayModeBar": False})
    st.caption("You are looking at a vertical slice through one side of the torus; the machine's central "
               "axis is at R = 0, off to the left. Red objects have a problem (see the checks on the right).")
with right:
    st.subheader("Profile shapes")
    st.plotly_chart(profile_figure(design), width="stretch", config={"displayModeBar": False})
    st.subheader("Check and solve")
    for m in messages:
        (st.error if m["level"] == "error" else st.warning)(m["message"])
    has_errors = any(m["level"] == "error" for m in messages)
    if st.button("Solve equilibrium", type="primary", disabled=has_errors, width="stretch",
                 help="Fix the errors above first." if has_errors else "Takes a few seconds."):
        with st.spinner("TokaMaker is solving for the equilibrium. This takes a few seconds..."):
            with solve_slots():
                st.session_state.solved = run_solver(design)
        st.rerun()

if solved is not None:
    show_results(solved, stale)
