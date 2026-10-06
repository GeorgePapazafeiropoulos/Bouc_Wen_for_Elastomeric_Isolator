"""
Figure generation and model diagnostics for the UEI Bouc-Wen study (parallel).

Reads Analytical_Raw.xlsx (next to this script) and writes, in the script folder:

  fig_model_fit_<load>kN.svg               reduced-model fit vs experiment (per load)
  fig_parameters_vs_load_<param>.svg       identified parameters vs vertical load
  fig_prediction.svg                       leave-one-out held-out-load prediction
  fig_profile_likelihood_<param>_<load>kN.svg
                                           profile-likelihood curves (every load)
  hysteresis_fitting_results_<load>kN_<panel>.svg
                                           experimental data and the 3 model fits
  best_models_comparison_<load>kN.svg      normalized Bouc-Wen fit vs measured (per load)
  diagnostic_comparison_<metric>.svg       global optimality / NRMSE / R2 / score
  bouc_wen_results.xlsx                    parameters and metrics of every model fit

and prints the diagnostics report (global optimality, identifiability by
multi-start CoV and profile likelihood, accuracy, overall score).
This single script replaces the former bouc_wen_analysis.py,
diagnostics_analysis_corrected.py and the serial plots.py.

Requirements: Python 3 (tested on 3.10), numpy, pandas, scipy, matplotlib, openpyxl, and
uei_boucwen.py in the same folder.   Run:  python plots.py
"""

import os
import sys

# Every worker process runs scalar Python code: keep BLAS/OpenMP to one thread per
# process so N workers do not oversubscribe the cores. Must precede the numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Never let a non-UTF-8 Windows console code page crash print().
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # files only: safe headless and in workers
import matplotlib.pyplot as plt

import uei_boucwen as U

BASE_DIR = Path(__file__).resolve().parent

LABEL = {"k_x": r"$k_x$", "k_w": r"$k_w$", "rho": r"$\rho$",
         "sigma": r"$\sigma$", "n": r"$n$"}


# ----------------------------------------------------------------------------
def plot_model_fit(data, reduced, loads, fname="fig_model_fit.svg"):
    """Reduced model (sigma=1, n=1) vs experimental loops, one figure per load
    (fig_model_fit_<load>kN.svg)."""
    stem, ext = fname.rsplit(".", 1)
    for L in loads:
        fig, ax = plt.subplots(figsize=(7, 6.5))
        d, p = data[L], reduced[L]
        fp, _ = U.NormalizedBoucWen(
            {"k_x": p["k_x"], "k_w": p["k_w"], "rho": p["rho"],
             "sigma": 1.0, "n": 1.0}).simulate(d["x"])
        ax.scatter(d["x"], d["f"], s=3, c="blue", alpha=0.35, label="Experimental")
        ax.plot(d["x"], fp, "r-", lw=2.2, label="Reduced model")
        ax.set_title(f"{L} kN", fontsize=15)
        ax.set_xlabel("Displacement (mm)", fontsize=13)
        ax.set_ylabel("Force (kN)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.text(0.03, 0.97,
                (f"{LABEL['k_x']}={p['k_x']:.3f}, {LABEL['k_w']}={p['k_w']:.3f}, "
                 f"{LABEL['rho']}={p['rho']:.3f}\n"
                 f"($\\sigma$=1, $n$=1)  R$^2$={p['R2']:.4f}, NRMSE={p['NRMSE']:.4f}"),
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="black"))
        ax.legend(loc="lower right", fontsize=11)
        fig.tight_layout()
        out = f"{stem}_{L}kN.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


# ----------------------------------------------------------------------------
def _profile_task(data_L, bounds, bp, prm, n_fine, n_coarse):
    return U.profile_likelihood(U.NormalizedBoucWen, bounds, data_L, bp, prm,
                                n_fine=n_fine, n_coarse=n_coarse)


def _identify_task(model_cls, bounds, data_L, fixed, maxiter, popsize, seed):
    return U.identify(model_cls, bounds, data_L, fixed=fixed,
                      maxiter=maxiter, popsize=popsize, seed=seed)


def plot_profile_likelihood(profiles, load, params=("k_x", "k_w", "rho", "sigma", "n"),
                            fname="fig_profile_likelihood.svg"):
    """One figure per parameter for one load (fig_profile_likelihood_<param>_<load>kN.svg);
    `profiles` is {param: result} from U.profile_likelihood."""
    stem, ext = fname.rsplit(".", 1)
    for prm in params:
        pl = profiles[prm]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(pl["grid"], pl["rmse"], "b-o", ms=4, lw=2, label="Profile RMSE")
        ax.axhline(pl["base_rmse"], color="g", ls="--", lw=1.5, label="Base RMSE")
        ax.axhline(pl["limit"], color="r", ls="--", lw=1.5, label="+10% RMSE limit")
        ax.axvline(pl["best_val"], color="purple", ls=":", lw=1.8, label="Best-fit value")
        lo, hi = pl["acceptable"]
        if lo is not None and hi is not None and hi > lo:
            ax.axvspan(lo, hi, alpha=0.2, color="green", label="Acceptable region")
        ax.set_xlabel(LABEL[prm], fontsize=14)
        ax.set_ylabel("RMSE (kN)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out = f"{stem}_{prm}_{load}kN.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


# ----------------------------------------------------------------------------
def plot_parameters_vs_load(reduced, loads, cal_loads=(20, 40),
                            fname="fig_parameters_vs_load.svg"):
    """Identified k_x, k_w, rho vs vertical load, each saved as its own figure
    (fig_parameters_vs_load_<param>.svg), with interpolation line."""
    loads = sorted(loads)
    keys = ["k_x", "k_w", "rho"]
    units = {"k_x": r"$k_x$ (kN/mm)", "k_w": r"$k_w$", "rho": r"$\rho$"}
    la, lb = sorted(cal_loads)
    stem, ext = fname.rsplit(".", 1)
    for key in keys:
        fig, ax = plt.subplots(figsize=(6, 5))
        v = np.array([reduced[L][key] for L in loads])
        ax.plot(loads, v, "o", ms=10, color="navy", zorder=3, label="Identified")
        va, vb = reduced[la][key], reduced[lb][key]
        ax.plot([la, lb], [va, vb], "r--", lw=1.8, label=f"Linear ({la} & {lb} kN)")
        for L in loads:
            if la < L < lb:
                vi = va + (vb - va) * (L - la) / (lb - la)
                ax.plot(L, vi, "r^", ms=11, zorder=3, label=f"Interpolated at {L} kN")
        ax.set_xlabel("Vertical load (kN)", fontsize=12)
        ax.set_ylabel(units[key], fontsize=13)
        ax.set_xticks(loads)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        out = f"{stem}_{key}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


# ----------------------------------------------------------------------------
def plot_prediction(data, pred_params, metrics, held_load, cal_loads=(20, 40),
                    fname="fig_prediction.svg"):
    """Held-out prediction: predicted loops vs measured loops."""
    d = data[held_load]
    fp, _ = U.NormalizedBoucWen(pred_params).simulate(d["x"])
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(d["x"], d["f"], s=4, c="0.45", alpha=0.5,
               label=f"Measured ({held_load} kN)")
    ax.plot(d["x"], fp, "r-", lw=2.3,
            label=f"Predicted (from {cal_loads[0]} & {cal_loads[1]} kN)")
    ax.set_xlabel("Displacement (mm)", fontsize=13)
    ax.set_ylabel("Force (kN)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.97,
            (f"Held-out prediction of {held_load} kN\n"
             f"R$^2$={metrics['R2']:.4f}, NRMSE={metrics['NRMSE']:.4f}"),
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="black"))
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Comparison of the three model families (formerly bouc_wen_analysis.py)
# ----------------------------------------------------------------------------
# Lower bound of the exponent n. The manuscript models evaluate |w|^(n-1)*w, which is
# undefined (0*inf) at w = 0 for n < 1, so every n < 1 candidate was infeasible and the
# reported fits and profiles (Fig. 6, Tables 2-4) effectively search n in [1, 5].
# uei_boucwen.py uses the equivalent sign(w)*|w|^n form, which would otherwise let the
# optimizer run to n = 0.5; keep N_MIN = 1.0 to reproduce the paper, 0.5 to explore.
N_MIN = 1.0

BOUNDS_NORM = {"k_x": (0.01, 5.0), "k_w": (0.001, 3.0), "rho": (0.01, 5.0),
               "sigma": (0.5, 1.5), "n": (N_MIN, 5.0)}
BOUNDS_RED = {"k_x": (0.01, 5.0), "k_w": (0.001, 3.0), "rho": (0.01, 5.0)}
BOUNDS_AMP = {"alpha": (0.01, 0.9), "p": (-1.0, -0.001), "q": (0.0, 2.0),
              "k0": (0.01, 5.0), "A": (0.1, 2.0), "beta": (0.01, 2.0),
              "gamma": (0.01, 2.0), "n": (N_MIN, 5.0)}
BOUNDS_PINCH = {"k_x": (0.01, 5.0), "k_w": (0.001, 3.0), "rho": (0.01, 5.0),
                "sigma": (0.5, 1.5), "n": (N_MIN, 5.0), "xi": (0.01, 0.99),
                "x_c": (0.1, 5.0)}

# (key, label, colour, model class, bounds); 'normalized' reuses the full fit
MODELS = [("normalized", "Normalized Bouc-Wen", "blue", U.NormalizedBoucWen, BOUNDS_NORM),
          ("amplitude_dependent", "Amplitude-Dependent", "green",
           U.AmplitudeDependentBoucWen, BOUNDS_AMP),
          ("pinching", "Pinching", "red", U.PinchingBoucWen, BOUNDS_PINCH)]


def build_results(full, extra):
    """results[load][model] = {'params', 'metrics'}; best model by R^2 per load."""
    results = {L: {"normalized": {"params": p, "metrics": m}}
               for L, (p, m) in full.items()}
    for (L, key), (p, m) in extra.items():
        results[L][key] = {"params": p, "metrics": m}
    best = {L: max(r.items(), key=lambda kv: kv[1]["metrics"]["R2"])
            for L, r in results.items()}
    return results, best


def plot_results(data, results):
    """One figure per load/panel (hysteresis_fitting_results_<load>kN_<panel>.svg)."""
    for L in sorted(data):
        d = data[L]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(d["x"], d["f"], "k.", markersize=1.5, alpha=0.4, label="Data")
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Force (kN)")
        ax.set_title(f"{L} kN - Experimental Data")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"hysteresis_fitting_results_{L}kN_experimental.svg",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        for key, label, color, _, _ in MODELS:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(d["x"], d["f"], "k.", markersize=1, alpha=0.2)
            m = results[L][key]["metrics"]
            ax.plot(d["x"], m["force_pred"], color, linewidth=1.5, label="Fitted")
            ax.text(0.05, 0.95, f"R$^2$ = {m['R2']:.4f}\nRMSE = {m['RMSE']:.4f}",
                    transform=ax.transAxes, verticalalignment="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.set_xlabel("Displacement (mm)")
            ax.set_ylabel("Force (kN)")
            ax.set_title(f"{L} kN - {label}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(f"hysteresis_fitting_results_{L}kN_{key}.svg",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_comparison(data, results):
    """One figure per load (best_models_comparison_<load>kN.svg): measured response
    and the fit of the normalized Bouc-Wen model."""
    for L in sorted(data):
        d = data[L]
        m = results[L]["normalized"]["metrics"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(d["x"], d["f"], "k.", markersize=1.5, alpha=0.4, label="Data")
        ax.plot(d["x"], m["force_pred"], "r-", linewidth=2, label="Normalized Bouc-Wen")
        ax.text(0.05, 0.95, f"R$^2$ = {m['R2']:.4f}\nRMSE = {m['RMSE']:.4f}",
                transform=ax.transAxes, verticalalignment="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Force (kN)")
        ax.set_title(f"{L} kN - Normalized Bouc-Wen Fit")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"best_models_comparison_{L}kN.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)


def export_results_to_excel(results, best, filename="bouc_wen_results.xlsx"):
    """Fit metrics and parameters of every model/load, plus a console summary."""
    rows = [{"Amplitude (kN)": L, "Model": key, "RMSE": r["metrics"]["RMSE"],
             "R2": r["metrics"]["R2"], "NRMSE": r["metrics"]["NRMSE"], **r["params"]}
            for L, mods in results.items() for key, r in mods.items()]
    try:
        pd.DataFrame(rows).to_excel(filename, index=False)
        print("saved", filename)
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
    print(f"{'Amplitude':<12} {'Best Model':<25} {'R2':<10} {'RMSE':<10}")
    for L, (key, r) in sorted(best.items()):
        print(f"{L} kN{'':<6} {key:<25} {r['metrics']['R2']:.4f}     {r['metrics']['RMSE']:.4f}")


# ----------------------------------------------------------------------------
# Diagnostics (formerly diagnostics_analysis_corrected.py)
#   1. global optimality  - spread of the objective over multi-start fits
#   2. identifiability    - parameter CoV over the same multi-start fits, and
#                           profile-likelihood width (+10 % RMSE band)
#   3. accuracy           - NRMSE and R2 of the full normalized fit
# ----------------------------------------------------------------------------
DIAG_STARTS = 12                          # multi-start fits per load
PARAMS = ("k_x", "k_w", "rho", "sigma", "n")


def _multistart_task(data_L, seed):
    """One multi-start differential-evolution fit; None if it fails."""
    try:
        return U.identify(U.NormalizedBoucWen, BOUNDS_NORM, data_L,
                          maxiter=200, popsize=8, seed=seed)
    except Exception:
        return None


def _rate(value, good, moderate, higher_is_better=False):
    """'good' / 'moderate' / 'poor' from two thresholds (NaN counts as poor)."""
    if higher_is_better:
        return "good" if value >= good else ("moderate" if value >= moderate else "poor")
    return "good" if value < good else ("moderate" if value < moderate else "poor")


def compute_diagnostics(starts, profiles, metrics):
    """Diagnostics of one load from its multi-start fits, profiles and full-fit metrics."""
    valid = [r for r in starts if r is not None]
    # objective = MSE; the admissibility penalty is zero inside the search bounds
    objs = np.array([m["RMSE"] ** 2 for _, m in valid])
    if len(objs) >= 2:
        spread = float(np.std(objs) / np.mean(objs)) if np.mean(objs) > 0 else np.inf
    else:
        spread = np.nan

    cov = {}
    for prm in PARAMS:
        vals = np.array([p[prm] for p, _ in valid])
        if len(vals) < 2:
            cov[prm] = np.inf
        else:
            cov[prm] = float(np.std(vals) / np.mean(vals)) if np.mean(vals) != 0 else np.inf

    width = {prm: profiles[prm]["width"] for prm in PARAMS}

    score = {"good": 1.0, "moderate": 0.5, "poor": 0.0}[_rate(spread, 0.05, 0.10)]
    score += sum({"good": 0.2, "moderate": 0.1, "poor": 0.0}[_rate(c, 0.1, 0.3)]
                 for c in cov.values())
    score += sum({"good": 0.2, "moderate": 0.1, "poor": 0.0}[_rate(w, 0.2, 0.5)]
                 for w in width.values())
    nrmse = metrics["NRMSE"]
    score += 1.0 if nrmse < 0.05 else 0.8 if nrmse < 0.10 else 0.5 if nrmse < 0.20 else 0.0
    max_score = 1 + 5 * 0.2 + 5 * 0.2 + 1
    return {"n_valid": len(valid), "spread": spread, "cov": cov, "width": width,
            "acceptable": {prm: profiles[prm]["acceptable"] for prm in PARAMS},
            "rmse": metrics["RMSE"], "nrmse": nrmse, "r2": metrics["R2"],
            "score": 100.0 * score / max_score}


_TAG = {"good": "OK", "moderate": "WARN", "poor": "POOR"}


def print_diagnostics(diag):
    """Console report: one table per diagnostic, one row per load."""
    loads = sorted(diag)
    print("\n" + "=" * 70)
    print("DIAGNOSTICS ACROSS LOADS")
    print("=" * 70)

    print("\n1. Global optimality (objective spread over multi-start fits, CoV):")
    for L in loads:
        d = diag[L]
        print(f"  {L} kN  valid starts {d['n_valid']}/{DIAG_STARTS}   "
              f"spread = {d['spread']:.4f}  [{_TAG[_rate(d['spread'], 0.05, 0.10)]}]")

    print("\n2a. Identifiability - multi-start parameter CoV:")
    print(f"  {'Load':<8}" + "".join(f"{p:>10}" for p in PARAMS))
    for L in loads:
        print(f"  {str(L) + ' kN':<8}" + "".join(f"{diag[L]['cov'][p]:10.4f}" for p in PARAMS))

    print("\n2b. Identifiability - profile-likelihood normalized width (+10% RMSE band):")
    print(f"  {'Load':<8}" + "".join(f"{p:>10}" for p in PARAMS))
    for L in loads:
        print(f"  {str(L) + ' kN':<8}" + "".join(f"{diag[L]['width'][p]:10.4f}" for p in PARAMS))

    print("\n3. Accuracy (full normalized fit):")
    print(f"  {'Load':<8}{'RMSE (kN)':>12}{'NRMSE':>10}{'R2':>10}")
    for L in loads:
        d = diag[L]
        print(f"  {str(L) + ' kN':<8}{d['rmse']:12.4f}{d['nrmse']:10.4f}{d['r2']:10.4f}")

    print("\n4. Overall diagnostic score:")
    for L in loads:
        s = diag[L]["score"]
        rating = ("Excellent" if s >= 80 else "Good" if s >= 60
                  else "Moderate" if s >= 40 else "Poor")
        print(f"  {L} kN: {s:.1f}%  {rating}")


def plot_diagnostic_comparison(diag, stem="diagnostic_comparison", ext="svg"):
    """Four separate bar charts: global optimality, NRMSE, R2, overall score."""
    loads = sorted(diag)
    labels = [f"{L} kN" for L in loads]
    specs = [  # file tag, key, ylabel, title, (good, moderate) lines + labels, better-high
        ("global_optimality", "spread", "Spread (CoV)", "Global Optimality",
         (0.05, 0.10), ("Good (5%)", "Moderate (10%)"), False),
        ("nrmse", "nrmse", "NRMSE", "Accuracy",
         (0.05, 0.10), ("Excellent (5%)", "Good (10%)"), False),
        ("r2", "r2", "R$^2$", "Goodness of Fit",
         (0.98, 0.95), ("Excellent (0.98)", "Good (0.95)"), True),
        ("score", "score", "Overall Score (%)", "Overall Diagnostic Score",
         (80, 60), ("Excellent (80%)", "Good (60%)"), True),
    ]
    for tag, key, ylabel, title, (g, m), (lg, lm), high in specs:
        vals = [diag[L][key] for L in loads]
        colors = [{"good": "green", "moderate": "orange", "poor": "red"}[_rate(v, g, m, high)]
                  for v in vals]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(labels, vals, color=colors)
        ax.axhline(y=g, color="green", linestyle="--", label=lg)
        ax.axhline(y=m, color="orange", linestyle="--", label=lm)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if tag == "score":                # bars reach the top: legend at the bottom of the axes
            ax.legend(loc="lower center")
        else:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = f"{stem}_{tag}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


# ----------------------------------------------------------------------------
# Task scheduling
# ----------------------------------------------------------------------------
def _submit(executor, fn, *args):
    """Run fn(*args) in the pool, or right away (as a finished Future) without one."""
    if executor is not None:
        return executor.submit(fn, *args)
    fut = Future()
    try:
        fut.set_result(fn(*args))
    except BaseException as e:            # re-raised at .result(), like a worker error
        fut.set_exception(e)
    return fut


def make_executor(n_tasks):
    """Process pool sized to this machine (None -> run serially)."""
    n_workers = max(1, min(os.cpu_count() or 1, n_tasks))
    if n_workers == 1:
        return None, 1
    try:
        return ProcessPoolExecutor(max_workers=n_workers), n_workers
    except (OSError, ImportError, NotImplementedError, ValueError):
        return None, 1                    # restricted environment: stay serial


def main():
    os.chdir(BASE_DIR)                    # inputs and outputs live next to the script
    DATA_PATH = "Analytical_Raw.xlsx"
    if not Path(DATA_PATH).exists():
        raise SystemExit(f"{DATA_PATH} not found next to {Path(__file__).name}.")
    LOADS = (20, 30, 40)
    HELD = 30
    CAL = (20, 40)
    data = U.load_data(DATA_PATH, LOADS)

    n_tasks = (len(LOADS) * (1 + len(MODELS) - 1 + 1 + DIAG_STARTS + len(PARAMS)))
    ex, n_workers = make_executor(n_tasks)
    print(f"Using {n_workers} worker process(es) for {n_tasks} tasks")
    try:
        # Longest tasks first (the 7-8 parameter models), then the full and reduced
        # fits, then the diagnostic multi-starts; profiles follow once a full fit exists.
        f_extra = {(L, key): _submit(ex, _identify_task, cls, bnds, data[L], None, 500, 10, 1)
                   for L in LOADS for key, _, _, cls, bnds in MODELS[1:]}
        f_full = {L: _submit(ex, _identify_task, U.NormalizedBoucWen, BOUNDS_NORM,
                             data[L], None, 500, 10, 1) for L in LOADS}
        f_red = {L: _submit(ex, _identify_task, U.NormalizedBoucWen, BOUNDS_RED, data[L],
                            {"sigma": 1.0, "n": 1.0}, 300, 12, 1) for L in LOADS}
        f_ms = {L: [_submit(ex, _multistart_task, data[L], 42 + i * 7)
                    for i in range(DIAG_STARTS)] for L in LOADS}

        full = {L: f.result() for L, f in f_full.items()}
        f_prof = {L: {prm: _submit(ex, _profile_task, data[L], BOUNDS_NORM,
                                   {k: full[L][0][k] for k in PARAMS}, prm, 6, 6)
                      for prm in PARAMS} for L in LOADS}

        # figures that only need the (fast) fits
        reduced = {L: {**p, **m} for L, (p, m) in
                   ((L, f.result()) for L, f in f_red.items())}
        plot_model_fit(data, reduced, LOADS)
        plot_parameters_vs_load(reduced, LOADS, cal_loads=CAL)
        pred, mpred = U.predict_heldout(reduced, data, cal_loads=CAL, held_load=HELD)
        plot_prediction(data, pred, mpred, HELD, cal_loads=CAL)

        results, best = build_results(full, {k: f.result() for k, f in f_extra.items()})
        plot_results(data, results)
        plot_comparison(data, results)
        export_results_to_excel(results, best)

        # profile-likelihood figures and diagnostics
        profiles = {L: {prm: f.result() for prm, f in f_prof[L].items()} for L in LOADS}
        for L in LOADS:
            plot_profile_likelihood(profiles[L], L, PARAMS)
        diag = {L: compute_diagnostics([f.result() for f in f_ms[L]], profiles[L],
                                       results[L]["normalized"]["metrics"]) for L in LOADS}
        print_diagnostics(diag)
        plot_diagnostic_comparison(diag)
    finally:
        if ex is not None:
            ex.shutdown(wait=True)
    print("All figures generated.")


if __name__ == "__main__":
    multiprocessing.freeze_support()      # needed for frozen Windows executables
    main()
