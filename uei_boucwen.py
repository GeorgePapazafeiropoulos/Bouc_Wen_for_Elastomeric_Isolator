"""
Bouc-Wen hysteresis modelling of an unreinforced elastomeric isolator (UEI)
===========================================================================

Reproduces the analysis of the manuscript:
  1. Normalized Bouc-Wen model + two comparison models (pinching; Liu & Li
     amplitude-dependent).
  2. Parameter identification by differential evolution.
  3. Identifiability analysis by profile likelihood (warm-started continuation).
  4. Model reduction (fix sigma=1, n=1; keep k_x, k_w, rho) with a case study.
  5. Leave-one-out prediction of a held-out vertical load.

Data format (Analytical_Raw.xlsx): three side-by-side blocks, one per vertical
load, each with columns [Displacement (mm), Force (kN)]; header row at the top.

References
----------
- Bouc R. (1967). Forced vibrations of mechanical systems with hysteresis.
  Proc. 4th Conf. on Nonlinear Oscillations, Prague.
- Wen Y.K. (1976). Method for random vibration of hysteretic systems.
  J. Eng. Mech. Div. 102(2):249-263.
- Ikhouane F., Rodellar J. (2007). Systems with Hysteresis. Wiley.  [normalized form]
- Baber T.T., Noori M.N. (1985). Random vibration of degrading, pinching systems.
  J. Eng. Mech. 111(8):1010-1026.  [pinching model]
- Liu S., Li A. (2018). Hysteretic friction behavior of aluminum foam/polyurethane
  interpenetrating phase composites. Compos. Struct. 203:18-29.  [amplitude-dependent model]
- Storn R., Price K. (1997). Differential evolution. J. Glob. Optim. 11(4):341-359.
- Byrd R.H., Lu P., Nocedal J., Zhu C. (1995). A limited memory algorithm for
  bound constrained optimization. SIAM J. Sci. Comput. 16(5):1190-1208.  [L-BFGS-B]
- Raue A. et al. (2009). Structural and practical identifiability analysis ...
  profile likelihood. Bioinformatics 25(15):1923-1929.

License: MIT
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize


def mean_squared_error(y, y_pred):
    """MSE (same value as sklearn's, without its per-call validation overhead)."""
    d = np.asarray(y, float) - np.asarray(y_pred, float)
    return float(np.dot(d, d) / d.size)


def r2_score(y, y_pred):
    """Coefficient of determination (same value as sklearn's)."""
    y = np.asarray(y, float)
    ss_res = np.sum((y - np.asarray(y_pred, float)) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_data(path, loads=(20, 30, 40)):
    """Load side-by-side (displacement, force) blocks, one per vertical load.

    Returns {load: {'x': ndarray, 'f': ndarray}}.
    """
    raw = pd.read_excel(path, header=None)
    data = {}
    for i, load in enumerate(loads):
        x = pd.to_numeric(raw.iloc[2:, 2 * i], errors="coerce").dropna().to_numpy()
        f = pd.to_numeric(raw.iloc[2:, 2 * i + 1], errors="coerce").dropna().to_numpy()
        n = min(len(x), len(f))
        data[load] = {"x": x[:n], "f": f[:n]}
    return data


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class NormalizedBoucWen:
    """Normalized Bouc-Wen model (Ikhouane & Rodellar, 2007).

        F  = k_x * x + k_w * w
        dw = rho * [ dx - sigma*|dx|*|w|^(n-1)*w + (sigma-1)*dx*|w|^n ]

    k_x : elastic stiffness (kN/mm);  k_w : hysteretic force scale (kN);
    rho, sigma, n : dimensionless shape parameters.
    """

    def __init__(self, p):
        self.k_x, self.k_w = p["k_x"], p["k_w"]
        self.rho, self.sigma, self.n = p["rho"], p["sigma"], p["n"]

    def simulate(self, x):
        x = np.asarray(x, float)
        if len(x) < 2:
            return np.zeros_like(x), np.zeros_like(x)
        dx = np.gradient(x, edge_order=2).tolist()  # rate; dt = 1 (rate-independent)
        rho, sigma, n = float(self.rho), float(self.sigma), float(self.n)
        w_max = (1.0 / sigma) ** (1.0 / n)
        # plain-float loop (no NumPy scalars): same arithmetic, much faster
        w = [0.0] * len(dx)
        wi = 0.0
        for i in range(1, len(dx)):
            d = dx[i - 1]
            aw = math.fabs(wi) ** n            # |w|^(n-1)*w == sign(w)*|w|^n (finite at w=0, n<1)
            sw = aw if wi > 0 else (-aw if wi < 0 else 0.0)
            wi = wi + rho * (d - sigma * math.fabs(d) * sw + (sigma - 1) * d * aw)
            wi = -w_max if wi < -w_max else (w_max if wi > w_max else wi)
            w[i] = wi
        w = np.asarray(w)
        return self.k_x * x + self.k_w * w, w


class PinchingBoucWen:
    """Pinching Bouc-Wen model (degrading-pinching family, Baber & Noori 1985).

    Bell-shaped pinching function h(x) = 1 - xi*exp(-x^2 / x_c) multiplies the
    hysteretic rate (manuscript Eq. 9; rho scales only the dx term):

        dw = h(x) * [ rho*dx - sigma*|dx|*|w|^(n-1)*w + (sigma-1)*dx*|w|^n ]

    xi (depth, -), x_c (width, mm^2).
    """

    def __init__(self, p):
        self.k_x, self.k_w = p["k_x"], p["k_w"]
        self.rho, self.sigma, self.n = p["rho"], p["sigma"], p["n"]
        self.xi, self.x_c = p["xi"], p["x_c"]

    def simulate(self, x):
        x = np.asarray(x, float)
        if len(x) < 2:
            return np.zeros_like(x), np.zeros_like(x)
        dx = np.gradient(x, edge_order=2).tolist()
        xs = x.tolist()
        rho, sigma, n = float(self.rho), float(self.sigma), float(self.n)
        xi, x_c = float(self.xi), float(self.x_c)
        w_max = (1.0 / sigma) ** (1.0 / n)
        w = [0.0] * len(dx)
        wi = 0.0
        for i in range(1, len(dx)):
            d = dx[i - 1]
            h = 1.0 - xi * math.exp(-xs[i - 1] ** 2 / x_c)
            aw = math.fabs(wi) ** n
            sw = aw if wi > 0 else (-aw if wi < 0 else 0.0)
            wi = wi + h * (rho * d - sigma * math.fabs(d) * sw + (sigma - 1) * d * aw)
            wi = -w_max if wi < -w_max else (w_max if wi > w_max else wi)
            w[i] = wi
        w = np.asarray(w)
        return self.k_x * x + self.k_w * w, w


class AmplitudeDependentBoucWen:
    """Amplitude-dependent model (Liu & Li, 2018).

        z_dot = A*dx - beta*|dx|*z*|z|^(n-1) - gamma*dx*|z|^n       (Eq. 6)
        k_d(dU) = p*dU^2 + q*dU + k0 , clamped at 0 (Eqs. 8-9)
        F = k_d(dU)*x + alpha*z                                     (Eq. 7)

    dU is the running peak amplitude; n is fixed at 2 in the reference.
    Constraints (Eq. 12): A>0, beta+gamma>0, beta-gamma>=0, alpha>1, p<0.
    """

    def __init__(self, p):
        self.alpha = p["alpha"]
        self.p, self.q, self.k0 = p["p"], p["q"], p["k0"]
        self.A, self.beta, self.gamma = p["A"], p["beta"], p["gamma"]
        self.n = p.get("n", 2.0)

    def k_d(self, dU):
        return np.maximum(self.p * dU ** 2 + self.q * dU + self.k0, 0.0)   # Eq. 9 clamp

    def simulate(self, x):
        x = np.asarray(x, float)
        if len(x) < 2:
            return np.zeros_like(x), np.zeros_like(x)
        dx = np.gradient(x, edge_order=2).tolist()
        A, beta, gamma, n = float(self.A), float(self.beta), float(self.gamma), float(self.n)
        denom = beta + gamma
        z_max = (A / denom) ** (1.0 / n) if denom > 0 else math.inf
        bounded = math.isfinite(z_max)
        z = [0.0] * len(dx)
        zi = 0.0
        for i in range(1, len(dx)):
            d = dx[i - 1]
            az = math.fabs(zi) ** n
            sz = az if zi > 0 else (-az if zi < 0 else 0.0)
            zi = zi + (A * d - beta * math.fabs(d) * sz - gamma * d * az)
            if bounded:
                zi = -z_max if zi < -z_max else (z_max if zi > z_max else zi)
            z[i] = zi
        z = np.asarray(z)
        dU = np.maximum.accumulate(np.abs(x))
        return self.k_d(dU) * x + self.alpha * z, z


# ----------------------------------------------------------------------------
# Objective and identification
# ----------------------------------------------------------------------------
def objective(values, model_cls, names, fixed, data):
    """MSE between model and measurement, plus admissibility penalties."""
    p = dict(fixed)
    for k, v in zip(names, values):
        p[k] = v
    penalty = 0.0
    if "sigma" in p and p["sigma"] < 0.5:
        penalty += 100 * (0.5 - p["sigma"]) ** 2
    if "xi" in p:
        if p["xi"] < 0:
            penalty += 100 * p["xi"] ** 2
        elif p["xi"] >= 1:
            penalty += 100 * (p["xi"] - 1) ** 2
    if {"A", "beta", "gamma", "p"} <= set(p):          # Liu & Li Eq. 12
        if p["beta"] + p["gamma"] <= 0:
            penalty += 100 * (p["beta"] + p["gamma"]) ** 2 + 10
        if p["beta"] - p["gamma"] < 0:
            penalty += 100 * (p["beta"] - p["gamma"]) ** 2 + 10
        if p["p"] > 0:
            penalty += 100 * p["p"] ** 2
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fp, _ = model_cls(p).simulate(data["x"])   # extreme trial params may overflow
        if not np.all(np.isfinite(fp)):
            return 1e10
        return mean_squared_error(data["f"], fp) + penalty
    except Exception:
        return 1e10


def identify(model_cls, bounds, data, fixed=None, maxiter=500, popsize=10, seed=None):
    """Identify free parameters by differential evolution with L-BFGS-B polish.

    bounds : {name: (lo, hi)} for the FREE parameters.
    fixed  : {name: value} held constant.  Returns (params, metrics).
    """
    fixed = fixed or {}
    names = list(bounds.keys())
    res = differential_evolution(
        objective, [bounds[k] for k in names],
        args=(model_cls, names, fixed, data),
        maxiter=maxiter, popsize=popsize, polish=True, seed=seed, tol=1e-2,
    )
    p = dict(fixed)
    for k, v in zip(names, res.x):
        p[k] = float(v)
    fp, _ = model_cls(p).simulate(data["x"])
    metrics = _metrics(data["f"], fp)
    metrics["force_pred"] = fp
    return p, metrics


def _metrics(f_exp, f_mod):
    rmse = float(np.sqrt(mean_squared_error(f_exp, f_mod)))
    return {
        "R2": float(r2_score(f_exp, f_mod)),
        "RMSE": rmse,
        "NRMSE": float(rmse / (np.max(f_exp) - np.min(f_exp))),
    }


# ----------------------------------------------------------------------------
# Profile likelihood (warm-started continuation; Raue et al. 2009)
# ----------------------------------------------------------------------------
def profile_likelihood(model_cls, bounds, data, best_params, param,
                       n_fine=6, n_coarse=6, threshold=1.10):
    """Profile the RMSE of one parameter.

    Freezes `param` on a grid dense near the optimum and reaching the bounds;
    at each node the remaining parameters are re-optimized by L-BFGS-B,
    warm-started from the neighbouring node (continuation anchored at the best
    fit). Returns a dict with the grid, profiled RMSE, acceptable range and its
    normalized width.
    """
    names = list(bounds.keys())
    idx = names.index(param)
    lo, hi = bounds[param]
    free_names = [n for n in names if n != param]
    free_bounds = [bounds[n] for n in free_names]
    base_rmse = _metrics(data["f"], model_cls(best_params).simulate(data["x"])[0])["RMSE"]
    limit = base_rmse * threshold
    best_val = best_params[param]
    best_free = np.array([best_params[n] for n in free_names])

    # grid: fine (relative) near best_val + coarse stretch to the bounds
    fr = np.linspace(0, 1, n_fine + 1)[1:] * 0.5
    st = np.linspace(0, 1, n_coarse + 1)[1:] ** 1.7
    grid = np.unique(np.concatenate([
        [best_val], best_val * (1 + fr), best_val * (1 - fr),
        best_val + st * (hi - best_val), best_val - st * (best_val - lo)]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    b_i = int(np.argmin(np.abs(grid - best_val)))

    rmse = np.full(len(grid), np.nan)
    rmse[b_i] = base_rmse

    def reopt(val, inits):
        best = (np.inf, None)
        for x0 in inits:
            def wrapped(v):
                p = dict(zip(free_names, np.clip(v, [b[0] for b in free_bounds],
                                                  [b[1] for b in free_bounds])))
                p[param] = val
                return objective([p[n] for n in names], model_cls, names, {}, data)
            r = minimize(wrapped, x0, method="L-BFGS-B",
                         bounds=free_bounds, options={"maxiter": 120})
            xf = np.clip(r.x, [b[0] for b in free_bounds], [b[1] for b in free_bounds])
            p = dict(zip(free_names, xf)); p[param] = val
            m = _metrics(data["f"], model_cls(p).simulate(data["x"])[0])["RMSE"]
            if m < best[0]:
                best = (m, xf)
        return best

    for direction in (+1, -1):                       # continuation outward
        prev = best_free.copy()
        i = b_i + direction
        while 0 <= i < len(grid):
            m, xf = reopt(grid[i], [prev, best_free])
            rmse[i] = m
            if xf is not None:
                prev = xf
            i += direction

    acc = rmse <= limit
    if acc.any():
        lo_a, hi_a = float(grid[acc].min()), float(grid[acc].max())
        width = (hi_a - lo_a) / (hi - lo)
    else:
        lo_a = hi_a = None
        width = np.inf
    return {"grid": grid, "rmse": rmse, "base_rmse": base_rmse, "limit": limit,
            "best_val": best_val, "acceptable": (lo_a, hi_a), "width": width}


# ----------------------------------------------------------------------------
# Leave-one-out prediction of a held-out load
# ----------------------------------------------------------------------------
def predict_heldout(reduced_params, data, cal_loads, held_load, free=("k_x", "k_w", "rho")):
    """Linearly interpolate the reduced-model parameters from `cal_loads` to
    `held_load` and evaluate against the measured held-out response."""
    (la, pa), (lb, pb) = sorted((L, reduced_params[L]) for L in cal_loads)
    frac = (held_load - la) / (lb - la)
    pred = {k: pa[k] + (pb[k] - pa[k]) * frac for k in free}
    pred["sigma"], pred["n"] = 1.0, 1.0
    fp, _ = NormalizedBoucWen(pred).simulate(data[held_load]["x"])
    return pred, _metrics(data[held_load]["f"], fp)


# ----------------------------------------------------------------------------
# Example driver
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "Analytical_Raw.xlsx"
    LOADS = (20, 30, 40)
    data = load_data(DATA_PATH, LOADS)

    bounds_norm = {"k_x": (0.01, 5.0), "k_w": (0.001, 3.0), "rho": (0.01, 5.0),
                   "sigma": (0.5, 1.5), "n": (0.5, 5.0)}

    # --- (1) full normalized fit per load ---
    full = {}
    for L in LOADS:
        p, m = identify(NormalizedBoucWen, bounds_norm, data[L], maxiter=500, popsize=10)
        full[L] = {**p, **m}
        print(f"[full]    {L} kN  R2={m['R2']:.4f}  NRMSE={m['NRMSE']:.4f}  "
              f"k_x={p['k_x']:.3f} k_w={p['k_w']:.3f} rho={p['rho']:.3f} "
              f"sigma={p['sigma']:.3f} n={p['n']:.3f}")

    # --- (2) reduced model: fix sigma=1, n=1 ; free k_x, k_w, rho ---
    bounds_red = {"k_x": (0.01, 5.0), "k_w": (0.001, 3.0), "rho": (0.01, 5.0)}
    reduced = {}
    for L in LOADS:
        p, m = identify(NormalizedBoucWen, bounds_red, data[L],
                        fixed={"sigma": 1.0, "n": 1.0}, maxiter=300, popsize=12)
        reduced[L] = {**p, **m}
        print(f"[reduced] {L} kN  R2={m['R2']:.4f}  NRMSE={m['NRMSE']:.4f}  "
              f"k_x={p['k_x']:.3f} k_w={p['k_w']:.3f} rho={p['rho']:.3f}")

    # --- (3) profile likelihood for one load (e.g. 40 kN) ---
    bp = {k: full[40][k] for k in ("k_x", "k_w", "rho", "sigma", "n")}
    for prm in ("k_x", "k_w", "rho", "sigma", "n"):
        pl = profile_likelihood(NormalizedBoucWen, bounds_norm, data[40], bp, prm)
        print(f"[profile] 40 kN  {prm:5s}  acceptable={pl['acceptable']}  "
              f"width={pl['width']:.4f}")

    # --- (4) leave-one-out prediction of 30 kN from 20 & 40 kN ---
    pred, m = predict_heldout(reduced, data, cal_loads=(20, 40), held_load=30)
    print(f"[predict] 30 kN  R2={m['R2']:.4f}  NRMSE={m['NRMSE']:.4f}  "
          f"(k_w interp={pred['k_w']:.3f}, fitted={reduced[30]['k_w']:.3f})")
