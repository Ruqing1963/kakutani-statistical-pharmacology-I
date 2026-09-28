#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp01_marginal_dki_criticality.py
=================================
Statistical Pharmacology via Kakutani Dichotomy, Paper I -- Experiment 01.
Minimal mathematical model and numerical experiment for the marginal drug
Kakutani index (DKI).

Model
-----
The conformational space of a protein is modelled by N approximately
independent two-state degrees of freedom X = (X_1, ..., X_N) in {0,1}^N.
  Ligand L_A :  p_i = p = 0.5
  Ligand L_B :  q_i = p + c * i^(-alpha),  c = 0.1
Local squared Hellinger distance
  h_i = (sqrt(p_i) - sqrt(q_i))^2 + (sqrt(1-p_i) - sqrt(1-q_i))^2
Cumulative DKI and Kakutani affinity
  K_N   = sum_{i<=N} h_i
  Pi_N  = prod_{i<=N} ( sqrt(p_i q_i) + sqrt((1-p_i)(1-q_i)) ) = prod_{i<=N} (1 - h_i/2)

Background (Kakutani dichotomy)
-------------------------------
For the product measures P_A = (x) Bern(p_i), P_B = (x) Bern(q_i),
      P_A ~ P_B  <=>  sum_i h_i < infinity   <=>  Pi_inf > 0
      P_A _|_ P_B  <=>  sum_i h_i = infinity  <=>  Pi_inf = 0
Since h_i ~ (q_i - p_i)^2 / (4 p (1-p)) = c^2 i^(-2 alpha) at p = 0.5,
the critical exponent is alpha_c = 1/2:
      alpha < 1/2 :  K_N ~ N^(1-2alpha)          (power-law divergence, singular)
      alpha = 1/2 :  K_N ~ (c^2/(4p(1-p))) ln N  (logarithmic divergence, singular)
      alpha > 1/2 :  K_N -> c^2 zeta(2 alpha)    (finite limit, equivalent)
This is the same mechanism as Theorem 3.6 of Chapter 1 of "Localized
Bost-Connes systems": every local factor is harmless and the transition is
decided by the square-summability of the tail.

Outputs
-------
  figures/fig01_marginal_dki_phase_transition.png   four-panel figure (raster, 200 dpi)
  figures/fig01_marginal_dki_phase_transition.pdf   the same figure as vector PDF (used by the paper)
  data/exp01_results_benchmark.csv                  K_N, Pi_N, D_tail(N) at N = 10^k
  data/exp01_results_scaling.csv                    gamma_num(alpha) vs gamma_th(alpha)
(When the script is not inside the repository layout, all files are written
next to the script.)
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------------
# Model parameters
# ----------------------------------------------------------------------------
P = 0.5            # activation probability under ligand A
C = 0.1            # perturbation amplitude
N_MAX = 10 ** 5    # largest system size
ALPHAS_BENCH = (0.25, 0.50, 0.75, 1.00)
ALPHAS_SCALING = np.round(np.arange(0.05, 0.45 + 1e-9, 0.05), 2)
ALPHA_C = 0.5
N_FIT_LO, N_FIT_HI = 10 ** 3, 10 ** 5

# Benchmark targets (fixed in advance) and acceptance tolerances
BENCH_TARGET = {0.25: 6.31, 0.50: 0.121, 0.75: 0.0262, 1.00: 0.0166}
BENCH_RTOL = 0.02          # 2 % relative tolerance (targets are given to 3 significant figures)
GAMMA_TARGET_025 = 0.502
GAMMA_ATOL = 0.01
SLOPE_TH_CRIT = C ** 2 / (4 * P * (1 - P))   # = 0.01
SLOPE_RTOL = 0.02

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _repo_dir(name: str) -> str:
    """Inside the repository layout (code/ data/ figures/ results/) write to the
    corresponding sub-directory; otherwise write next to the script."""
    d = os.path.join(ROOT, name)
    return d if os.path.isdir(d) else HERE


FIG_PATH = os.path.join(_repo_dir("figures"), "fig01_marginal_dki_phase_transition.png")
FIG_PATH_PDF = os.path.splitext(FIG_PATH)[0] + ".pdf"   # vector version for the paper
CSV_BENCH = os.path.join(_repo_dir("data"), "exp01_results_benchmark.csv")
CSV_SCALING = os.path.join(_repo_dir("data"), "exp01_results_scaling.csv")


# ----------------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------------
def hellinger_sq_bernoulli(p: np.ndarray | float, q: np.ndarray) -> np.ndarray:
    """Squared Hellinger distance h = sum_x (sqrt(p(x)) - sqrt(q(x)))^2 between two Bernoulli laws."""
    return (np.sqrt(p) - np.sqrt(q)) ** 2 + (np.sqrt(1.0 - p) - np.sqrt(1.0 - q)) ** 2


def dki_profile(alpha: float, n_max: int = 2 * N_MAX):
    """Return the cumulative K_N (DKI) and log Pi_N (log of the affinity, avoids underflow) for i = 1..n_max."""
    i = np.arange(1, n_max + 1, dtype=np.float64)
    q = P + C * i ** (-alpha)
    if np.any(q >= 1.0) or np.any(q <= 0.0):
        raise ValueError("q_i out of range: check p, c, alpha")
    h = hellinger_sq_bernoulli(P, q)
    K = np.cumsum(h)
    log_pi = np.cumsum(np.log1p(-0.5 * h))   # log(1 - h/2) to machine precision
    return K, log_pi


def k_at(K: np.ndarray, n: int) -> float:
    return float(K[n - 1])


def fit_loglog_exponent(K: np.ndarray, n_lo: int, n_hi: int, n_pts: int = 60) -> float:
    """Least-squares slope gamma of log K_N on log N over [n_lo, n_hi]."""
    ns = np.unique(np.logspace(np.log10(n_lo), np.log10(n_hi), n_pts).astype(int))
    x = np.log(ns.astype(float))
    y = np.log(K[ns - 1])
    return float(np.polyfit(x, y, 1)[0])


def fit_semilog_slope(K: np.ndarray, n_lo: int, n_hi: int, n_pts: int = 60) -> float:
    """Least-squares slope s of K_N on ln N over [n_lo, n_hi]."""
    ns = np.unique(np.logspace(np.log10(n_lo), np.log10(n_hi), n_pts).astype(int))
    x = np.log(ns.astype(float))
    y = K[ns - 1]
    return float(np.polyfit(x, y, 1)[0])


# ----------------------------------------------------------------------------
# Figure style (categorical slots 1-4 of the reference palette; text and axes in ink colours)
# ----------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {0.25: "#2a78d6", 0.50: "#eb6834", 0.75: "#1baf7a", 1.00: "#eda100"}
LW = 1.6   # about 2 px

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.unicode_minus": False,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 10,
    "axes.titlelocation": "left",
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.labelcolor": INK_2,
    "mathtext.fontset": "dejavusans",
})


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.6)


def alpha_label(alpha: float) -> str:
    return rf"$\alpha={alpha:.2f}$"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    ok_all = True
    print("=" * 78)
    print("Kakutani statistical pharmacology - Experiment 01 - marginal DKI criticality")
    print(f"p = {P}, c = {C}, N_max = {N_MAX:,}, alpha_c = {ALPHA_C}")
    print("=" * 78)

    # ---------- 1. Benchmarks ----------
    profiles = {a: dki_profile(a) for a in ALPHAS_BENCH}
    decades = [10 ** k for k in range(1, 6)]

    print("\n[1] Benchmarks: K_N, Pi_N, D_tail(N) = K_2N - K_N")
    bench_rows = []
    for a in ALPHAS_BENCH:
        K, log_pi = profiles[a]
        print(f"\n  alpha = {a:.2f}")
        print(f"  {'N':>8} {'K_N':>12} {'Pi_N':>12} {'D_tail(N)':>12}")
        for n in decades:
            kn = k_at(K, n)
            pin = float(np.exp(log_pi[n - 1]))
            dtail = k_at(K, 2 * n) - kn
            bench_rows.append((a, n, kn, pin, dtail))
            print(f"  {n:>8} {kn:>12.6g} {pin:>12.6g} {dtail:>12.6g}")

    print("\n  Acceptance (N = 100000):")
    print(f"  {'alpha':>6} {'K_N(num)':>12} {'K_N(target)':>12} {'rel.err':>9}  status")
    for a in ALPHAS_BENCH:
        kn = k_at(profiles[a][0], N_MAX)
        tgt = BENCH_TARGET[a]
        rel = abs(kn - tgt) / tgt
        passed = rel <= BENCH_RTOL
        ok_all &= passed
        print(f"  {a:>6.2f} {kn:>12.5g} {tgt:>12.5g} {rel:>9.2%}  {'PASS' if passed else 'FAIL'}")

    with open(CSV_BENCH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "N", "K_N", "Pi_N", "D_tail"])
        w.writerows(bench_rows)

    # ---------- 2. Scaling exponents ----------
    print("\n[2] Scaling exponents: regression of log K_N on log N (N in [1e3, 1e5])")
    print(f"  {'alpha':>6} {'gamma_num':>10} {'gamma_th':>10} {'diff':>9}")
    scaling_rows = []
    gamma_num = {}
    for a in ALPHAS_SCALING:
        K, _ = dki_profile(float(a), n_max=N_MAX)
        g = fit_loglog_exponent(K, N_FIT_LO, N_FIT_HI)
        g_th = 1.0 - 2.0 * float(a)
        gamma_num[float(a)] = g
        scaling_rows.append((float(a), g, g_th, g - g_th))
        print(f"  {a:>6.2f} {g:>10.4f} {g_th:>10.4f} {g - g_th:>+9.4f}")

    g025 = gamma_num[0.25]
    passed = abs(g025 - GAMMA_TARGET_025) <= GAMMA_ATOL
    ok_all &= passed
    print(f"\n  Acceptance: alpha = 0.25 -> gamma_num = {g025:.4f} (target {GAMMA_TARGET_025})"
          f"  {'PASS' if passed else 'FAIL'}")

    # Critical point: K_N vs ln N
    K_crit, _ = profiles[0.50]
    slope_crit = fit_semilog_slope(K_crit, N_FIT_LO, N_FIT_HI)
    rel = abs(slope_crit - SLOPE_TH_CRIT) / SLOPE_TH_CRIT
    passed = rel <= SLOPE_RTOL
    ok_all &= passed
    print(f"  Acceptance: alpha = 0.50 -> slope of K_N vs ln N = {slope_crit:.5f}"
          f" (theory c^2/(4p(1-p)) = {SLOPE_TH_CRIT:.4g}, rel.err {rel:.2%})  {'PASS' if passed else 'FAIL'}")

    with open(CSV_SCALING, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "gamma_num", "gamma_th", "diff"])
        w.writerows(scaling_rows)
        w.writerow([])
        w.writerow(["critical_slope_num", slope_crit, "critical_slope_th", SLOPE_TH_CRIT])

    # Panel C also shows the effective exponent for alpha > 0.5 (tends to 0) to complete the phase curve
    alphas_c = np.round(np.arange(0.05, 1.00 + 1e-9, 0.05), 2)
    gamma_c = []
    for a in alphas_c:
        if float(a) in gamma_num:
            gamma_c.append(gamma_num[float(a)])
        else:
            K, _ = dki_profile(float(a), n_max=N_MAX)
            gamma_c.append(fit_loglog_exponent(K, N_FIT_LO, N_FIT_HI))
    gamma_c = np.asarray(gamma_c)
    gamma_th_c = np.maximum(1.0 - 2.0 * alphas_c, 0.0)

    # ---------- 3. Four-panel figure ----------
    print("\n[3] Plotting ->", FIG_PATH)
    ns_plot = np.unique(np.logspace(1, 5, 300).astype(int))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), dpi=200)
    (axA, axB), (axC, axD) = axes

    # (A) log-log K_N vs N
    for a in ALPHAS_BENCH:
        K, _ = profiles[a]
        axA.plot(ns_plot, K[ns_plot - 1], color=SERIES[a], lw=LW, label=alpha_label(a))
        axA.annotate(alpha_label(a), xy=(ns_plot[-1], K[ns_plot[-1] - 1]),
                     xytext=(4, 0), textcoords="offset points", va="center",
                     fontsize=8, color=INK_2)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(10, 10 ** 5 * 2.2)
    axA.set_xlabel(r"$N$ (conformational degrees of freedom)")
    axA.set_ylabel(r"$K_N$ (DKI)")
    axA.set_title(r"A  $\log_{10}K_N$ vs $\log_{10}N$: power law / logarithm / finite limit")
    axA.legend(loc="upper left", title=None)
    tidy(axA)

    # (B) critical alpha=0.5: K_N vs ln N
    K_c = profiles[0.50][0]
    lnN = np.log(ns_plot.astype(float))
    axB.plot(lnN, K_c[ns_plot - 1], color=SERIES[0.50], lw=LW, label=r"$K_N$, $\alpha=0.50$")
    mask = (ns_plot >= N_FIT_LO) & (ns_plot <= N_FIT_HI)
    b_fit = np.polyfit(lnN[mask], K_c[ns_plot - 1][mask], 1)
    x_line = np.array([lnN[mask].min(), lnN[mask].max()])
    axB.plot(x_line, np.polyval(b_fit, x_line), color=INK_2, lw=1.0, ls="--",
             label=rf"regression: slope {slope_crit:.4f} (theory {SLOPE_TH_CRIT:.2f})")
    axB.set_xlabel(r"$\ln N$")
    axB.set_ylabel(r"$K_N$")
    axB.set_title(r"B  Critical point $\alpha_c=1/2$: $K_N \sim \frac{c^2}{4p(1-p)}\ln N$")
    axB.legend(loc="upper left")
    tidy(axB)

    # (C) effective exponent gamma(alpha)
    axC.plot(alphas_c, gamma_th_c, color=INK_2, lw=1.0, ls="--", label=r"theory $\gamma=\max(1-2\alpha,0)$")
    axC.plot(alphas_c, gamma_c, color=SERIES[0.25], lw=0, marker="o", ms=5.5,
             mec=SURFACE, mew=0.8, label=r"numerical $\gamma_{\mathrm{num}}$ ($N\in[10^3,10^5]$)")
    axC.axvline(ALPHA_C, color=AXIS, lw=0.8)
    axC.text(ALPHA_C + 0.015, 0.92, r"$\alpha_c=1/2$", color=INK_2, fontsize=8, va="top")
    axC.text(0.06, 0.05, r"$P_A\perp P_B$ (singular)", color=INK_2, fontsize=8)
    axC.text(0.62, 0.05, r"$P_A\sim P_B$ (equivalent)", color=INK_2, fontsize=8)
    axC.set_xlim(0, 1.05)
    axC.set_ylim(-0.05, 1.0)
    axC.set_xlabel(r"perturbation decay exponent $\alpha$")
    axC.set_ylabel(r"effective scaling exponent $\gamma$")
    axC.set_title(r"C  Phase diagram $\gamma(\alpha)$")
    axC.legend(loc="center right")
    tidy(axC)

    # (D) affinity product Pi_N
    for a in ALPHAS_BENCH:
        _, log_pi = profiles[a]
        pi = np.exp(log_pi[ns_plot - 1])
        axD.plot(ns_plot, pi, color=SERIES[a], lw=LW, label=alpha_label(a))
        # the two curves with alpha >= 0.75 nearly coincide at N = 1e5 (Pi ~ 0.99): legend only, no end labels
        if a <= 0.50:
            axD.annotate(alpha_label(a), xy=(ns_plot[-1], pi[-1]),
                         xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=8, color=INK_2)
    axD.set_xscale("log")
    axD.set_xlim(10, 10 ** 5 * 2.2)
    axD.set_ylim(0, 1.02)
    axD.set_xlabel(r"$N$ (conformational degrees of freedom)")
    axD.set_ylabel(r"$\Pi_N=\prod_i(1-h_i/2)$")
    axD.set_title(r"D  Collapse of the Kakutani affinity $\Pi_N$")
    axD.legend(loc="lower left")
    tidy(axD)

    for ax in (axA, axD):
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_minor_formatter(NullFormatter())

    fig.suptitle("Criticality of the marginal drug Kakutani index (DKI): "
                 "$q_i = p + c\\,i^{-\\alpha}$, $p=0.5$, $c=0.1$",
                 fontsize=11, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG_PATH)
    fig.savefig(FIG_PATH_PDF)
    plt.close(fig)

    print("\nOutput files:")
    print("  ", FIG_PATH)
    print("  ", FIG_PATH_PDF)
    print("  ", CSV_BENCH)
    print("  ", CSV_SCALING)
    print("\nOVERALL ACCEPTANCE:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
