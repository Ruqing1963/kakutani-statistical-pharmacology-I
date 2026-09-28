#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paper01_compute_tables.py
=========================
Analytic companion of Paper I: supplies the Euler-Maclaurin prediction columns
of Tables 1 and 2 of paper01_marginal_dki_criticality.tex and independently
checks every analytic constant quoted in the text.

Outputs (results/ inside the repository layout, otherwise next to the script)
----------------------------------------------------------------------------
  paper01_tables.json       all analytic quantities, for item-by-item citation in the paper
  paper01_table1.csv        Table 1: measured K_N, Pi_N, D_tail with analytic K_N^{EM} and relative deviation
  paper01_table2.csv        Table 2: gamma_th, gamma_num, gamma_EM(1e4), gamma_EM^fit, gamma_tail
  paper01_table*_rows.tex   table bodies in LaTeX, pasted verbatim into the paper
  paper01_table*_rows.md    the same in Markdown
"""
from __future__ import annotations

import csv
import json
import os

import mpmath as mp
import numpy as np

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _repo_dir(name: str) -> str:
    """Inside the repository layout (code/ data/ results/) use the corresponding
    sub-directory; otherwise use the directory of the script."""
    d = os.path.join(ROOT, name)
    return d if os.path.isdir(d) else HERE


DATA_DIR = _repo_dir("data")      # reads exp01_results_*.csv
RES_DIR = _repo_dir("results")    # writes paper01_table*.csv / .json / _rows.*
P, C = 0.5, 0.1
A = C ** 2 / (4 * P * (1 - P))          # = 0.01
N_FIT_LO, N_FIT_HI = 10 ** 3, 10 ** 5

# ---------------------------------------------------------------------------
# 1. Exact Taylor coefficients of the local Hellinger distance at p = 1/2:
#    h(delta) = sum_k a_k delta^{2k},  h = 2 - (sqrt(1+x) + sqrt(1-x)),  x = 2 delta
# ---------------------------------------------------------------------------
def hellinger_series_coeffs(kmax: int = 6):
    """Return a_1..a_kmax with h(delta)|_{p=1/2} = sum_k a_k delta^{2k}."""
    coeffs = []
    for k in range(1, kmax + 1):
        b = mp.binomial(mp.mpf(1) / 2, 2 * k)
        # sqrt(1+x)+sqrt(1-x) = 2 sum_{even n} b_n x^n  =>  h = -2 sum_{k>=1} b_{2k} x^{2k},  x = 2 delta
        coeffs.append(-2 * b * mp.mpf(2) ** (2 * k))
    return coeffs


A_K = hellinger_series_coeffs(6)   # a_1 = 1, a_2 = 5/4, a_3 = 21/8, ...


def h_exact(delta: float) -> float:
    q = P + delta
    return float((mp.sqrt(P) - mp.sqrt(q)) ** 2 + (mp.sqrt(1 - P) - mp.sqrt(1 - q)) ** 2)


# Fourth-order coefficients for general p (Lemma 2.3)
def lemma23_coeffs(p: float):
    p = mp.mpf(p)
    c2 = 1 / (4 * p * (1 - p))
    c3 = -(1 - 2 * p) / (8 * p ** 2 * (1 - p) ** 2)
    c4 = 5 * (1 - 3 * p + 3 * p ** 2) / (64 * p ** 3 * (1 - p) ** 3)
    return float(c2), float(c3), float(c4)


# ---------------------------------------------------------------------------
# 2. Generalised harmonic numbers H_N(s) = sum_{i<=N} i^{-s}: exact values and Euler-Maclaurin expansion
# ---------------------------------------------------------------------------
def H_exact(N: int, s: float) -> mp.mpf:
    """Exact: H_N(s) = zeta(s) - zeta(s, N+1) (Hurwitz); harmonic number for s = 1."""
    s = mp.mpf(s)
    if s == 1:
        return mp.harmonic(N)
    return mp.zeta(s) - mp.zeta(s, N + 1)


def H_em(N: int, s: float, order: int = 2) -> mp.mpf:
    """Truncated Euler-Maclaurin expansion (up to the 1/(2N^s) and s/(12 N^{s+1}) terms)."""
    s = mp.mpf(s)
    N = mp.mpf(N)
    if s == 1:
        val = mp.log(N) + mp.euler + 1 / (2 * N)
        if order >= 2:
            val -= 1 / (12 * N ** 2)
        return val
    val = N ** (1 - s) / (1 - s) + mp.zeta(s) + N ** (-s) / 2
    if order >= 2:
        val -= s * N ** (-s - 1) / 12
    return val


def K_analytic(N: int, alpha: float, kmax: int = 6, em: bool = False) -> mp.mpf:
    """K_N(alpha) = sum_k a_k c^{2k} H_N(2k alpha), with H_N exact (Hurwitz) or from the EM expansion."""
    tot = mp.mpf(0)
    for k in range(1, kmax + 1):
        s = 2 * k * alpha
        Hk = H_em(N, s) if em else H_exact(N, s)
        tot += A_K[k - 1] * mp.mpf(C) ** (2 * k) * Hk
    return tot


def K_direct(N: int, alpha: float) -> float:
    i = np.arange(1, N + 1, dtype=np.float64)
    q = P + C * i ** (-alpha)
    h = (np.sqrt(P) - np.sqrt(q)) ** 2 + (np.sqrt(1 - P) - np.sqrt(1 - q)) ** 2
    return float(h.sum())


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    out = {}
    out["A"] = A
    out["hellinger_series_coeffs_p_half"] = [str(mp.nstr(a, 12)) for a in A_K]
    out["hellinger_series_coeffs_rational"] = [str(mp.identify(a)) for a in A_K]
    out["lemma23_coeffs_general_p"] = {
        "p=0.5": lemma23_coeffs(0.5), "p=0.1": lemma23_coeffs(0.1), "p=0.3": lemma23_coeffs(0.3)}
    # numerical check of Lemma 2.3
    for p_chk, d_chk in [(0.1, 0.05), (0.3, 0.05), (0.5, 0.1)]:
        q = p_chk + d_chk
        h_num = float((mp.sqrt(p_chk) - mp.sqrt(q)) ** 2 + (mp.sqrt(1 - p_chk) - mp.sqrt(1 - q)) ** 2)
        c2, c3, c4 = lemma23_coeffs(p_chk)
        h_ser = c2 * d_chk ** 2 + c3 * d_chk ** 3 + c4 * d_chk ** 4
        out[f"lemma23_check_p{p_chk}_d{d_chk}"] = {"h_exact": h_num, "h_series_to_4th": h_ser,
                                                   "abs_err": h_num - h_ser}
    out["h1_exact_delta0.1"] = h_exact(0.1)
    out["quartic_relative_correction_c0.1"] = float(A_K[1] * C ** 2)   # (5/4) c^2

    # constants
    out["euler_gamma"] = float(mp.euler)
    out["zeta"] = {s: float(mp.zeta(s)) for s in ["0.5", "0.9", "1.5", "2", "3", "4", "4.5", "6"]}
    out["C4_half_correct"] = float(A_K[1] * C ** 4 * mp.zeta(2))       # (5/4) c^4 zeta(2)
    out["C4_half_brief"] = float(mp.mpf(C) ** 4 / 4 * mp.zeta(2))      # c^4/4 zeta(2) as in the original brief (wrong coefficient)
    out["C6_half"] = float(A_K[2] * C ** 6 * mp.zeta(3))

    # analytic decomposition at alpha = 1/2
    N = 10 ** 5
    lead = A * (mp.log(N) + mp.euler + mp.mpf(1) / (2 * N))
    out["alpha0.5_N1e5"] = {
        "leading_A(lnN+gamma+1/2N)": float(lead),
        "leading_plus_C4_correct": float(lead + out["C4_half_correct"]),
        "leading_plus_C4_brief": float(lead + out["C4_half_brief"]),
        "full_series_k<=6_exact_H": float(K_analytic(N, 0.5)),
        "direct_sum": K_direct(N, 0.5),
    }
    # limits and truncation for alpha = 0.75, 1.00
    for al in (0.75, 1.0):
        Kinf = sum(A_K[k - 1] * mp.mpf(C) ** (2 * k) * mp.zeta(2 * k * al) for k in range(1, 7))
        tail_lead = A * mp.zeta(2 * al, N + 1)   # leading truncation remainder sum_{i>N} A i^{-2 alpha}
        out[f"alpha{al}_limit"] = {
            "K_inf_leading_A_zeta": float(A * mp.zeta(2 * al)),
            "K_inf_full_k<=6": float(Kinf),
            "truncation_tail_N1e5_leading": float(tail_lead),
            "truncation_tail_approx_2A/sqrtN_or_A/N": float(A * 2 / mp.sqrt(N)) if al == 0.75 else float(A / N),
            "K_N1e5_analytic": float(K_analytic(N, al)),
            "K_N1e5_direct": K_direct(N, al),
        }
    # alpha = 0.25
    out["alpha0.25_N1e5"] = {"analytic": float(K_analytic(N, 0.25)), "direct": K_direct(N, 0.25),
                             "leading_2A_sqrtN": float(2 * A * mp.sqrt(N)), "A_zeta_half": float(A * mp.zeta(0.5))}
    # critical doubling tail
    out["Dtail_half_limit_Aln2"] = float(A * mp.log(2))

    # ---------------- Table 1 ----------------
    bench = {}
    with open(os.path.join(DATA_DIR, "exp01_results_benchmark.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bench[(float(row["alpha"]), int(row["N"]))] = (float(row["K_N"]), float(row["Pi_N"]), float(row["D_tail"]))
    t1 = []
    for al in (0.25, 0.5, 0.75, 1.0):
        for n in (10, 100, 1000, 10000, 100000):
            k, pi, dt = bench[(al, n)]
            k_em = float(K_analytic(n, al, em=True))
            t1.append({"alpha": al, "N": n, "K_N": k, "Pi_N": pi, "D_tail": dt,
                       "K_EM": k_em, "rel_dev_EM": (k - k_em) / k})
    with open(os.path.join(RES_DIR, "paper01_table1.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(t1[0].keys())); w.writeheader(); w.writerows(t1)
    out["table1"] = t1

    # ---------------- Table 2 ----------------
    scal = []
    with open(os.path.join(DATA_DIR, "exp01_results_scaling.csv"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] not in ("alpha", "critical_slope_num"):
                scal.append((float(row[0]), float(row[1]), float(row[2])))
    ns = np.unique(np.logspace(np.log10(N_FIT_LO), np.log10(N_FIT_HI), 60).astype(int))
    t2 = []
    for al, g_num, g_th in scal:
        z = float(mp.zeta(2 * al))
        Nstar = 10 ** 4
        g_em = (1 - 2 * al) / (1 + (1 - 2 * al) * z * Nstar ** (2 * al - 1))
        # gamma_tail: regression of log D_tail on log N over the same grid
        i = np.arange(1, 2 * N_FIT_HI + 1, dtype=np.float64)
        q = P + C * i ** (-al)
        h = (np.sqrt(P) - np.sqrt(q)) ** 2 + (np.sqrt(1 - P) - np.sqrt(1 - q)) ** 2
        K = np.cumsum(h)
        dtail = K[2 * ns - 1] - K[ns - 1]
        g_tail = float(np.polyfit(np.log(ns.astype(float)), np.log(dtail), 1)[0])
        # analytic gamma_eff regressed over the same grid (not only at the centre): fit ln K_EM on ln N
        k_em_grid = np.array([float(K_analytic(int(n), al, kmax=1, em=True)) for n in ns])
        g_em_fit = float(np.polyfit(np.log(ns.astype(float)), np.log(k_em_grid), 1)[0])
        t2.append({"alpha": al, "gamma_th": g_th, "gamma_num": g_num, "zeta_2alpha": z,
                   "gamma_EM_1e4": g_em, "gamma_EM_fit": g_em_fit, "gamma_tail": g_tail})
    with open(os.path.join(RES_DIR, "paper01_table2.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(t2[0].keys())); w.writeheader(); w.writerows(t2)
    out["table2"] = t2

    # critical slope
    with open(os.path.join(DATA_DIR, "exp01_results_scaling.csv"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] == "critical_slope_num":
                out["critical_slope_num"] = float(row[1])

    with open(os.path.join(RES_DIR, "paper01_tables.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---------------- LaTeX / Markdown table bodies (pasted into the paper, no manual transcription) ----------------
    def sci(x: float) -> str:
        m, e = f"{x:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"

    lines_tex, lines_md = [], []
    for r in t1:
        lines_tex.append(f"{r['alpha']:.2f} & $10^{{{int(round(np.log10(r['N'])))}}}$ & {r['K_N']:.7g} & "
                         f"{r['Pi_N']:.6f} & {r['D_tail']:.6g} & {r['K_EM']:.7g} & {sci(r['rel_dev_EM'])} \\\\")
        lines_md.append(f"| {r['alpha']:.2f} | 10^{int(round(np.log10(r['N'])))} | {r['K_N']:.7g} | "
                        f"{r['Pi_N']:.6f} | {r['D_tail']:.6g} | {r['K_EM']:.7g} | {r['rel_dev_EM']:+.1e} |")
    with open(os.path.join(RES_DIR, "paper01_table1_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_tex) + "\n")
    with open(os.path.join(RES_DIR, "paper01_table1_rows.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_md) + "\n")

    lines_tex, lines_md = [], []
    for r in t2:
        lines_tex.append(f"{r['alpha']:.2f} & {r['gamma_th']:.4f} & {r['zeta_2alpha']:.4f} & {r['gamma_num']:.4f} & "
                         f"{r['gamma_EM_1e4']:.4f} & {r['gamma_EM_fit']:.4f} & {r['gamma_tail']:.4f} \\\\")
        lines_md.append(f"| {r['alpha']:.2f} | {r['gamma_th']:.4f} | {r['zeta_2alpha']:.4f} | {r['gamma_num']:.4f} | "
                        f"{r['gamma_EM_1e4']:.4f} | {r['gamma_EM_fit']:.4f} | {r['gamma_tail']:.4f} |")
    with open(os.path.join(RES_DIR, "paper01_table2_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_tex) + "\n")
    with open(os.path.join(RES_DIR, "paper01_table2_rows.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_md) + "\n")

    print(json.dumps({k: v for k, v in out.items() if k not in ("table1", "table2")}, indent=2))
    print("\nTable 1:")
    for r in t1:
        print(f"  a={r['alpha']:.2f} N={r['N']:>6} K={r['K_N']:.7g} Pi={r['Pi_N']:.6f} Dt={r['D_tail']:.6g} "
              f"K_EM={r['K_EM']:.7g} rel={r['rel_dev_EM']:+.2e}")
    print("\nTable 2:")
    for r in t2:
        print(f"  a={r['alpha']:.2f} th={r['gamma_th']:.4f} num={r['gamma_num']:.4f} zeta={r['zeta_2alpha']:+.4f} "
              f"EM(1e4)={r['gamma_EM_1e4']:.4f} EMfit={r['gamma_EM_fit']:.4f} tail={r['gamma_tail']:.4f}")


if __name__ == "__main__":
    main()
