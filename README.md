# Statistical Pharmacology via Kakutani Dichotomy I

**The Marginal Drug Kakutani Index and the α<sub>c</sub> = 1/2 Criticality Theorem in High-Dimensional Conformational Ensembles**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23005921.svg)](https://doi.org/10.5281/zenodo.23005921)

Zhengyi Chen<sup>1</sup>, Ruqing Chen<sup>2</sup>

<sup>1</sup> Guangxi Key Laboratory of Drug Discovery and Optimization, School of Pharmacy, Guilin Medical University, Guilin 541199, P. R. China — chenzhengyi@glmc.edu.cn
<sup>2</sup> GUT Geoservice Inc., Montreal, Québec, Canada — ruqing@hotmail.com

Paper I of the series *Statistical Pharmacology via Kakutani Dichotomy*, which transports the central mechanism of *Localized Bost–Connes systems* (R. Chen, 2026, DOI 10.5281/zenodo.22883179) — a phase transition that is a tail event of an infinite product, invisible to every finite factor — to the statistical geometry of protein conformational ensembles.

## Summary

Two ligands with the same binding free energy and residue-by-residue almost identical conformational footprints can nevertheless drive a protein into disjoint downstream functional states. We model the joint conformational law as a product measure on `{0,1}^N` and show that the question "are the two ensembles close?" has, as `N → ∞`, only two answers (Kakutani, 1948): equivalent or mutually singular.

- **Marginal drug Kakutani index (DKI)** `K_N = Σ h_i` (sum of local squared Hellinger distances), **Kakutani affinity** `Π_N = Π (1 − h_i/2)`, **doubling tail** `D_tail(N) = K_{2N} − K_N`.
- **Lemma 2.3.** Exact quartic expansion of the local Hellinger distance; at `p = 1/2`, `h(δ) = 2 − √(1+2δ) − √(1−2δ) = δ² + (5/4)δ⁴ + (21/8)δ⁶ + …`.
- **Theorem 3.1 (phase trichotomy).** For `q_i = 1/2 + c·i^{−α}`, `c = 0.1`, `A = c² = 0.01`:
  `α < 1/2`: `K_N ~ A N^{1−2α}/(1−2α)` (singular); `α = 1/2`: `K_N = A(ln N + γ) + C_∞(1/2) + O(1/N)` (singular, `Π_N ~ N^{−A/2}`); `α > 1/2`: `K_N → A ζ(2α) + C_∞(α)` (equivalent).
- **Proposition 3.2.** At criticality `D_tail(N) → A ln 2 = 0.00693147`.
- **Theorem 4.1 / Corollary 4.2.** The regression exponent in a finite window is `γ_eff = (1−2α) / (1 + (1−2α) ζ(2α) N^{2α−1})`; the upward bias near `α_c` is the pole of `ζ` at `2α = 1`, and the doubling-tail estimator cancels it identically.

Version 1.1 (2026-09-28): the general-`s` Euler–Maclaurin expansion in Lemma 3.2 is `H_N(s) = N^{1-s}/(1-s) + zeta(s) + N^{-s}/2 - s N^{-s-1}/12 + O(N^{-s-3})`; version 1.0 printed the `N^{-s-1}` term with a plus sign, and `paper01_compute_tables.py` evaluated the prediction column of Table 1 with that sign. Only the `N = 10` and `N = 100` rows of that column were affected (deviations of `1e-4` to `1e-6` that are now `1e-7` to `1e-11`); no theorem, constant or exponent uses the term.

Measured values at `N = 10^5` agree with the Euler–Maclaurin predictions to relative accuracy `10^{−10}`:

| α | K_N (measured) | K_N (analytic) |
|---|---|---|
| 0.25 | 6.311486 | 6.311486 |
| 0.50 | 0.1211103 | 0.1211103 |
| 0.75 | 0.0262136 | 0.0262136 |
| 1.00 | 0.01658727 | 0.01658727 |

## Repository layout

```
.
├── code/
│   ├── exp01_marginal_dki_criticality.py   # numerical experiment: data, figure, acceptance tests
│   └── paper01_compute_tables.py           # analytic constants, Euler–Maclaurin predictions, Tables 1–2
├── data/
│   ├── exp01_results_benchmark.csv         # K_N, Π_N, D_tail for α ∈ {0.25, 0.5, 0.75, 1}, N = 10^1 … 10^5
│   └── exp01_results_scaling.csv           # regression exponents γ_num(α), α = 0.05 … 0.45; critical slope
├── figures/
│   ├── fig01_marginal_dki_phase_transition.pdf   # four-panel figure, vector (Figure 1 of the paper)
│   └── fig01_marginal_dki_phase_transition.png   # the same figure, raster 200 dpi
├── paper/
│   ├── paper01_marginal_dki_criticality.tex      # LaTeX source (amsart)
│   └── paper01_marginal_dki_criticality.pdf      # compiled paper
├── results/
│   ├── paper01_table1.csv, paper01_table2.csv    # Tables 1 and 2 (measured + analytic columns)
│   ├── paper01_tables.json                       # every analytic constant used in the text
│   └── paper01_table*_rows.{tex,md}              # table bodies as pasted into the paper
├── CITATION.cff
├── LICENSE
├── requirements.txt
└── README.md
```

## Reproducing the results

Requirements: Python ≥ 3.10 with `numpy`, `matplotlib`, `mpmath` (see `requirements.txt`); a LaTeX distribution with `pdflatex` for the paper.

```bash
pip install -r requirements.txt

# 1. Numerical experiment → data/*.csv, figures/fig01_*.{pdf,png}, acceptance report on stdout
python code/exp01_marginal_dki_criticality.py

# 2. Analytic companion → results/paper01_table1.csv, paper01_table2.csv, paper01_tables.json, table rows
python code/paper01_compute_tables.py

# 3. Paper (run from paper/; the figure is found via \graphicspath{{../figures/}})
cd paper
pdflatex -interaction=nonstopmode paper01_marginal_dki_criticality.tex
pdflatex -interaction=nonstopmode paper01_marginal_dki_criticality.tex
```

Both scripts run in seconds. Step 1 prints `OVERALL ACCEPTANCE: PASS` when the four benchmark values, the exponent at α = 0.25 and the critical slope are all within tolerance. The scripts detect the repository layout automatically; run outside it, they read and write next to themselves.

Run on 27 September 2026 with Python 3.12.4, NumPy 1.26.4, Matplotlib 3.8.4, mpmath 1.3.0, MiKTeX 24.1 (pdfTeX 4.18).

## Citation

```bibtex
@article{ChenChen2026KakutaniI,
  author  = {Chen, Zhengyi and Chen, Ruqing},
  title   = {Statistical Pharmacology via Kakutani Dichotomy I: The Marginal Drug Kakutani Index
             and the $\alpha_c=1/2$ Criticality Theorem in High-Dimensional Conformational Ensembles},
  year    = {2026},
  doi     = {10.5281/zenodo.23005921},
  url     = {https://doi.org/10.5281/zenodo.23005921}
}
```

## Funding

This project was supported by the National Natural Science Foundation of China (No. 22464010), the Guangxi Natural Science Foundation of China (No. 2025GXNSFAA069294), and the 2025 Bagui Youth Top Talent Project.

## License

Code (`code/`) is released under the MIT License (see `LICENSE`). The paper text, figures and data (`paper/`, `figures/`, `data/`, `results/`) are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
