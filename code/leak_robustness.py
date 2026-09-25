#!/usr/bin/env python3
"""Table S10: sensitivity of the leak estimate to key assumptions.

Counts are new leaks per year in the Titas network. Mains use Eq. (4) with
reference length L; risers use Eq. (5), riser prevalence x number of risers /
mean riser-leak duration tau. Rows are printed in the order of Table S10.

Run after build_model.py.   Usage:  python3 leak_robustness.py
"""
import math, build_model as B, robustness_analyses as R, leak_comparison as L

KM, RISERS = L.NET_KM_TITAS, L.N_RISERS
COMPLAINTS = 4891          # leak complaints, FY 2021-22 (Prothom Alo, 2023a)
RECORDED_1819 = 5876       # leaks recorded by Titas, FY 2018-19 (TBS, 2020); not complaints

# Pipeline Leak under the two alternative verbal-to-numeric scales, as printed
# (rounded) by robustness_analyses.py (58.63% and 58.83%). The rounded values are
# kept so that the rows match Table S10 exactly.
ALT_SCALES = [("Scale 0.10/0.25/0.50/0.75/0.90", 0.5863),
              ("Scale 0.10/0.30/0.50/0.70/0.90", 0.5883)]

# Riser prevalence in the later Titas project: 17,072 of 143,394 = 11.906%.
# Table S10 uses the rounded 11.9% (0.119); Table 4 uses the unrounded share.
RISER_LATER = 0.119


def mains(p, unit_km=1.0, k=None):
    """New leaks per year on the mains. Poisson by default; with k, a negative
    binomial count with dispersion k: P(>=1) = 1 - (1 + m/k)^(-k)."""
    if k is None:
        m = -math.log(1 - p)
    else:
        m = k * ((1 - p) ** (-1.0 / k) - 1)
    return m / unit_km * KM


def row(label, pf, pr, unit=1.0, tau=1.0, det=1.0, repeat=0.0, reports=COMPLAINTS, k=None):
    m = mains(pf, unit, k)
    r = pr * RISERS / (det * tau)
    tot = m + r
    print(f"{label:<46}{m:>10,.0f}{tot:>12,.0f}{tot / (reports * (1 - repeat)):>8.1f}")


if __name__ == "__main__":
    nodes, links = R.fresh()
    b = B.marginals(nodes, links)
    pf, pr = b["Pipeline_Failure"], b["Riser_Failure"]
    print(f"{'assumption':<46}{'mains':>10}{'total':>12}{'ratio':>8}")
    row("Base case (1 km, tau = 1 year)", pf, pr)
    row("Reference length 0.5 km", pf, pr, unit=0.5)
    row("Reference length 2 km", pf, pr, unit=2.0)
    row("Mean riser-leak duration tau = 0.5 year", pf, pr, tau=0.5)
    row("Mean riser-leak duration tau = 3 years", pf, pr, tau=3.0)
    row("Negative binomial count, k = 1", pf, pr, k=1.0)
    row("Negative binomial count, k = 0.5", pf, pr, k=0.5)
    for lab, p_alt in ALT_SCALES:
        row(lab, p_alt, pr)
    row("Riser anchor 11.9% (later project)", pf, RISER_LATER)
    row("Riser detection efficiency 0.8", pf, pr, det=0.8)
    row("Riser detection efficiency 0.6", pf, pr, det=0.6)
    row("30% of complaints are repeat calls", pf, pr, repeat=0.3)
    row("50% of complaints are repeat calls", pf, pr, repeat=0.5)
    row("Leaks recorded in FY 2018-19 (5,876)", pf, pr, reports=RECORDED_1819)
