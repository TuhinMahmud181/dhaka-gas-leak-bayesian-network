#!/usr/bin/env python3
"""Convert model probabilities to new leaks per year and compare with Titas
records (Table 4, Fig. 4 and Section 3.3 of the paper).

The model has two terminal nodes:
  Pipeline Leak (Pipeline_Failure)  P(at least one NEW leak within one year on a
                                    reference length L = 1 km of main)
  Riser Leak (Riser_Failure)        P(a given riser is leaking at a given time);
                                    a prevalence, anchored to the 2017 riser survey

Mains (Eq. 4): new leaks on L in one year taken as Poisson with mean m*L, so
  m = -ln(1 - P) / L        new leaks per km per year (scales as 1/L)
  mains new leaks = m x network length
Risers (Eq. 5, Little's law): prevalence = incidence x mean leak duration TAU, so
  riser new leaks = P(riser leak) x number of risers / TAU

The modelled new leaks per year are set against the leak complaints Titas
receives in a year; the inverse ratio is an upper bound on the share of new
leaks that lead to a complaint. TAU has not been measured: 1 year is the
reference and 0.5-3 years are reported. About 90% of the total is the riser
term, which scales the survey anchor and does not depend on the network. The
Titas-wide counts extrapolate Dhaka conditions to the whole network.

The mains survey counts leaks PRESENT when surveyed, not new leaks per year, so
the mains comparison is a check of order of magnitude only (Section 2.9).

Run after build_model.py.   Usage:  python3 leak_comparison.py
"""
import os
import numpy as np
import build_model as B
import robustness_analyses as R

# ---------------------------------------------------------------- observations
# Each value is a reported figure; sources are given in the paper.
NET_KM_TITAS = 13320          # Titas distribution network, km
NET_KM_DHAKA = 7000           # of which in Dhaka
N_RISERS = 1_200_000          # Titas risers (12 lakh)
UNIT_KM = 1.0                 # reference length L of the Pipeline Leak node
TAU = 1.0                     # mean riser-leak duration, years (reference value)
TAU_RANGE = (0.5, 1.0, 3.0)   # durations reported in the paper

SURVEY_KM = 1682              # Titas leak-detection survey, FY 2021-22
SURVEY_LEAKS = (459, 985)     # 459: TBS and Prothom Alo (primary); 985: one
                              # secondary report (The Daily Star), not a range
SURVEY_METHANE = 9926         # locations where methane was detected
RISER_OBS = {                 # riser surveys: (leaking, inspected)
    "2017 baseline (calibration anchor)": (35252, 565952),
    "later inspection project (check)":   (17072, 143394),
}
REPORTED = {                  # leaks reported to / recorded by Titas in one year
    "FY 2021-22": 4891,       # leak complaints (Prothom Alo, 2023a)
    "FY 2018-19": 5876,       # leaks recorded by Titas, not complaints (TBS, 2020)
}


def density(p, unit=UNIT_KM):
    return -np.log(1.0 - p) / unit


def counts(m, km=NET_KM_TITAS, tau=TAU):
    """New leaks per year on the mains, at risers and in total."""
    mains = density(m["Pipeline_Failure"]) * km
    risers = m["Riser_Failure"] * N_RISERS / tau
    return mains, risers, mains + risers


def main():
    nodes, links = R.fresh()
    base = B.marginals(nodes, links)
    rng = np.random.default_rng(20260905)   # same draws as build_model.py base sweep
    draws = [B.marginals(nodes, B.perturb(links, rng)) for _ in range(B.N_DRAWS)]

    def band(f):
        v = np.array([f(d) for d in draws])
        return f(base), np.percentile(v, 5), np.percentile(v, 95)

    print("=" * 76)
    print("1. MODEL PROBABILITIES")
    print("=" * 76)
    for k in ("Pipeline_Failure", "Riser_Failure"):
        b, lo, hi = band(lambda d: d[k])
        print(f"  {B.LABEL[k]:<14}{100*b:7.1f}%   [{100*lo:.1f}-{100*hi:.1f}]")

    print("\n" + "=" * 76)
    print("2. MAINS: MODEL NEW LEAKS PER KM PER YEAR vs SURVEY (leaks present per km)")
    print("=" * 76)
    b, lo, hi = band(lambda d: density(d["Pipeline_Failure"]))
    print(f"  model, L = 1 km              {b:6.2f}   [{lo:.2f}-{hi:.2f}]")
    for u in (0.5, 2.0):
        print(f"  model, L = {u:.1f} km            "
              f"{density(base['Pipeline_Failure'], u):6.2f}")
    print(f"  Titas survey, leaks found    {SURVEY_LEAKS[0]/SURVEY_KM:6.2f} (primary); "
          f"{SURVEY_LEAKS[1]/SURVEY_KM:.2f} (secondary report)")
    print(f"  Titas survey, methane found  {SURVEY_METHANE/SURVEY_KM:6.2f}")

    print("\n" + "=" * 76)
    print("3. RISER LEAK PREVALENCE: MODEL vs RISER SURVEYS")
    print("=" * 76)
    b, lo, hi = band(lambda d: d["Riser_Failure"])
    print(f"  model                        {100*b:5.1f}%  [{100*lo:.1f}-{100*hi:.1f}]")
    for lab, (k, n) in RISER_OBS.items():
        print(f"  {lab:<36}{100*k/n:5.1f}%")

    print("\n" + "=" * 76)
    print(f"4. MODELLED NEW LEAKS PER YEAR vs ANNUAL COMPLAINTS, Titas network "
          f"({NET_KM_TITAS:,} km, {N_RISERS:,} risers, TAU = {TAU:g} yr)")
    print("=" * 76)
    mb = band(lambda d: counts(d)[0])
    rb = band(lambda d: counts(d)[1])
    tb = band(lambda d: counts(d)[2])
    print(f"  {'':<24}{'base':>10}{'5th pct':>10}{'95th pct':>10}")
    for lab, (b, lo, hi) in (("mains new leaks", mb), ("riser new leaks", rb),
                             ("total new leaks", tb)):
        print(f"  {lab:<24}{b:>10,.0f}{lo:>10,.0f}{hi:>10,.0f}")
    obs_m = [SURVEY_LEAKS[0] / SURVEY_KM * NET_KM_TITAS,
             SURVEY_LEAKS[1] / SURVEY_KM * NET_KM_TITAS]
    obs_r = [k / n * N_RISERS for k, n in RISER_OBS.values()]
    obs_t = [obs_m[0] + min(obs_r), obs_m[1] + max(obs_r)]
    print(f"  {'scaled field records':<24}{obs_t[0]:>10,.0f} - {obs_t[1]:,.0f}"
          f"   (NOT independent: reuses the riser anchor)")
    print()
    print(f"  {'reports':<14}{'count':>7}{'model: new per report':>30}"
          f"{'scaled records per report':>28}")
    for yr, n in REPORTED.items():
        rt = (tb[0] / n, tb[1] / n, tb[2] / n)
        print(f"  {yr:<14}{n:>7,}{rt[0]:>14.0f}  [{rt[1]:.0f}-{rt[2]:.0f}]"
              f"{obs_t[0]/n:>22.0f}-{obs_t[1]/n:.0f}")

    print("\n  riser-leak duration (base parameters):")
    for tau in TAU_RANGE:
        m_, r_, t_ = counts(base, tau=tau)
        c = REPORTED["FY 2021-22"]
        print(f"    TAU = {tau:<4g} yr   total {t_:>9,.0f}   per complaint {t_/c:5.1f}"
              f"   upper bound on share reported {100*c/t_:5.1f}%")

    print("\n" + "=" * 76)
    print(f"5. INTERVENTIONS: NEW LEAKS AVOIDED PER YEAR (Titas network, TAU = {TAU:g} yr)")
    print("=" * 76)
    t0 = counts(base)[2]
    print(f"  {'scenario':<48}{'mains':>9}{'risers':>10}{'total':>10}{'avoided':>10}")
    for k, lab, ev in B.scenarios(nodes):
        mm, rr, tt = counts(B.marginals(nodes, links, ev))
        print(f"  {k} {lab:<46}{mm:>9,.0f}{rr:>10,.0f}{tt:>10,.0f}"
              f"{t0 - tt:>10,.0f}")
    mm, rr, tt = counts(B.marginals(nodes, links, {"Riser_Installation_Quality_": "Good"}))
    print(f"  - {'Riser installation to Good (Sec. 3.5, S8)':<46}{mm:>9,.0f}{rr:>10,.0f}"
          f"{tt:>10,.0f}{t0 - tt:>10,.0f}")



if __name__ == "__main__":
    main()
