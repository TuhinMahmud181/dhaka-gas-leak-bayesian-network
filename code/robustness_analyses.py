#!/usr/bin/env python3
"""Robustness and decomposition analyses for the Dhaka gas pipeline BN model.

Reproduces results reported in the manuscript:

  Section 3.2, Table S10   sensitivity of Pipeline Leak to the verbal-to-numeric
                           scale used to convert questionnaire answers (the
                           anchoring is solved again for each scale)
  Section 3.2              effect of relaxing root-node independence, via a
                           log-linear association model fitted by IPF
  Section 3.5, Table 5     intervention bundles on the mains (a burial-depth
                           variant is printed for reference; not reported)
  Section 3.4, Fig. 5      apportionment of Pipeline Leak across the five hazard
                           pathways and the leak term

Requires build_model.py and Pipeline_Failure.xdsl in the
same directory. Run after build_model.py.

Usage:  python3 robustness_analyses.py
"""
import copy
import itertools
import json
import numpy as np
import build_model as B

nodes = None


def fresh():
    """Load structure, apply elicited medians, calibrate. Returns (nodes, links)."""
    tree, nd = B.load(B.SRC)
    links = {c: {p: dict(t) for p, t in s["links"].items()}
             for c, s in B.PARAMS.items()}
    B.apply_elicited(links)
    B.calibrate(nd, links)
    return nd, links


# ---------------------------------------------------------------- 1. dependence
# Positive association imposed on plausibly-coupled root pairs, marginals
# preserved exactly by iterative proportional fitting on a log-linear seed.
PAIRS = [
    ("Pipeline_Age_", "Cathodic_Protection__"),
    ("Illegal_Connection_", "Burial_Depth_"),
    ("Soil_Corrosivity_", "Water_logging_"),
]


def joint(px, py, theta):
    """3x3 joint with margins px, py and log-linear association theta.
    States are ordered adverse -> favourable in both, so theta > 0 makes
    adverse-with-adverse and favourable-with-favourable more likely."""
    k = len(px)
    z = (np.arange(k) - (k - 1) / 2.0)
    M = np.exp(theta * np.outer(z, z))
    M = M * np.outer(px, py)
    for _ in range(200):                      # IPF onto the stated margins
        M *= (np.asarray(px) / M.sum(1))[:, None]
        M *= (np.asarray(py) / M.sum(0))[None, :]
    return M


def rank_score_correlation(M):
    """Pearson correlation of the ordinal state scores 0/1/2 under the joint M.
    Reported in the paper as a rank correlation (about 0.5 for the strongest
    case between Pipeline Age and Cathodic Protection); it is not Spearman's
    rho with mid-ranks for ties."""
    k = M.shape[0]
    r = np.arange(k)
    mx, my = M.sum(1), M.sum(0)
    ex, ey = (r * mx).sum(), (r * my).sum()
    cov = sum(M[i, j] * (i - ex) * (j - ey) for i in range(k) for j in range(k))
    sx = np.sqrt(((r - ex) ** 2 * mx).sum())
    sy = np.sqrt(((r - ey) ** 2 * my).sum())
    return cov / (sx * sy)


def correlated_grid(nd, theta):
    """Monkeypatch B._grid so root weights carry the pairwise association."""
    orig = B._grid

    def patched(nodes_, ev):
        A, w = orig(nodes_, ev)
        w = w.copy()
        for a, b in PAIRS:
            if a in ev or b in ev:
                continue                       # intervened: leave as set
            px, py = nodes_[a]["probs"], nodes_[b]["probs"]
            M = joint(px, py, theta)
            ratio = M / np.outer(px, py)
            w = w * ratio[A[a], A[b]]
        return A, w / w.sum()

    B._GRID_CACHE.clear()
    B._grid = patched
    return orig


def restore(orig):
    B._grid = orig
    B._GRID_CACHE.clear()


# ------------------------------------------------------------------ 2. mapping
ORIG_SCALE = [0.05, 0.15, 0.35, 0.60, 0.85]
ALT_SCALE = [0.10, 0.25, 0.50, 0.75, 0.90]


def to_index(v, scale):
    """Invert a (possibly half-way) scale value to a fractional index."""
    for i, s in enumerate(scale):
        if abs(v - s) < 1e-9:
            return float(i)
    for i in range(len(scale) - 1):
        if abs(v - (scale[i] + scale[i + 1]) / 2) < 1e-9:
            return i + 0.5
    raise ValueError(f"{v} not on scale")


def from_index(ix, scale):
    lo, hi = int(np.floor(ix)), int(np.ceil(ix))
    return scale[lo] if lo == hi else (scale[lo] + scale[hi]) / 2


def remap_elicited(scale):
    out = {}
    for k, rec in B.ELICITED.items():
        out[k] = {f: from_index(to_index(rec[f], ORIG_SCALE), scale)
                  for f in ("median", "lo", "hi")}
        out[k]["n"] = rec["n"]
    return out


# --------------------------------------------------------------- 4. pathway mix
def pathway_shares(nd, links):
    """Share of terminal-node probability attributable to each hazard pathway.

    For each root configuration the five pathways activate independently with
    probability a_i = c_i * P(hazard_i); the failure mass is apportioned
    across whichever activated, in proportion to a_i.
    """
    A, w = B._grid(nd, {})
    pl = links["Pipeline_Failure"]
    lam = B.PARAMS["Pipeline_Failure"]["leak"]

    def p(ch, fixed=None):
        fixed = fixed or {}
        q = np.full(w.size, 1.0 - B.PARAMS[ch]["leak"])
        for par, tab in links[ch].items():
            if par in fixed:
                q = q * (1.0 - tab["Yes" if fixed[par] == 0 else "No"])
            else:
                lut = np.array([tab[s] for s in nd[par]["states"]])
                q = q * (1.0 - lut[A[par]])
        return 1.0 - q

    ec, ofd, unm, of = (p("External_Corrosion_"), p("Outside_Force_Damage"),
                        p("Unauthorized_Modification_"), p("Operational_Failure"))
    acc = {h: np.zeros(w.size) for h in B.HAZARDS}
    acc["leak"] = np.zeros(w.size)
    tot = np.zeros(w.size)
    for eci, ecp in ((0, ec), (1, 1 - ec)):
        m = p("Material_Degradation_", {"External_Corrosion_": eci})
        for ui, up in ((0, unm), (1, 1 - unm)):
            a = {
                "External_Corrosion_": pl["External_Corrosion_"]["Yes"] * (eci == 0),
                "Unauthorized_Modification_": pl["Unauthorized_Modification_"]["Yes"] * (ui == 0),
                "Material_Degradation_": pl["Material_Degradation_"]["Yes"] * m,
                "Outside_Force_Damage": pl["Outside_Force_Damage"]["Yes"] * ofd,
                "Operational_Failure": pl["Operational_Failure"]["Yes"] * of,
                "leak": np.full(w.size, lam),
            }
            q = np.full(w.size, 1.0)
            for v in a.values():
                q = q * (1.0 - v)
            pf = 1.0 - q
            s = sum(np.broadcast_to(v, w.shape) for v in a.values())
            cw = ecp * up
            for k, v in a.items():
                acc[k] += cw * pf * np.broadcast_to(v, w.shape) / s
            tot += cw * pf
    T = float(np.dot(w, tot))
    return {k: float(np.dot(w, v)) / T for k, v in acc.items()}, T


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    nd, links = fresh()
    base = B.marginals(nd, links)
    print(f"reproduced base PF = {100*base['Pipeline_Failure']:.2f}%\n")

    print("=" * 72)
    print("1. ROOT-NODE DEPENDENCE")
    print("=" * 72)
    px = nd["Pipeline_Age_"]["probs"]
    py = nd["Cathodic_Protection__"]["probs"]
    print(f"{'theta':>6}{'rho(Age,CP)':>13}{'EC %':>9}{'UM %':>9}{'PF %':>9}{'d(PF) pp':>10}")
    for th in (0.0, 0.4, 0.8, 1.2):
        orig = correlated_grid(nd, th)
        m = B.marginals(nd, links)
        rho = rank_score_correlation(joint(px, py, th))
        print(f"{th:>6.1f}{rho:>13.2f}{100*m['External_Corrosion_']:>9.2f}"
              f"{100*m['Unauthorized_Modification_']:>9.2f}"
              f"{100*m['Pipeline_Failure']:>9.2f}"
              f"{100*(m['Pipeline_Failure']-base['Pipeline_Failure']):>10.2f}")
        restore(orig)

    print("\n" + "=" * 72)
    print("2. VERBAL-NUMERIC MAPPING")
    print("=" * 72)
    saved = B.ELICITED
    print(f"{'mapping':<34}{'PF %':>9}{'d(PF) pp':>10}")
    print(f"{'0.05/0.15/0.35/0.60/0.85 (paper)':<34}"
          f"{100*base['Pipeline_Failure']:>9.2f}{0.0:>10.2f}")
    for name, sc in [("0.10/0.25/0.50/0.75/0.90", ALT_SCALE),
                     ("0.10/0.30/0.50/0.70/0.90 (linear)",
                      [0.10, 0.30, 0.50, 0.70, 0.90])]:
        B.ELICITED = remap_elicited(sc)
        nd2, l2 = fresh()
        m = B.marginals(nd2, l2)
        print(f"{name:<34}{100*m['Pipeline_Failure']:>9.2f}"
              f"{100*(m['Pipeline_Failure']-base['Pipeline_Failure']):>10.2f}")
        B.ELICITED = saved

    print("\n" + "=" * 72)
    print("3. MANAGERIAL BUNDLE VARIANTS")
    print("=" * 72)
    b0 = base["Pipeline_Failure"]
    variants = [
        ("Managerial bundle as published, Table 5 Sc. 8 (IC + TPI + maint.)",
         {"Illegal_Connection_": "Low", "Third_Party_Interference_": "Low",
          "Maintenance_Frequency_": "Regular"}),
        ("Enforcement + coordination only (IC + TPI)",
         {"Illegal_Connection_": "Low", "Third_Party_Interference_": "Low"}),
        ("Burial-depth variant, not reported (IC + TPI + burial depth)",
         {"Illegal_Connection_": "Low", "Third_Party_Interference_": "Low",
          "Burial_Depth_": "Deep"}),
        ("Technical bundle (for comparison)",
         {"Cathodic_Protection__": "Present", "Maintenance_Frequency_": "Regular",
          "Pipeline_Age_": "New", "Material_Quality_": "High"}),
    ]
    for lab, ev in variants:
        m = B.marginals(nd, links, ev)
        print(f"{lab:<56}{100*m['Pipeline_Failure']:>7.2f}%"
              f"{100*(b0-m['Pipeline_Failure']):>8.1f} pp")

    print("\n" + "=" * 72)
    print("4. PATHWAY DECOMPOSITION")
    print("=" * 72)
    sh, T = pathway_shares(nd, links)
    egig = {"External_Corrosion_": 25.7, "Outside_Force_Damage": 22.8,
            "Material_Degradation_": 17.5}  # EGIG 2013-2022; construction defect/material failure
    print(f"{'pathway':<30}{'model share %':>15}{'EGIG cause share %':>21}")
    for k in B.HAZARDS + ["leak"]:
        lab = B.LABEL.get(k, "Leak (outside model)")
        e = egig.get(k)
        print(f"{lab:<30}{100*sh[k]:>15.1f}{(f'{e:.1f}' if e else '-'):>21}")
    print(f"{'sum':<30}{100*sum(sh.values()):>15.1f}")
