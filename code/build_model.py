#!/usr/bin/env python3
"""
Build the Dhaka gas distribution pipeline Bayesian network.

Framing: this is a STRUCTURED-JUDGEMENT SCREENING MODEL, not a validated
predictive model. No incident dataset underlies the parameters. Of the 54 link
probabilities, 17 take the median answer of seven experts in the
Bangladeshi gas sector (18 parameter items were asked; one is not used; see
ELICITED below and Section 2.7 of the paper); the rest are the authors'
judgements informed by the documented condition of the network. Results are
reported as base cases with 5th-95th percentile simulation ranges.

CPT construction: leaky noisy-OR (noisy-MAX for ordinal parents)

    P(child = Yes | pa) = 1 - (1 - lambda) * PROD_i (1 - c_i(pa_i))

  c_i(s) = link probability: probability that parent i alone, in state s,
           with all other parents at baseline, produces the child event
  lambda = leak: causes outside the model

References: Henrion (1989); Diez (1993); Fenton & Neil (2012);
Khakzad, Khan & Amyotte (2013).

Outputs (paper numbering)
  Pipeline_Failure.xdsl        model for GeNIe; the CPTs are regenerated and
                               written back only after checking them against
                               the committed file (see __main__)
  tex/parameters.tex           link probabilities used (Table 2)
  tex/extreme_condition.tex    extreme-condition test (Table 3, Table S5)
  tex/scenario_analysis.tex    intervention scenarios (Table 5)
  tex/sensitivity.tex          sensitivity of Pipeline Leak (Fig. 3)
  tex/sensitivity_nodes.tex    sensitivity of every node (Table S6)
  tex/results_summary.tex      base-case marginals with ranges (Table 3)
  tex/elicited.tex             elicited parameters, medians and ranges (Table S2)
  figures/*.pdf                working figures (not the submitted figures)

The paper's Figures 1-6 and S1 are produced by figure1_framework.py ...
figure6_interventions.py and figureS1_condition.py.
"""

import xml.etree.ElementTree as ET
import itertools
import os
import shutil
import numpy as np

# The script reads the model, regenerates every child CPT from the
# parameters below, and writes the tables back to the same file.
SRC = "Pipeline_Failure.xdsl"
DST = "Pipeline_Failure.xdsl"
TEXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tex")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RNG = np.random.default_rng(20260905)
N_DRAWS = 5000
SPREAD = 0.40                   # log-uniform factor 1/1.4 to 1.4, uncertainty sweep
TORNADO_SPREAD = 0.25           # +/- 25% uniform spread, tornado leverage column

# ---------------------------------------------------------------------------
# ELICITED PARAMETERS
# ---------------------------------------------------------------------------
PARAMS = {
    "External_Corrosion_": {
        "leak": 0.01,
        "links": {
            "Cathodic_Protection__":   {"Absent": 0.45, "Partial": 0.15, "Present": 0.00},
            "Soil_Corrosivity_":       {"High": 0.35, "Medium": 0.12, "Low": 0.00},
            "Pipeline_Age_":           {"Old": 0.25, "Moderate": 0.08, "New": 0.00},
            "Water_logging_":          {"Frequent": 0.20, "Seasonal": 0.08, "Rare": 0.00},
            "Material_Quality_":       {"Low": 0.15, "Medium": 0.05, "High": 0.00},
            "Maintenance_Frequency_":  {"Rare": 0.12, "Irregular": 0.05, "Regular": 0.00},
        },
    },
    "Outside_Force_Damage": {
        "leak": 0.02,
        "links": {
            "Third_Party_Interference_": {"High": 0.55, "Medium": 0.15, "Low": 0.00},
            "Burial_Depth_":             {"Shallow": 0.30, "Adequate": 0.08, "Deep": 0.00},
            "Maintenance_Frequency_":    {"Rare": 0.10, "Irregular": 0.04, "Regular": 0.00},
        },
    },
    "Unauthorized_Modification_": {
        "leak": 0.01,
        "links": {
            "Illegal_Connection_":    {"High": 0.60, "Medium": 0.18, "Low": 0.00},
            "Burial_Depth_":          {"Shallow": 0.22, "Adequate": 0.06, "Deep": 0.00},
            "Maintenance_Frequency_": {"Rare": 0.15, "Irregular": 0.05, "Regular": 0.00},
        },
    },
    "Material_Degradation_": {
        "leak": 0.01,
        "links": {
            "External_Corrosion_": {"Yes": 0.45, "No": 0.00},
            "Pipeline_Age_":       {"Old": 0.30, "Moderate": 0.10, "New": 0.00},
            "Material_Quality_":   {"Low": 0.25, "Medium": 0.08, "High": 0.00},
            "Operating_Pressure_": {"High": 0.20, "Normal": 0.03, "Low": 0.00},
            "Diameter_":           {"Small": 0.08, "Medium": 0.03, "Large": 0.00},
        },
    },
    "Operational_Failure": {
        # non-monotone in pressure by design: Normal is baseline, High is the
        # mechanical stressor, Low carries a small term for the supply
        # instability and cycling documented for the Dhaka network
        "leak": 0.01,
        "links": {
            "Operating_Pressure_":    {"High": 0.35, "Normal": 0.00, "Low": 0.06},
            "Maintenance_Frequency_": {"Rare": 0.18, "Irregular": 0.06, "Regular": 0.00},
            "Pipeline_Age_":          {"Old": 0.12, "Moderate": 0.04, "New": 0.00},
            "Diameter_":              {"Small": 0.05, "Medium": 0.02, "Large": 0.00},
        },
    },
    "Riser_Failure": {
        # Riser leak: a point-component submodel, separate from the mains.
        # A leaking riser is itself the loss of containment, so it has no
        # escalation step and does not feed the mains node. Its parents are
        # root factors shared with the mains model, so interventions act on
        # both consistently. The Illegal Connection and Pipeline Age links
        # are authored, not elicited, and rest on no direct evidence. They
        # decide the ranking of the intervention bundles; riser_link_test.py
        # (Table S11) tests them.
        "leak": 0.01,
        "links": {
            "Riser_Installation_Quality_": {"Poor": 0.40, "Average": 0.10, "Good": 0.00},
            "Maintenance_Frequency_":      {"Rare": 0.20, "Irregular": 0.07, "Regular": 0.00},
            "Illegal_Connection_":         {"High": 0.30, "Medium": 0.09, "Low": 0.00},
            "Pipeline_Age_":               {"Old": 0.15, "Moderate": 0.05, "New": 0.00},
        },
    },
    "Pipeline_Failure": {
        # PIPELINE LEAK. Terminal node of the mains submodel: probability that a
        # reference length L = 1 km of distribution main develops at least one
        # NEW leak within one year (Section 2.3 of the paper).
        # c_i = P(new leak on that length within one year | hazard i present).
        # The escalation items did not state a segment length, so L is a
        # modelling choice and the leak frequency scales as 1/L.
        "leak": 0.005,
        "links": {
            "Outside_Force_Damage":       {"Yes": 0.45, "No": 0.00},
            "Material_Degradation_":      {"Yes": 0.40, "No": 0.00},
            "External_Corrosion_":        {"Yes": 0.35, "No": 0.00},
            "Unauthorized_Modification_": {"Yes": 0.30, "No": 0.00},
            "Operational_Failure":        {"Yes": 0.25, "No": 0.00},
        },
    },
}

# Base-case anchors (Section 2.8, Table S7). The five mains anchors are the
# authors' judgements of the documented condition of the network (most pipe
# past design life, cathodic protection largely absent, illegal connections
# widespread); the cited reports give no percentages. They are HELD FIXED in the
# Monte Carlo (the exponents are solved once), so the simulation ranges contain
# no anchor uncertainty. anchor_scenarios.py reports their effect (Section 3.2).
# The riser anchor is the 2017 Titas riser survey.
TARGETS = {
    "External_Corrosion_":        0.45,
    "Outside_Force_Damage":       0.35,
    "Unauthorized_Modification_": 0.45,
    "Material_Degradation_":      0.30,
    "Operational_Failure":        0.10,
    # Titas 2017 baseline riser survey: 35,252 of 565,952 risers leaking.
    "Riser_Failure":              35252 / 565952,
    "Pipeline_Failure":           None,   # derived, never tuned
}

# Mains hazard states. Riser_Failure is a separate terminal node (riser leak).
HAZARDS = ["External_Corrosion_", "Outside_Force_Damage", "Unauthorized_Modification_",
           "Material_Degradation_", "Operational_Failure"]
ORDER = HAZARDS + ["Riser_Failure", "Pipeline_Failure"]

LABEL = {
    "External_Corrosion_": "External Corrosion",
    "Outside_Force_Damage": "Outside Force Damage",
    "Unauthorized_Modification_": "Unauthorized Modification",
    "Material_Degradation_": "Material Degradation",
    "Operational_Failure": "Operational Failure",
    "Riser_Failure": "Riser Leak",
    "Pipeline_Failure": "Pipeline Leak",
    "Soil_Corrosivity_": "Soil Corrosivity",
    "Cathodic_Protection__": "Cathodic Protection",
    "Water_logging_": "Water Logging",
    "Maintenance_Frequency_": "Maintenance Frequency",
    "Pipeline_Age_": "Pipeline Age",
    "Material_Quality_": "Material Quality",
    "Diameter_": "Diameter",
    "Third_Party_Interference_": "Third Party Interference",
    "Illegal_Connection_": "Illegal Connection",
    "Operating_Pressure_": "Operating Pressure",
    "Burial_Depth_": "Burial Depth",
    "Riser_Installation_Quality_": "Riser Installation Quality",
}

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EXPERT-ELICITED PARAMETERS
# ---------------------------------------------------------------------------
# 17 of the 54 link probabilities take the median answer of a questionnaire
# completed by seven experts in the Bangladeshi gas sector (Section 2.7).
# The
# median response replaces the author's value for the least-favourable state of
# each elicited parent; intermediate states are rescaled in proportion to the
# author's original ordering. The observed minimum and maximum across
# experts replace the default log-uniform interval (a factor of 1.4 either way)
# in the uncertainty analysis
# for these parameters only.

import json

_HERE = os.path.dirname(os.path.abspath(__file__))
# Median, minimum, maximum and number of usable responses across the seven
# experts (aggregate values only).
ELICITED = {'Pipeline_Failure|External_Corrosion_|Yes': {'median': 0.35, 'lo': 0.15, 'hi': 0.6, 'n': 6},
 'Pipeline_Failure|Outside_Force_Damage|Yes': {'median': 0.35, 'lo': 0.05, 'hi': 0.85, 'n': 7},
 'Pipeline_Failure|Unauthorized_Modification_|Yes': {'median': 0.35,
                                                     'lo': 0.35,
                                                     'hi': 0.85,
                                                     'n': 7},
 'Pipeline_Failure|Material_Degradation_|Yes': {'median': 0.35, 'lo': 0.15, 'hi': 0.6, 'n': 6},
 'Pipeline_Failure|Operational_Failure|Yes': {'median': 0.15, 'lo': 0.15, 'hi': 0.35, 'n': 6},
 'External_Corrosion_|Cathodic_Protection__|Absent': {'median': 0.35,
                                                      'lo': 0.15,
                                                      'hi': 0.85,
                                                      'n': 7},
 'External_Corrosion_|Soil_Corrosivity_|High': {'median': 0.6, 'lo': 0.35, 'hi': 0.85, 'n': 7},
 'External_Corrosion_|Pipeline_Age_|Old': {'median': 0.6, 'lo': 0.15, 'hi': 0.85, 'n': 7},
 'External_Corrosion_|Water_logging_|Frequent': {'median': 0.35, 'lo': 0.15, 'hi': 0.6, 'n': 7},
 'Outside_Force_Damage|Third_Party_Interference_|High': {'median': 0.6,
                                                         'lo': 0.35,
                                                         'hi': 0.85,
                                                         'n': 7},
 'Outside_Force_Damage|Burial_Depth_|Shallow': {'median': 0.6, 'lo': 0.15, 'hi': 0.85, 'n': 7},
 'Unauthorized_Modification_|Illegal_Connection_|High': {'median': 0.6,
                                                         'lo': 0.6,
                                                         'hi': 0.85,
                                                         'n': 7},
 'Unauthorized_Modification_|Burial_Depth_|Shallow': {'median': 0.6,
                                                      'lo': 0.35,
                                                      'hi': 0.85,
                                                      'n': 7},
 'Material_Degradation_|Pipeline_Age_|Old': {'median': 0.6, 'lo': 0.35, 'hi': 0.6, 'n': 7},
 'Material_Degradation_|Material_Quality_|Low': {'median': 0.6, 'lo': 0.35, 'hi': 0.85, 'n': 6},
 'Riser_Failure|Riser_Installation_Quality_|Poor': {'median': 0.6, 'lo': 0.15, 'hi': 0.85, 'n': 5},
 'Riser_Failure|Maintenance_Frequency_|Rare': {'median': 0.35, 'lo': 0.15, 'hi': 0.6, 'n': 5}}


def apply_elicited(links):
    """Overwrite elicited link probabilities with expert medians."""
    applied = []
    for key, rec in ELICITED.items():
        ch, par, state = key.split("|")
        tab = links[ch][par]
        old, new = tab[state], rec["median"]
        if old > 0:
            sc = new / old
            for s2 in tab:
                if tab[s2] > 0:
                    tab[s2] = min(0.95, tab[s2] * sc)
        tab[state] = new
        rec["author"] = old
        applied.append((ch, par, old, new, rec["lo"], rec["hi"], rec["n"]))
    return applied


def elicited_bounds(ch, par):
    for key, rec in ELICITED.items():
        c, p, _ = key.split("|")
        if c == ch and p == par and rec["median"] > 0:
            return rec["lo"] / rec["median"], rec["hi"] / rec["median"]
    return None


def load(path):
    tree = ET.parse(path)
    nodes = {}
    for c in tree.getroot().find("nodes"):
        p = c.find("parents")
        nodes[c.get("id")] = {
            "elem": c,
            "states": [s.get("id") for s in c.findall("state")],
            "parents": p.text.split() if p is not None else [],
            "probs": [float(x) for x in c.find("probabilities").text.split()],
        }
    return tree, nodes


def check(nodes):
    bad = []
    for ch, spec in PARAMS.items():
        if set(nodes[ch]["parents"]) != set(spec["links"]):
            bad.append(f"{ch}: parent mismatch {sorted(nodes[ch]['parents'])} vs "
                       f"{sorted(spec['links'])}")
            continue
        for par, tab in spec["links"].items():
            if set(tab) != set(nodes[par]["states"]):
                bad.append(f"{ch}/{par}: state mismatch")
            elif min(tab.values()) != 0.0:
                bad.append(f"{ch}/{par}: no baseline state with c = 0")
            else:
                vals = [tab[s] for s in nodes[par]["states"]]
                if vals != sorted(vals, reverse=True) and ch != "Operational_Failure":
                    bad.append(f"{ch}/{par}: not monotone across ranked states {vals}")
    return bad


def build_cpt(ch, nodes, links=None, leak=None):
    links = links or PARAMS[ch]["links"]
    leak = PARAMS[ch]["leak"] if leak is None else leak
    out = []
    for combo in itertools.product(*[nodes[p]["states"] for p in nodes[ch]["parents"]]):
        q = 1.0 - leak
        for par, st in zip(nodes[ch]["parents"], combo):
            q *= 1.0 - links[par][st]
        py = float(f"{1.0 - q:.6g}")
        out += [py, round(1.0 - py, 10)]
    return out


# ---------------------------------------------------------------------------
# EXACT INFERENCE
# ---------------------------------------------------------------------------
# EC, OFD, UNM, OF and RF (riser leak) depend only on roots. MD depends on EC.
# PF (mains leak) depends on the five mains hazards. Conditioning on (EC, UNM)
# makes the rest conditionally independent, so closed forms apply throughout.

_GRID_CACHE = {}


def _grid(nodes, ev):
    """Root-configuration grid, cached: identical evidence reuses the arrays."""
    key = tuple(sorted(ev.items()))
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    ROW1 = ["Material_Quality_", "Soil_Corrosivity_", "Water_logging_",
            "Cathodic_Protection__", "Pipeline_Age_", "Maintenance_Frequency_"]
    ROW2 = ["Operating_Pressure_", "Diameter_", "Third_Party_Interference_",
            "Burial_Depth_", "Illegal_Connection_", "Riser_Installation_Quality_"]
    roots = ROW1 + ROW2
    cols = []
    for r in roots:
        st = nodes[r]["states"]
        if r in ev:
            cols.append((r, np.array([st.index(ev[r])]), np.array([1.0])))
        else:
            cols.append((r, np.arange(len(st)), np.array(nodes[r]["probs"])))
    g = np.meshgrid(*[c[1] for c in cols], indexing="ij")
    wg = np.meshgrid(*[c[2] for c in cols], indexing="ij")
    A = {c[0]: gi.ravel().astype(np.int8) for c, gi in zip(cols, g)}
    w = np.ones(g[0].size)
    for x in wg:
        w = w * x.ravel()
    keep = w > 0
    if keep.sum() < w.size:
        A = {k: v[keep] for k, v in A.items()}
        w = w[keep]
    _GRID_CACHE[key] = (A, w)
    return A, w


def marginals(nodes, links, evidence=None):
    ev = evidence or {}
    A, w = _grid(nodes, ev)

    def p(ch, fixed=None):
        fixed = fixed or {}
        q = np.full(w.size, 1.0 - PARAMS[ch]["leak"])
        for par, tab in links[ch].items():
            if par in fixed:
                q = q * (1.0 - tab["Yes" if fixed[par] == 0 else "No"])
            else:
                lut = np.array([tab[s] for s in nodes[par]["states"]])
                q = q * (1.0 - lut[A[par]])
        return 1.0 - q

    ec, ofd, unm, of = p("External_Corrosion_"), p("Outside_Force_Damage"), \
        p("Unauthorized_Modification_"), p("Operational_Failure")
    rf = p("Riser_Failure")                    # roots only: separate submodel
    md, pf = np.zeros(w.size), np.zeros(w.size)
    pl, lam = links["Pipeline_Failure"], PARAMS["Pipeline_Failure"]["leak"]
    for eci, ecp in ((0, ec), (1, 1 - ec)):
        m = p("Material_Degradation_", {"External_Corrosion_": eci})
        md += ecp * m
        for ui, up in ((0, unm), (1, 1 - unm)):
            q = np.full(w.size, 1.0 - lam)
            q *= 1.0 - (pl["External_Corrosion_"]["Yes"] if eci == 0 else 0.0)
            q *= 1.0 - (pl["Unauthorized_Modification_"]["Yes"] if ui == 0 else 0.0)
            q *= 1.0 - pl["Material_Degradation_"]["Yes"] * m
            q *= 1.0 - pl["Outside_Force_Damage"]["Yes"] * ofd
            q *= 1.0 - pl["Operational_Failure"]["Yes"] * of
            pf += ecp * up * (1.0 - q)
    vals = dict(zip(HAZARDS, [ec, ofd, unm, md, of]))
    vals["Riser_Failure"] = rf
    vals["Pipeline_Failure"] = pf
    return {k: float(np.dot(w, v)) for k, v in vals.items()}


def scale(links, ch, a):
    return {p: {s: 1 - (1 - v) ** a for s, v in t.items()} for p, t in links[ch].items()}


def calibrate(nodes, links):
    alphas = {}
    for ch in PARAMS:
        if TARGETS.get(ch) is None:
            continue
        base = {p: dict(t) for p, t in links[ch].items()}
        lo, hi = 1e-3, 60.0
        for _ in range(50):
            mid = (lo + hi) / 2
            links[ch] = {p: {s: 1 - (1 - v) ** mid for s, v in t.items()}
                         for p, t in base.items()}
            if marginals(nodes, links)[ch] < TARGETS[ch]:
                lo = mid
            else:
                hi = mid
        a = (lo + hi) / 2
        alphas[ch] = a
        links[ch] = {p: {s: 1 - (1 - v) ** a for s, v in t.items()}
                     for p, t in base.items()}
    return alphas


BEST = {"Pipeline_Age_": "New", "Cathodic_Protection__": "Present", "Diameter_": "Large",
        "Material_Quality_": "High", "Soil_Corrosivity_": "Low", "Water_logging_": "Rare",
        "Third_Party_Interference_": "Low", "Illegal_Connection_": "Low",
        "Operating_Pressure_": "Normal", "Burial_Depth_": "Deep",
        "Maintenance_Frequency_": "Regular", "Riser_Installation_Quality_": "Good"}
WORST = {"Pipeline_Age_": "Old", "Cathodic_Protection__": "Absent", "Diameter_": "Small",
         "Material_Quality_": "Low", "Soil_Corrosivity_": "High",
         "Water_logging_": "Frequent", "Third_Party_Interference_": "High",
         "Illegal_Connection_": "High", "Operating_Pressure_": "High",
         "Burial_Depth_": "Shallow", "Maintenance_Frequency_": "Rare",
         "Riser_Installation_Quality_": "Poor"}

# Intervention scenarios. Scenario 1 is the base case; 2-6 are single-lever
# interventions; 7-8 are bundles. This answers the question the paper actually
# asks - which levers move the risk - rather than sweeping an arbitrary
# adverse-to-favourable gradient.
#
# These eight rows are Table 5 of the paper, in order; converted to leak
# counts they are the bars of Fig. 6. The managerial bundle is enforcement
# against illegal connections + control of third-party excavation + regular
# maintenance (3 + 4 + 5). Burial depth is not part of it, because it cannot be
# changed on an existing network without relaying pipe; a burial-depth variant
# is kept under VARIANTS for reference only.
SCENARIOS = [
    ("Base case (current network)", {}),
    ("Cathodic protection restored", {"Cathodic_Protection__": "Present"}),
    ("Illegal connections curbed", {"Illegal_Connection_": "Low"}),
    ("Third-party interference controlled", {"Third_Party_Interference_": "Low"}),
    ("Maintenance regularized", {"Maintenance_Frequency_": "Regular"}),
    ("Aged pipe replaced (age and material quality)",
     {"Pipeline_Age_": "New", "Material_Quality_": "High"}),
    ("Technical bundle (2 + 5 + 6)",
     {"Cathodic_Protection__": "Present", "Maintenance_Frequency_": "Regular",
      "Pipeline_Age_": "New", "Material_Quality_": "High"}),
    ("Managerial bundle (3 + 4 + 5)",
     {"Illegal_Connection_": "Low", "Third_Party_Interference_": "Low",
      "Maintenance_Frequency_": "Regular"}),
]

# Bundle variants printed for reference only; they are NOT reported in the paper.
VARIANTS = [
    ("Enforcement and third-party coordination only (3 + 4)",
     {"Illegal_Connection_": "Low", "Third_Party_Interference_": "Low"}),
    ("Managerial bundle with burial depth substituted for maintenance (3 + 4 "
     "+ burial depth)",
     {"Illegal_Connection_": "Low", "Third_Party_Interference_": "Low",
      "Burial_Depth_": "Deep"}),
]


def scenarios(nodes):
    return [(i, lab, ev) for i, (lab, ev) in enumerate(SCENARIOS, start=1)]


# ---------------------------------------------------------------------------
# UNCERTAINTY
# ---------------------------------------------------------------------------

def perturb(links, rng):
    """Elicited parameters vary over the observed expert range; the rest
    over the default interval."""
    out = {}
    for ch, tab in links.items():
        out[ch] = {}
        for par, st in tab.items():
            b = elicited_bounds(ch, par)
            if b:
                f = float(rng.uniform(b[0], b[1]))
            else:
                # log-symmetric about 1.0: median multiplier is exactly 1.
                # (uniform on [log(1-SPREAD), log(1+SPREAD)] would have
                #  median sqrt((1-SPREAD)(1+SPREAD)) < 1, biasing downward.)
                f = float(np.exp(rng.uniform(-np.log(1 + SPREAD),
                                             np.log(1 + SPREAD))))
            out[ch][par] = {s: min(0.95, v * f) for s, v in st.items()}
    return out


def sweep(nodes, links, evidence=None):
    draws = []
    for _ in range(N_DRAWS):
        draws.append(marginals(nodes, perturb(links, RNG), evidence))
    return {k: (np.percentile([d[k] for d in draws], 5),
                np.percentile([d[k] for d in draws], 95)) for k in ORDER}


def tornado(nodes, links):
    """One-at-a-time sensitivity of P(pipeline failure) to each link probability.

    Two spans are reported for every parameter:

      uniform   +/-25% on the parameter, identical for all. Measures LEVERAGE:
                how far the output moves per unit relative change. Comparable
                across parameters, but assumes an uncertainty width that is
                not the actual one.
      own range the range the parameter is actually credible over: the span of
                expert answers where the parameter was elicited, +/-25%
                otherwise. Measures CONTRIBUTION TO UNCERTAINTY.

    The two orderings differ, and the difference is informative: a parameter
    with high leverage but a narrow own range is one the experts agreed on.

    Note that the leaky noisy-OR is linear in each link probability with the
    others held fixed, so each span is symmetric about the base case except
    where a parameter acts through more than one path.
    """
    base = marginals(nodes, links)["Pipeline_Failure"]
    rows = []
    for ch, tab in links.items():
        for par in tab:
            lo_l = {c: {p: dict(t) for p, t in v.items()} for c, v in links.items()}
            hi_l = {c: {p: dict(t) for p, t in v.items()} for c, v in links.items()}
            for s in tab[par]:
                lo_l[ch][par][s] = tab[par][s] * (1 - TORNADO_SPREAD)
                hi_l[ch][par][s] = min(0.95,
                                       tab[par][s] * (1 + TORNADO_SPREAD))
            lo = marginals(nodes, lo_l)["Pipeline_Failure"]
            hi = marginals(nodes, hi_l)["Pipeline_Failure"]

            # span over the parameter's own credible range
            b = elicited_bounds(ch, par)
            flo, fhi = b if b else (1 - TORNADO_SPREAD, 1 + TORNADO_SPREAD)
            own = {}
            for tag, f in (("lo", flo), ("hi", fhi)):
                m = {c: {p: dict(t) for p, t in v.items()}
                     for c, v in links.items()}
                for s in tab[par]:
                    m[ch][par][s] = min(0.95, tab[par][s] * f)
                own[tag] = marginals(nodes, m)["Pipeline_Failure"]
            own_span = abs(own["hi"] - own["lo"])

            rows.append((ch, par, lo, hi, abs(hi - lo),
                         own["lo"], own["hi"], own_span, bool(b)))
    rows.sort(key=lambda r: -r[4])
    return base, rows


# ---------------------------------------------------------------------------
# LATEX OUTPUT
# ---------------------------------------------------------------------------

def pct(x):
    return f"{100 * x:.1f}"


def emit(nodes, links, alphas, base, best, worst, ci_base, ci_best, ci_worst,
         scen, tor_base, tor_rows, var=()):
    os.makedirs(TEXDIR, exist_ok=True)

    with open(f"{TEXDIR}/parameters.tex", "w") as f:
        f.write("% elicited noisy-OR link probabilities\n")
        f.write("\\begin{longtable}{l l c c c c}\n")
        f.write("\\caption{Elicited link probabilities $c_i(s)$ and leak terms "
                "$\\lambda$ for the noisy-OR conditional probability tables. "
                "Values are analyst judgements informed by documented network "
                "condition; all are varied in the uncertainty analysis.}"
                "\\label{tab:linkprobs}\\\\\n\\hline\n")
        f.write("\\textbf{Node} & \\textbf{Parent} & \\textbf{State 1} & "
                "\\textbf{State 2} & \\textbf{State 3} & $\\boldsymbol{\\lambda}$"
                "\\\\ \\hline\n\\endfirsthead\n\\hline\n")
        f.write("\\textbf{Node} & \\textbf{Parent} & \\textbf{State 1} & "
                "\\textbf{State 2} & \\textbf{State 3} & $\\boldsymbol{\\lambda}$"
                "\\\\ \\hline\n\\endhead\n")
        for ch in PARAMS:
            first = True
            for par, tab in links[ch].items():
                cells = [f"{tab[s]:.3f}" for s in nodes[par]["states"]]
                while len(cells) < 3:
                    cells.append("--")
                f.write(f"{LABEL[ch] if first else ''} & {LABEL[par]} & "
                        + " & ".join(cells) + " & "
                        + (f"{PARAMS[ch]['leak']:.3f}" if first else "") + "\\\\\n")
                first = False
            f.write("\\hline\n")
        f.write("\\end{longtable}\n")

    with open(f"{TEXDIR}/results_summary.tex", "w") as f:
        f.write("\\begin{table}[h!]\\centering\n\\caption{Base-case probabilities "
                "with 90\\% intervals from the parameter uncertainty analysis.}"
                "\\label{tab:results}\n\\begin{tabular}{l c c}\n\\hline\n")
        f.write("\\textbf{Node} & \\textbf{Base case (\\%)} & "
                "\\textbf{90\\% interval (\\%)} \\\\ \\hline\n")
        for k in ORDER:
            f.write(f"{LABEL[k]} & {pct(base[k])} & "
                    f"{pct(ci_base[k][0])}--{pct(ci_base[k][1])} \\\\\n")
        f.write("\\hline\n\\end{tabular}\\end{table}\n")

    with open(f"{TEXDIR}/extreme_condition.tex", "w") as f:
        f.write("\\begin{table}[h!]\\centering\n\\caption{Extreme-condition test. "
                "All twelve background factors set to their most favourable and "
                "least favourable states. Intervals are 90\\% ranges from the "
                "parameter uncertainty analysis.}\\label{tab:extreme}\n")
        f.write("\\begin{tabular}{l c c c}\n\\hline\n\\textbf{Node} & "
                "\\textbf{Favourable (\\%)} & \\textbf{Base (\\%)} & "
                "\\textbf{Adverse (\\%)} \\\\ \\hline\n")
        for k in ORDER:
            f.write(f"{LABEL[k]} & {pct(best[k])} [{pct(ci_best[k][0])}--"
                    f"{pct(ci_best[k][1])}] & {pct(base[k])} & {pct(worst[k])} "
                    f"[{pct(ci_worst[k][0])}--{pct(ci_worst[k][1])}] \\\\\n")
        f.write("\\hline\n\\end{tabular}\\end{table}\n")

    with open(f"{TEXDIR}/scenario_analysis.tex", "w") as f:
        f.write("\\begin{table}[h!]\\centering\\small\n\\caption{Intervention "
                "scenarios. Scenario 1 is the current network; 2--6 set a "
                "single background factor to its favourable state; 7--8 "
                "combine them.}"
                "\\label{tab:scenario}\n\\begin{tabular}{c l c c}\\hline\n")
        f.write("\\textbf{Sc.} & \\textbf{Intervention} & "
                "\\textbf{P(failure) (\\%)} & \\textbf{Reduction (pp)} "
                "\\\\ \\hline\n")
        b0 = scen[0][2]["Pipeline_Failure"]
        for k, lab, m in scen:
            d = 100 * (b0 - m["Pipeline_Failure"])
            f.write(f"{k} & {lab} & {pct(m['Pipeline_Failure'])} & "
                    f"{'--' if k == 1 else f'{d:.1f}'} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\par\\smallskip\\footnotesize Reductions are computed from "
                "unrounded values and may differ by 0.1 pp from the difference "
                "of the rounded probabilities shown. Variants not reported "
                "in the paper: ")
        f.write("; ".join(
            f"{lab.lower()}, {pct(m['Pipeline_Failure'])}\\% "
            f"({100 * (b0 - m['Pipeline_Failure']):.1f} pp)"
            for lab, m in var))
        f.write(".\n\\end{table}\n")

    with open(f"{TEXDIR}/sensitivity.tex", "w") as f:
        f.write("\\begin{table}[h!]\\centering\n\\caption{Sensitivity of "
                "$P(\\mathrm{pipeline\\ failure})$ to the link probabilities. "
                "The leverage column varies each parameter by $\\pm 25\\%$; the "
                "own-range column varies it over the span of expert answers "
                "where it was elicited (marked E) and by $\\pm 25\\%$ otherwise. "
                "All other parameters are held fixed. Base case "
                f"{pct(tor_base)}\\%.}}\\label{{tab:sensitivity}}\n")
        f.write("\\begin{tabular}{l l c c c c}\\hline\n\\textbf{Node} & "
                "\\textbf{Parameter} & & \\textbf{Leverage (pp)} & "
                "\\textbf{Own range (pp)} & \\textbf{Rank} \\\\ \\hline\n")
        own_order = sorted(tor_rows, key=lambda r: -r[7])
        own_rank = {(r[0], r[1]): i + 1 for i, r in enumerate(own_order)}
        for ch, par, lo, hi, sw, olo, ohi, osw, el in tor_rows[:15]:
            f.write(f"{LABEL[ch]} & {LABEL[par]} & {'E' if el else ''} & "
                    f"{100 * sw:.2f} & {100 * osw:.2f} & "
                    f"{own_rank[(ch, par)]} \\\\\n")
        f.write("\\hline\n\\end{tabular}\\end{table}\n")


# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# PER-NODE SENSITIVITY + FIGURES
# ---------------------------------------------------------------------------

def tornado_node(nodes, links, target, delta=0.25):
    """Swing in P(target) from +/-delta on each ancestor parameter group."""
    base = marginals(nodes, links)[target]
    rows = []
    for ch, tab in links.items():
        for par in tab:
            lo_l = {c: {p: dict(t) for p, t in v.items()} for c, v in links.items()}
            hi_l = {c: {p: dict(t) for p, t in v.items()} for c, v in links.items()}
            for st in tab[par]:
                lo_l[ch][par][st] = tab[par][st] * (1 - delta)
                hi_l[ch][par][st] = min(0.95, tab[par][st] * (1 + delta))
            lo = marginals(nodes, lo_l)[target]
            hi = marginals(nodes, hi_l)[target]
            if abs(hi - lo) > 1e-6:
                rows.append((ch, par, lo, hi, abs(hi - lo)))
    rows.sort(key=lambda r: -r[4])
    return base, rows


def fig_tornado(ax, base, rows, title, n=8):
    rows = rows[:n][::-1]
    y = np.arange(len(rows))
    for i, (ch, par, lo, hi, sw, *_rest) in enumerate(rows):
        ax.barh(i, (hi - base) * 100, left=base * 100, color="#4a4a4a",
                height=0.62, edgecolor="black", linewidth=0.5)
        ax.barh(i, (lo - base) * 100, left=base * 100, color="#e8e8e8",
                height=0.62, edgecolor="black", linewidth=0.5, hatch="////")
    ax.axvline(base * 100, color="#333333", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABEL[p]}\n({LABEL[c]})" for c, p, *_ in rows], fontsize=6.6)
    ax.set_title(f"{title}   (base {base*100:.1f}%)", fontsize=8.5, pad=4)
    ax.tick_params(axis="x", labelsize=7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)


def make_figures(nodes, links, base, best, worst, ci_base, scen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)

    # 1. six-panel tornado over the hazard nodes
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 10.2))
    for ax, tgt in zip(axes.ravel(), HAZARDS + ["Riser_Failure"]):
        b, r = tornado_node(nodes, links, tgt)
        fig_tornado(ax, b, r, LABEL[tgt], n=6)
    fig.suptitle("Sensitivity of each hazard node to elicited parameters "
                 "($\\pm$25%)", fontsize=10.5)
    fig.supxlabel("Probability (%)", fontsize=9)
    fig.tight_layout(rect=[0, 0.01, 1, 0.975])
    fig.savefig(os.path.join(FIGDIR, "tornado_hazards.pdf"))
    plt.close(fig)

    # 2. tornado for the terminal node
    b, r = tornado_node(nodes, links, "Pipeline_Failure")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig_tornado(ax, b, r, "Pipeline Leak", n=12)
    ax.set_xlabel("P(pipeline leak on 1 km of main), %", fontsize=9)
    ax.set_title("")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#e8e8e8", edgecolor="black", hatch="////",
                             label="parameter reduced 25%"),
                       Patch(facecolor="#4a4a4a", edgecolor="black",
                             label="parameter increased 25%")],
              fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "tornado_pipeline.pdf"))
    plt.close(fig)

    # 3. base case with uncertainty intervals
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ks = ORDER
    xs = np.arange(len(ks))
    lo = [100 * ci_base[k][0] for k in ks]
    hi = [100 * ci_base[k][1] for k in ks]
    mid = [100 * base[k] for k in ks]
    ax.vlines(xs, lo, hi, color="#8fa4bd", lw=7, alpha=0.75)
    ax.plot(xs, mid, "o", color="#a83232", ms=6, zorder=3)
    for x, m in zip(xs, mid):
        ax.text(x, m + 2.2, f"{m:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k].replace(" ", "\n") for k in ks], fontsize=7.5)
    ax.set_ylabel("Probability (%)", fontsize=9)
    ax.set_title("Base case with 5th–95th percentile ranges under parameter perturbation", fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "uncertainty.pdf"))
    plt.close(fig)

    # 4. intervention reductions on the mains, single levers and bundles (Table 5)
    b0 = scen[0][2]["Pipeline_Failure"]
    rows = sorted(((lab, 100 * (b0 - m["Pipeline_Failure"]), k >= 7)
                   for k, lab, m in scen if k > 1), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ys = np.arange(len(rows))
    ax.barh(ys, [r[1] for r in rows],
            color=["#a83232" if r[2] else "#2f6aa8" for r in rows],
            edgecolor="black", lw=0.6, height=0.62)
    for y, r in zip(ys, rows):
        ax.text(r[1] + 0.18, y, f"{r[1]:.1f}", va="center", fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Reduction in P(pipeline leak), percentage points",
                  fontsize=9)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.14)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "interventions.pdf"))
    plt.close(fig)

    # 5. annotated network with base-case posteriors
    # The twelve root nodes are wrapped onto ROOTS_PER_ROW-wide rows. A single
    # row of twelve produces a figure about 3.4:1, which at the ACS maximum
    # width of 7 in leaves the node text near 3 pt, below the 4.5 pt minimum.
    ROOTS_PER_ROW = 6
    dot = ["digraph BN {",
           "rankdir=TB; splines=spline; nodesep=0.18; ranksep=1.05;",
           'node [shape=box, style="rounded,filled", fontname="Helvetica", '
           'fontsize=12, margin="0.08,0.05"];',
           'edge [color="#7a7a7a", arrowsize=0.6, penwidth=0.85];']
    ROW1 = ["Material_Quality_", "Soil_Corrosivity_", "Water_logging_",
            "Cathodic_Protection__", "Pipeline_Age_", "Maintenance_Frequency_"]
    ROW2 = ["Operating_Pressure_", "Diameter_", "Third_Party_Interference_",
            "Burial_Depth_", "Illegal_Connection_", "Riser_Installation_Quality_"]
    roots = ROW1 + ROW2
    for n in roots:
        st = nodes[n]["states"]
        pr = nodes[n]["probs"]
        lab = LABEL[n] + "\\n" + "\\n".join(
            f"{s[:9]} {100*p:.0f}%" for s, p in zip(st, pr))
        dot.append(f'"{n}" [label="{lab}", fillcolor="#eef2f7", color="#8fa4bd"];')
    for n in HAZARDS:
        dot.append(f'"{n}" [label="{LABEL[n]}\\nYes {100*base[n]:.1f}%", '
                   f'fillcolor="#fdf0d5", color="#c9a227"];')
    dot.append(f'"Riser_Failure" [label="RISER LEAK\\nYes '
               f'{100*base["Riser_Failure"]:.1f}%", fillcolor="#f6d5d5", '
               f'color="#a83232", penwidth=1.8, fontsize=15];')
    dot.append(f'"Pipeline_Failure" [label="PIPELINE LEAK\\n'
               f'Yes {100*base["Pipeline_Failure"]:.1f}%", fillcolor="#f6d5d5", '
               f'color="#a83232", penwidth=1.8, fontsize=15];')
    bands = [roots[i:i + ROOTS_PER_ROW]
             for i in range(0, len(roots), ROOTS_PER_ROW)]
    for band in bands:
        dot.append("{rank=same; " + " ".join(f'"{n}"' for n in band) + "}")
    # invisible edges stack the root bands and hold the hazard tier below them.
    # Without the second group External Corrosion, which parents two other
    # hazard nodes, is placed among the roots and the tiering reads poorly.
    for a, b in zip(bands, bands[1:]):
        dot.append(f'"{a[0]}" -> "{b[0]}" [style=invis, weight=100];')
    for n in bands[-1]:
        dot.append(f'"{n}" -> "External_Corrosion_" [style=invis, weight=50];')
    for ch, d in nodes.items():
        for p in d["parents"]:
            dot.append(f'"{p}" -> "{ch}";')
    # fixed left-to-right order of the root rows keeps the riser parents on the
    # right, and the two terminal nodes share the bottom row
    dot.append(" -> ".join(f'"{n}"' for n in ROW1) + " [style=invis];")
    dot.append(" -> ".join(f'"{n}"' for n in ROW2) + " [style=invis];")
    dot.append('{rank=same; "Pipeline_Failure" "Riser_Failure"}')
    dot.append('"Pipeline_Failure" -> "Riser_Failure" [style=invis, minlen=3];')
    dot.append("}")
    open(os.path.join(FIGDIR, "bn_posteriors.dot"), "w").write("\n".join(dot))
    if shutil.which("dot"):
        dot_in = os.path.join(FIGDIR, "bn_posteriors.dot")
        dot_out = os.path.join(FIGDIR, "bn_posteriors.pdf")
        os.system('dot -Tpdf "%s" -o "%s"' % (dot_in, dot_out))
    else:
        print("graphviz 'dot' not found; skipping network diagram")


def emit_elicited_table(links):
    with open(f"{TEXDIR}/elicited.tex", "w") as f:
        f.write("\\begin{table}[p]\\centering\\small\n\\caption{Link probabilities "
                "elicited from experts. The authors' initial value, the "
                "median response, the range across experts, and the number "
                "of usable responses are shown for each. The median is the value "
                "adopted; the range is used as the perturbation interval in the "
                "uncertainty analysis.}\\label{tab:elicited}\n")
        f.write("\\begin{tabular}{l l c c c c}\n\\hline\n")
        f.write("\\textbf{Node} & \\textbf{Parameter} & \\textbf{Authors} & "
                "\\textbf{Median} & \\textbf{Range} & $\\boldsymbol{n}$ "
                "\\\\ \\hline\n")
        for key, rec in ELICITED.items():
            ch, par, _ = key.split("|")
            f.write(f"{LABEL[ch]} & {LABEL[par]} & {rec['author']:.3f} & "
                    f"{rec['median']:.3f} & {rec['lo']:.2f}--{rec['hi']:.2f} & "
                    f"{rec['n']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\\end{table}\n")


def emit_node_sensitivity(nodes, links):
    with open(f"{TEXDIR}/sensitivity_nodes.tex", "w") as f:
        f.write("\\begin{longtable}{l l c c c}\n")
        f.write("\\caption{Sensitivity of each hazard node and of the terminal "
                "node to the elicited parameters. Each parameter group is varied "
                "by $\\pm 25\\%$ with all others held at their assigned values. "
                "Swing is the resulting change in the node probability, in "
                "percentage points.}\\label{tab:sens_nodes}\\\\ \\hline\n")
        f.write("\\textbf{Target} & \\textbf{Parameter} & $\\boldsymbol{-25\\%}$ & "
                "$\\boldsymbol{+25\\%}$ & \\textbf{Swing (pp)} \\\\ \\hline\n"
                "\\endfirsthead\n\\hline\n\\textbf{Target} & \\textbf{Parameter} & "
                "$\\boldsymbol{-25\\%}$ & $\\boldsymbol{+25\\%}$ & "
                "\\textbf{Swing (pp)} \\\\ \\hline\n\\endhead\n")
        for tgt in ORDER:
            b, rows = tornado_node(nodes, links, tgt)
            first = True
            for ch, par, lo, hi, sw in rows[:6]:
                f.write(f"{LABEL[tgt] if first else ''} & {LABEL[par]} "
                        f"({LABEL[ch]}) & {pct(lo)} & {pct(hi)} & "
                        f"{100*sw:.2f} \\\\\n")
                first = False
            f.write("\\hline\n")
        f.write("\\end{longtable}\n")


if __name__ == "__main__":
    os.makedirs(TEXDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    tree, nodes = load(SRC)
    problems = check(nodes)
    if problems:
        for p in problems:
            print("PARAM PROBLEM:", p)
        raise SystemExit(1)

    links = {c: {p: dict(t) for p, t in s["links"].items()} for c, s in PARAMS.items()}
    applied = apply_elicited(links)
    print("EXPERT-ELICITED PARAMETERS (%d)" % len(applied))
    print(f"{'node / parent':<52}{'author':>8}{'median':>8}{'range':>14}{'n':>3}")
    for ch, par, old, new, lo, hi, n in applied:
        print(f"{LABEL[ch] + ' / ' + LABEL[par]:<52}{old:>8.3f}{new:>8.3f}"
              f"{lo:>7.2f}-{hi:<6.2f}{n:>3}")
    print()
    alphas = calibrate(nodes, links)
    print("calibration exponents:",
          {k: round(v, 3) for k, v in alphas.items()})

    base = marginals(nodes, links)
    best = marginals(nodes, links, BEST)
    worst = marginals(nodes, links, WORST)

    print(f"\n{'Node':<28}{'Fav':>8}{'Base':>8}{'Adv':>8}")
    ok = True
    for k in ORDER:
        b, m, w = 100 * best[k], 100 * base[k], 100 * worst[k]
        if not b <= m <= w:
            ok = False
        print(f"{LABEL[k]:<28}{b:>7.2f}%{m:>7.2f}%{w:>7.2f}%")
    print("monotonicity:", "PASS" if ok else "FAIL")

    print("\nuncertainty sweep...")
    ci_base, ci_best, ci_worst = (sweep(nodes, links), sweep(nodes, links, BEST),
                                  sweep(nodes, links, WORST))
    scen = [(k, lab, marginals(nodes, links, ev))
            for k, lab, ev in scenarios(nodes)]
    var = [(lab, marginals(nodes, links, ev)) for lab, ev in VARIANTS]
    base_pf = scen[0][2]["Pipeline_Failure"]
    ok_sc = all(m["Pipeline_Failure"] <= base_pf + 1e-9 for _, _, m in scen)
    print("every intervention reduces P(failure):", "PASS" if ok_sc else "FAIL")

    print("\nintervention scenarios (Table 5):")
    for k, lab, m in scen:
        d = 100 * (base_pf - m["Pipeline_Failure"])
        print(f"  {k}  {lab:<48}{100 * m['Pipeline_Failure']:>7.1f}%"
              f"{'    --' if k == 1 else f'{d:>8.1f} pp'}")
    print("  variants not reported in the paper (reference only):")
    for lab, m in var:
        d = 100 * (base_pf - m["Pipeline_Failure"])
        print(f"     {lab:<69}{100 * m['Pipeline_Failure']:>7.1f}%{d:>8.1f} pp")
    tor_base, tor_rows = tornado(nodes, links)

    # Regenerate every CPT and compare with the committed model file before
    # writing, so that a run never silently changes the published model.
    max_diff = 0.0
    for ch in PARAMS:
        new = build_cpt(ch, nodes, links[ch])
        old = nodes[ch]["probs"]
        if len(new) == len(old):
            max_diff = max(max_diff, max(abs(a - b) for a, b in zip(new, old)))
        else:
            max_diff = float("inf")
        nodes[ch]["elem"].find("probabilities").text = " ".join(
            f"{v:.10g}" for v in new)
    print(f"regenerated CPTs vs committed {SRC}: max |difference| = {max_diff:.2e}"
          + ("  (identical)" if max_diff < 1e-6 else "  (CHANGED - check before committing)"))
    tree.write(DST, encoding="UTF-8", xml_declaration=True)

    emit(nodes, links, alphas, base, best, worst, ci_base, ci_best, ci_worst,
         scen, tor_base, tor_rows, var)
    emit_elicited_table(links)
    emit_node_sensitivity(nodes, links)
    print("building figures...")
    make_figures(nodes, links, base, best, worst, ci_base, scen)
    print(f"\nwrote {DST} and {TEXDIR}/*.tex")
    print(f"PF base {100*base['Pipeline_Failure']:.2f}% "
          f"[{100*ci_base['Pipeline_Failure'][0]:.1f}-"
          f"{100*ci_base['Pipeline_Failure'][1]:.1f}]")
    print("\ntop parameter sensitivities on PF (pp):")
    print(f"  {'child':<26} {'parent':<28} {'leverage':>9} {'own range':>10}")
    for ch, par, lo, hi, sw, olo, ohi, osw, el in tor_rows[:8]:
        print(f"  {LABEL[ch]:<26} {LABEL[par]:<28} {100*sw:>9.2f} "
              f"{100*osw:>10.2f}{' E' if el else ''}")
    print("\n  ranked by own range:")
    for ch, par, lo, hi, sw, olo, ohi, osw, el in sorted(
            tor_rows, key=lambda r: -r[7])[:8]:
        print(f"  {LABEL[ch]:<26} {LABEL[par]:<28} {100*sw:>9.2f} "
              f"{100*osw:>10.2f}{' E' if el else ''}")
