#!/usr/bin/env python3
"""Section 3.2 and Table S7: sensitivity of Pipeline Leak to the hazard anchors.

The five mains anchors (Table S7) are the authors' judgements and are held fixed
in the Monte Carlo. This script shows their effect by solving the anchoring
again for each set of anchors (Riser Leak keeps its survey anchor):

  1. all five mains anchors scaled by 0.5, 0.75, 1, 1.25 and 1.5 (Section 3.2);
  2. the five hazards treated as independent at their anchors, which shows that
     the anchors and escalation probabilities set the level of Pipeline Leak
     (Section 2.8).

New leaks per km per year use Eq. (4) with L = 1 km.

Run after build_model.py.   Usage:  python3 anchor_scenarios.py   (about 3 min)
"""
import math
import build_model as B
import robustness_analyses as R

BASE = dict(B.TARGETS)
MAINS = ["External_Corrosion_", "Outside_Force_Damage", "Unauthorized_Modification_",
         "Material_Degradation_", "Operational_Failure"]


def run(targets):
    """Re-anchor with the given targets; return Pipeline Leak and leaks/km/yr."""
    B.TARGETS.clear(); B.TARGETS.update(targets)      # calibrate() reads B.TARGETS
    nodes, links = R.fresh()
    p = B.marginals(nodes, links)["Pipeline_Failure"]
    return p, -math.log(1.0 - p)


if __name__ == "__main__":
    try:
        print("1. all five mains anchors scaled")
        for f in (0.5, 0.75, 1.0, 1.25, 1.5):
            t = dict(BASE)
            t.update({k: min(0.95, BASE[k] * f) for k in MAINS})
            p, m = run(t)
            print(f"   x{f:<5} Pipeline Leak {100*p:5.1f}%   {m:.2f} new leaks per km per year")

    finally:
        B.TARGETS.clear(); B.TARGETS.update(BASE)

    print("2. hazards independent at their anchors")
    nodes, links = R.fresh()                           # base anchors restored above
    esc = {h: links["Pipeline_Failure"][h]["Yes"] for h in MAINS}   # escalation
    q = 1.0 - B.PARAMS["Pipeline_Failure"]["leak"]                   # (not anchored)
    for h in MAINS:
        q *= 1.0 - esc[h] * BASE[h]
    full = B.marginals(nodes, links)["Pipeline_Failure"]
    print(f"   from the anchors alone {100*(1-q):5.1f}%   full network {100*full:5.1f}%")
