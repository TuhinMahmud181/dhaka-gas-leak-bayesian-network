#!/usr/bin/env python3
"""Table S11: leaks avoided by the technical and managerial bundles when the two
authored riser links (Illegal Connection -> Riser Leak, Pipeline Age -> Riser Leak)
are scaled. The riser node is recalibrated to its 6.2% anchor in every case.
Run after build_model.py."""
import copy, build_model as B, robustness_analyses as R, leak_comparison as L
ORIG = copy.deepcopy(B.PARAMS)

def run(f_ic, f_age):
    B.PARAMS = copy.deepcopy(ORIG)
    lk = B.PARAMS["Riser_Failure"]["links"]
    for s in lk["Illegal_Connection_"]:
        lk["Illegal_Connection_"][s] *= f_ic
    for s in lk["Pipeline_Age_"]:
        lk["Pipeline_Age_"][s] *= f_age
    R.B = B
    nodes, links = R.fresh()
    base = B.marginals(nodes, links)
    t0 = L.counts(base)[2]
    out = {i: t0 - L.counts(B.marginals(nodes, links, ev))[2]
           for i, lab, ev in B.scenarios(nodes) if i in (7, 8)}
    return base["Riser_Failure"], out

if __name__ == "__main__":
    cases = [(1, 1), (0.8, 1), (0.7, 1), (0.6, 1), (0.5, 1), (0, 1),
             (1, 1.25), (1, 1.5), (1, 2), (1, 0), (2, 1)]
    print(f"{'IC x':>6}{'Age x':>7}{'riser %':>9}{'technical':>11}{'managerial':>12}  larger")
    for fic, fage in cases:
        r, o = run(fic, fage)
        print(f"{fic:>6}{fage:>7}{100*r:>9.1f}{o[7]:>11,.0f}{o[8]:>12,.0f}  "
              f"{'managerial' if o[8] > o[7] else 'technical'}")
