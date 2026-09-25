#!/usr/bin/env python3
"""Diagnostic (backward) analysis: posterior of each root factor's adverse
state given that a leak is observed on the mains or at a riser.
P(r = s | T = Yes) = P(r = s) P(T = Yes | r = s) / P(T = Yes).
Run after build_model.py.   Usage: python3 diagnostic.py"""
import build_model as B
import robustness_analyses as R

nodes, links = R.fresh()
base = B.marginals(nodes, links)
roots = [n for n, d in nodes.items() if not d["parents"]]
rows = []
for r in roots:
    s = nodes[r]["states"][0]                 # adverse state (first listed)
    if r == "Operating_Pressure_":
        s = "High"
    pr = nodes[r]["probs"][nodes[r]["states"].index(s)]
    m = B.marginals(nodes, links, {r: s})
    post = {t: pr * m[t] / base[t] for t in ("Pipeline_Failure", "Riser_Failure")}
    rows.append((r, s, pr, post["Pipeline_Failure"], post["Riser_Failure"]))
rows.sort(key=lambda x: -(x[3] - x[2]))
print(f"{'factor':<28}{'state':<10}{'prior':>7}{'|mains':>8}{'|riser':>8}")
for r, s, pr, pm, prr in rows:
    print(f"{B.LABEL[r]:<28}{s:<10}{pr:>7.3f}{pm:>8.3f}{prr:>8.3f}")
