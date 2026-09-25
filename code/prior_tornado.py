#!/usr/bin/env python3
"""Tornado over the twelve background factors.

The adverse-state probability of each root factor is scaled by 1 +/- DELTA,
the remaining mass being redistributed across the other states in proportion.
Reports the resulting swing in both terminal nodes. Run after build_model.py.
"""
import copy, numpy as np, build_model as B, robustness_analyses as R
DELTA = 0.5
ADV = {"Operating_Pressure_": "High"}

def shifted(nodes, r, f):
    nd = copy.deepcopy(nodes)
    B._GRID_CACHE.clear()
    st = nd[r]["states"]; p = np.array(nd[r]["probs"], float)
    i = st.index(ADV.get(r, st[0]))
    new = min(0.999, p[i] * f)
    rest = p.copy(); rest[i] = 0.0
    rest = rest / rest.sum() * (1.0 - new) if rest.sum() > 0 else rest
    rest[i] = new
    nd[r]["probs"] = list(rest)
    return nd

def solve(nodes, links, r, f):
    nd = shifted(nodes, r, f)
    B._GRID_CACHE.clear()
    out = B.marginals(nd, links)
    B._GRID_CACHE.clear()
    return out

if __name__ == "__main__":
    nodes, links = R.fresh()
    base = B.marginals(nodes, links)
    roots = [n for n, d in nodes.items() if not d["parents"]]
    rows = []
    for r in roots:
        lo = solve(nodes, links, r, 1 - DELTA)
        hi = solve(nodes, links, r, 1 + DELTA)
        rows.append((r, 100*lo["Pipeline_Failure"], 100*hi["Pipeline_Failure"],
                     100*lo["Riser_Failure"], 100*hi["Riser_Failure"]))
    rows.sort(key=lambda t: -abs(t[2]-t[1]))
    print(f"base: Pipeline Leak {100*base['Pipeline_Failure']:.2f}%  "
          f"Riser Leak {100*base['Riser_Failure']:.2f}%   (adverse prior {int(DELTA*100)}%)\n")
    print(f"{'background factor':<28}{'PL low':>8}{'PL high':>9}{'swing':>7}"
          f"{'RL low':>8}{'RL high':>9}{'swing':>7}")
    for r, a, b, c, d in rows:
        print(f"{B.LABEL[r]:<28}{a:>8.2f}{b:>9.2f}{b-a:>7.2f}{c:>8.2f}{d:>9.2f}{d-c:>7.2f}")

def curves(nodes, links, factors=np.linspace(0.0, 2.0, 21)):
    """Terminal probabilities as each factor's adverse-state prior is scaled."""
    roots = [n for n, d in nodes.items() if not d["parents"]]
    out = {}
    for r in roots:
        pl, rl = [], []
        for f in factors:
            m = solve(nodes, links, r, f)
            pl.append(100 * m["Pipeline_Failure"]); rl.append(100 * m["Riser_Failure"])
        out[r] = (np.array(pl), np.array(rl))
    return factors, out
