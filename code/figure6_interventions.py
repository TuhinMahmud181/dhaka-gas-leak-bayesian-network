#!/usr/bin/env python3
"""Fig. 6: leaks avoided by each intervention with 5th-95th percentile ranges
(1,000 parameter draws, paired). Run after build_model.py. Takes about 10 minutes."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import build_model as B, robustness_analyses as R, leak_comparison as L
nodes, links = R.fresh()
sc = [(i, lab, ev) for i, lab, ev in B.scenarios(nodes) if i > 1]
base = B.marginals(nodes, links); t0 = L.counts(base)
res = {"base": {str(i): [t0[0]-L.counts(B.marginals(nodes, links, ev))[0], t0[1]-L.counts(B.marginals(nodes, links, ev))[1]] for i, lab, ev in sc},
       "labels": {str(i): lab for i, lab, ev in sc}, "draws": []}
rng = np.random.default_rng(20260905)
import os
N = int(os.environ.get("N_DRAWS_FIG6", 1000))
for k in range(N):
    lk = B.perturb(links, rng)
    tb = L.counts(B.marginals(nodes, lk))[2]
    res["draws"].append({str(i): tb - L.counts(B.marginals(nodes, lk, ev))[2] for i, lab, ev in sc})
    if (k+1) % 100 == 0:
        json.dump(res, open("fig6_unc.json", "w")); print(k+1, flush=True)
json.dump(res, open("fig6_unc.json", "w")); print("done", flush=True)
r = res

# ---- plot
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans"})

order = ["8","7","5","3","6","4","2"]
lab = {"8":"Managerial bundle:\nillegal connections + third-party interference + maintenance",
       "7":"Technical bundle:\ncathodic protection + maintenance + pipe replacement",
       "5":"Maintenance regularized","3":"Illegal connections curbed",
       "6":"Aged pipe replaced","4":"Third-party interference controlled",
       "2":"Cathodic protection restored"}
D = {k: np.array([d[k] for d in r["draws"]]) for k in order}
fig, ax = plt.subplots(figsize=(7.6,3.9))
y = np.arange(len(order))[::-1]
for yi, k in zip(y, order):
    m, rs = [v/1e3 for v in r["base"][k]]
    ax.barh(yi, m, color="#c8d3df", edgecolor="black", lw=0.5, label="Mains" if k=="8" else None)
    ax.barh(yi, rs, left=m, color="#4a4a4a", edgecolor="black", lw=0.5, label="Risers" if k=="8" else None)
    lo, hi = np.percentile(D[k],[5,95])/1e3
    ax.errorbar(m+rs, yi, xerr=[[max(0, m+rs-lo)],[max(0, hi-m-rs)]], fmt="none", ecolor="#a83232", elinewidth=1.1, capsize=2.5)
    ax.text(hi+0.8, yi, f"{m+rs:.1f}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in order], fontsize=7.2)
ax.set_xlabel("New leaks avoided per year, Titas network (thousands; tau = 1 yr)", fontsize=8); ax.tick_params(axis="x", labelsize=7.5)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout(); import os; os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_6.pdf")
for k in order:
    lo,hi=np.percentile(D[k],[5,95]); print(k, round(sum(r["base"][k])), round(lo), round(hi))
print("P(manag>tech) =", np.mean(D["8"]>D["7"]), "n =", len(r["draws"]))
