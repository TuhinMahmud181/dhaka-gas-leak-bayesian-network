#!/usr/bin/env python3
"""Fig. S1: change in the two outputs when the adverse-state prior of each
background factor is scaled from zero to twice its value. Run after build_model.py."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
import os
import build_model as B, robustness_analyses as R, prior_tornado as T
os.makedirs("figures", exist_ok=True)
nodes, links = R.fresh(); base = B.marginals(nodes, links)
f, cur = T.curves(nodes, links)
b_pl, b_rl = 100*base["Pipeline_Failure"], 100*base["Riser_Failure"]
ordr = sorted(cur, key=lambda r: -(cur[r][0].max()-cur[r][0].min()))
riser = [r for r in cur if cur[r][1].max()-cur[r][1].min() > 0.01]
cmap = plt.get_cmap("tab20")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw={"width_ratios":[1.35,1]})
for i, r in enumerate(ordr):
    a1.plot(f, cur[r][0], lw=1.3, color=cmap(i % 20), label=B.LABEL[r])
a1.plot(1, b_pl, "o", color="black", ms=4, zorder=5)
a1.axvline(1, color="#999", lw=0.7, ls="--")
a1.set_xlabel("Scaling of the adverse-state prior", fontsize=8)
a1.set_ylabel("P(Pipeline Leak), %", fontsize=8)
a1.set_title("(a) Pipeline Leak", fontsize=8.5)

for i, r in enumerate(sorted(riser, key=lambda r: -(cur[r][1].max()-cur[r][1].min()))):
    a2.plot(f, cur[r][1], lw=1.3, color=cmap(ordr.index(r) % 20), label=B.LABEL[r])
a2.plot(1, b_rl, "o", color="black", ms=4, zorder=5)
a2.axvline(1, color="#999", lw=0.7, ls="--")
a2.set_xlabel("Scaling of the adverse-state prior", fontsize=8)
a2.set_ylabel("P(Riser Leak), %", fontsize=8)
a2.set_title("(b) Riser Leak", fontsize=8.5)

for ax in (a1, a2):
    ax.tick_params(labelsize=7)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(lw=.4, color="#eee"); ax.set_axisbelow(True)
h, lb = a1.get_legend_handles_labels()
fig.legend(h, lb, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.17, 1, 1]); fig.savefig("figures/Figure_S1.pdf"); plt.close(fig)
print("wrote figures/Figure_S1.pdf")

