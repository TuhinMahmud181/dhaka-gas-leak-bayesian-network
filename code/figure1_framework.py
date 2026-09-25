#!/usr/bin/env python3
"""Fig. 1: framework of the study (diagram). Writes figures/Figure_1.pdf."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
ACC = "#a83232"
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)

def box(x, y, w, h, t, fc="#eef2f7", ec="#8fa4bd"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.2)

def arr(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9))

box(1, 33, 20, 10, "Documented network\ncondition (Titas,\npress, literature)")
box(1, 18, 20, 10, "Expert\nquestionnaire\n(7 judgements)")
box(27, 25, 22, 11, "Bayesian network\n12 root factors\n5 mains hazards\nleaky noisy-OR CPTs", fc="#fdf0d5", ec="#c9a227")
box(55, 33, 20, 10, "Pipeline Leak\nP(new leak,\n1 km, 1 year)", fc="#f6d5d5", ec=ACC)
box(55, 18, 20, 10, "Riser Leak\nP(riser leaking)", fc="#f6d5d5", ec=ACC)
box(80, 25, 19, 11, "New leaks per year\nmains + risers\n(Titas network)")
box(27, 3, 22, 10, "Uncertainty, sensitivity,\ndiagnosis, robustness")
box(55, 3, 20, 10, "Comparison with\nTitas field records")
box(80, 3, 19, 10, "Interventions as\nleaks avoided")
arr(21, 38, 27, 32); arr(21, 23, 27, 29)
arr(49, 32, 55, 37); arr(49, 29, 55, 24)
arr(75, 38, 80, 33); arr(75, 23, 80, 28)
arr(38, 25, 38, 13); arr(89.5, 25, 65, 13); arr(89.5, 25, 89.5, 13)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_1.pdf")
print("wrote figures/Figure_1.pdf")
