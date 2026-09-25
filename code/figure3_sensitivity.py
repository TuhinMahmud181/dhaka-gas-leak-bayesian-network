#!/usr/bin/env python3
"""Fig. 3: sensitivity of Pipeline Leak to the link probabilities: leverage
(each parameter varied by +/-25%) against each elicited parameter's own
expert range. Run after build_model.py."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import build_model as B
import robustness_analyses as R

DARK, LIGHT = "#4a4a4a", "#c8d2de"

def main(n=9):
    nodes, links = R.fresh()
    _, rows = B.tornado(nodes, links)
    rows = sorted(rows, key=lambda r: -r[7])[:n][::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(y + 0.19, [100 * r[4] for r in rows], height=0.36, color=LIGHT,
            edgecolor="black", linewidth=0.5, label="uniform $\\pm$25% (leverage)")
    ax.barh(y - 0.19, [100 * r[7] for r in rows], height=0.36, color=DARK,
            edgecolor="black", linewidth=0.5,
            label="own elicited range (contribution to uncertainty)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{B.LABEL[r[1]]}\n\u2192 {B.LABEL[r[0]]}" for r in rows], fontsize=6.8)
    ax.set_xlabel("Swing in P(pipeline leak), percentage points", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", lw=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/Figure_3.pdf")
    print("wrote figures/Figure_3.pdf")

if __name__ == "__main__":
    main()
