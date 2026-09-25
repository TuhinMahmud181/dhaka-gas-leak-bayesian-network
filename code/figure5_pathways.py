#!/usr/bin/env python3
"""Fig. 5: pathway shares of Pipeline Leak (computed with Eq. 6 by
robustness_analyses.pathway_shares) and EGIG cause shares for 2013-2022
(EGIG, 2023). The model shares largely reflect the hazard anchors, because four
of the five escalation medians are equal (Section 3.4). Run after build_model.py."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import robustness_analyses as R
plt.rcParams.update({"font.family":"DejaVu Sans"})
cats=["Unauthorized\nModification","External\nCorrosion","Outside\nForce Damage","Material\nDegradation","Operational\nFailure","Ground\nmovement"]
_nd, _links = R.fresh()
_sh, _ = R.pathway_shares(_nd, _links)
model=[round(100*_sh[h], 1) for h in ("Unauthorized_Modification_", "External_Corrosion_",
       "Outside_Force_Damage", "Material_Degradation_", "Operational_Failure")] + [None]
print("model pathway shares (%):", model[:5])
egig=[None,25.7,22.8,17.5,None,19.3]
x=np.arange(len(cats)); w=0.38
fig,ax=plt.subplots(figsize=(7.6,3.9))
for i in range(len(cats)):
    if model[i] is not None:
        ax.bar(x[i]-w/2,model[i],w,color="#4a4a4a",edgecolor="black",lw=0.6,label="This model (Dhaka mains)" if i==0 else None)
        ax.text(x[i]-w/2,model[i]+0.4,f"{model[i]:.1f}",ha="center",fontsize=7)
    else:
        ax.text(x[i]-w/2,0.8,"not\nmodelled",ha="center",fontsize=6.5,color="#666")
    if egig[i] is not None:
        ax.bar(x[i]+w/2,egig[i],w,color="#c8d3df",edgecolor="black",lw=0.6,label="EGIG, 2013–2022" if i==1 else None)
        ax.text(x[i]+w/2,egig[i]+0.4,f"{egig[i]:.1f}",ha="center",fontsize=7)
    else:
        ax.text(x[i]+w/2,0.8,"no EGIG\ncategory",ha="center",fontsize=6.5,color="#666")
ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=7.5)
ax.set_ylabel("Share of incidents or leaks (%)",fontsize=8.5); ax.set_ylim(0,31)
ax.tick_params(axis="y",labelsize=7.5)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.yaxis.grid(True,color="#e5e5e5",lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7.5,frameon=False,loc="upper right")
fig.tight_layout(); import os; os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_5.pdf")
