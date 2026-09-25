#!/usr/bin/env python3
"""Fig. 4: model estimates and Titas field records. Run after build_model.py.
Uses the same 5,000 parameter draws (seed 20260905) as the uncertainty analysis.
The draws are cached in fig4_draws_<key>.npz, where <key> is a hash of every
model input, so a stale cache is never reused after a parameter changes."""
import os, json, hashlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import build_model as B, robustness_analyses as R, leak_comparison as L

SEED = 20260905
_inputs = {"PARAMS": B.PARAMS, "ELICITED": B.ELICITED, "TARGETS": B.TARGETS,
           "N_DRAWS": B.N_DRAWS, "SPREAD": B.SPREAD, "SEED": SEED}
KEY = hashlib.sha1(json.dumps(_inputs, sort_keys=True, default=str).encode()).hexdigest()[:10]
CACHE = f"fig4_draws_{KEY}.npz"      # computed before fresh(), which annotates ELICITED
nodes, links = R.fresh(); base = B.marginals(nodes, links)
if os.path.exists(CACHE):
    z = np.load(CACHE); d, t = z["d"], z["t"]
else:
    rng = np.random.default_rng(SEED)
    draws = [B.marginals(nodes, B.perturb(links, rng)) for _ in range(B.N_DRAWS)]
    d = np.array([L.density(x["Pipeline_Failure"]) for x in draws])
    t = np.array([L.counts(x)[2] for x in draws])
    np.savez(CACHE, d=d, t=t)
print("model leaks per km: 5th-95th", np.percentile(d, 5), np.percentile(d, 95))

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5))
# (a) three different quantities, drawn with different symbols
m0 = L.density(base["Pipeline_Failure"])
a1.vlines(0, np.percentile(d, 5), np.percentile(d, 95), color="#a83232", lw=6, alpha=0.5)
a1.plot(0, m0, "o", color="#a83232", ms=6)
lo, hi = L.SURVEY_LEAKS[0] / L.SURVEY_KM, L.SURVEY_LEAKS[1] / L.SURVEY_KM
# two accounts of one survey (459 primary, 985 secondary): separate markers, not a range
a1.plot(1, lo, "s", color="#4a4a4a", ms=5)
a1.plot(1, hi, "s", mfc="white", mec="#4a4a4a", ms=5)
a1.plot(2, L.SURVEY_METHANE / L.SURVEY_KM, "^", mfc="white", mec="#4a4a4a", ms=7)
a1.set_yscale("log"); a1.set_yticks([0.2, 0.5, 1, 2, 5]); a1.set_yticklabels(["0.2", "0.5", "1", "2", "5"])
a1.set_xlim(-0.5, 2.5); a1.set_xticks(range(3))
a1.set_xticklabels(["Model: new\nleaks per year", "Survey:\nleaks found", "Survey: methane-\npositive locations"], fontsize=7)
a1.set_ylabel("Number per km of main (log scale)", fontsize=8)
a1.set_title("(a) Mains", fontsize=8.5)
# (b) modelled new leaks per year (tau = 1 yr) against scaled field records (not
# independent: they reuse the riser anchor; bar = midpoint of their range) and complaints
tt = t / 1e3
obs = [78380 / 1e3, 150668 / 1e3]; mb, rb = [v / 1e3 for v in L.counts(base)[:2]]
a2.bar(0, mb, color="#8fa4bd", edgecolor="black", lw=0.5, label="mains")
a2.bar(0, rb, bottom=mb, color="#4a4a4a", edgecolor="black", lw=0.5, label="risers")
a2.vlines(0, np.percentile(tt, 5), np.percentile(tt, 95), color="#a83232", lw=1.2)
a2.bar(1, np.mean(obs), color="white", edgecolor="black", lw=0.5, hatch="////")
a2.vlines(1, obs[0], obs[1], color="#a83232", lw=1.2)
a2.bar(2, L.REPORTED["FY 2021-22"] / 1e3, color="#a83232", edgecolor="black", lw=0.5)
a2.set_xticks([0, 1, 2])
a2.set_xticklabels(["Model: new\nleaks per year", "Field records,\nscaled", "Leak complaints,\none year"], fontsize=7)
a2.set_ylabel("Number per year, Titas network (thousands)", fontsize=8)
a2.set_title("(b) Titas network", fontsize=8.5)
a2.legend(fontsize=7, frameon=False, loc="upper right")
for ax in (a1, a2):
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_4.pdf"); print("done")
