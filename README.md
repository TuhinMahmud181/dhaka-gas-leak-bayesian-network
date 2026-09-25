# Gas leak frequency in Dhaka: noisy-OR Bayesian network

Model, code and data for:

> Mahmud, T., Razia, S.S. Estimating gas leak frequency in a data-sparse urban distribution network: a noisy-OR Bayesian network for Dhaka, Bangladesh. *Process Safety and Environmental Protection* (submitted).

## Contents

- `code/Pipeline_Failure.xdsl`: the Bayesian network (GeNIe format). Node IDs keep their original names: `Pipeline_Failure` is **Pipeline Leak** and `Riser_Failure` is **Riser Leak** in the paper.
- `code/*.py`: scripts that reproduce every model result in the paper.
- `data/`: aggregated answers of the seven experts (Tables S2–S4). Individual answers are not shared.

## What the model gives

- **Pipeline Leak.** The probability that a reference length L = 1 km of distribution main develops at least one new leak within one year. Eq. (4) turns it into new leaks per km per year. The escalation items did not state a segment length, so L is a modelling choice and the frequency scales as 1/L.
- **Riser Leak.** The share of service risers leaking at a given time, anchored to the 2017 Titas riser survey. Eq. (5) turns it into new leaks per year by dividing by the mean riser-leak duration τ. τ has not been measured: 1 year is the reference and 0.5–3 years are reported.
- **Hazard anchors.** The five mains anchors (Table S7) are the authors' judgements. They are held fixed in the Monte Carlo; `anchor_scenarios.py` shows their effect.

## Run

Use Python 3.8+ with numpy and matplotlib (`pip install -r requirements.txt`). Figure 2 also needs the Graphviz `dot` program, installed as a system package.

Run everything from `code/`, starting with `build_model.py`. It regenerates every probability table, checks them against the committed model file (and prints the largest difference, currently 0) and writes them back.

| Script | Produces | Time |
|---|---|---|
| `build_model.py` | Tables 2, 3, 5, S2 and S5–S7; uncertainty ranges | about 25–35 min (3 × 5,000 draws) |
| `robustness_analyses.py` | Alternative scales and correlation test (Section 3.2); bundles; pathway shares (Fig. 5) | about 3 min |
| `anchor_scenarios.py` | Sensitivity to the hazard anchors (Sections 2.8 and 3.2) | about 3 min |
| `leak_comparison.py` | Table 4, riser-leak duration range, interventions in leak counts | about 10 min |
| `leak_robustness.py` | Table S10, including the τ = 0.5 and 3 year rows | about 1 min |
| `riser_link_test.py` | Table S11 | about 7 min |
| `diagnostic.py` | Table S8 | about 1 min |
| `value_of_information.py` | Table S9 | about 2 min |
| `figure1_framework.py` … `figure6_interventions.py`, `figureS1_condition.py` | Figures 1–6 and S1 as PDF files in `code/figures/` | up to 15 min |

The figures in the paper were drawn in TikZ/pgfplots from the values these scripts print. The figure scripts produce equivalent matplotlib versions.

## Expected values

| Quantity | Value |
|---|---|
| Pipeline Leak | 45.0%, i.e. 0.60 new leaks per km of main per year for L = 1 km (0.30–1.20 for L = 2–0.5 km) |
| Riser Leak | 6.2% |
| Titas network, new leaks per year | about 82,700 for τ = 1 year (32,900–157,500 for τ = 3–0.5 years) |
| Per annual leak complaint (4,891 in FY 2021–22) | 17 for τ = 1 year (7–32 for τ = 3–0.5 years) |
| Of the total, at risers | about 90% (scaled from the riser survey) |
| Hazard anchors ×0.5 to ×1.5 | 0.29–0.93 new leaks per km per year |

## Contact

- **Tuhin Mahmud**, MSc Student<br>
  Department of Chemical Engineering, Bangladesh University of Engineering and Technology (BUET), Dhaka, Bangladesh<br>
  Email: 1024022108@che.buet.ac.bd

- **Dr. Syeda Sultana Razia**, Professor<br>
  Department of Chemical Engineering, Bangladesh University of Engineering and Technology (BUET), Dhaka, Bangladesh<br>
  Email: syedasrazia@che.buet.ac.bd

