#!/usr/bin/env python3
"""Fig. 2: the Bayesian network with prior and base-case probabilities.
Writes figures/Figure_2.dot and renders it with Graphviz (dot) to PDF.
Run after build_model.py. Requires Graphviz."""
import os, shutil, subprocess
import build_model as B, robustness_analyses as R

ROW1 = ["Material_Quality_", "Soil_Corrosivity_", "Water_logging_",
        "Cathodic_Protection__", "Pipeline_Age_", "Maintenance_Frequency_"]
ROW2 = ["Operating_Pressure_", "Diameter_", "Third_Party_Interference_",
        "Burial_Depth_", "Illegal_Connection_", "Riser_Installation_Quality_"]

def main():
    nodes, links = R.fresh()
    base = B.marginals(nodes, links)
    dot = ["digraph BN {",
           "rankdir=TB; splines=spline; nodesep=0.18; ranksep=1.05;",
           'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.08,0.05"];',
           'edge [color="#7a7a7a", arrowsize=0.6, penwidth=0.85];']
    for n in ROW1 + ROW2:
        lab = B.LABEL[n] + "\\n" + "\\n".join(
            f"{s[:9]} {100*p:.0f}%" for s, p in zip(nodes[n]["states"], nodes[n]["probs"]))
        dot.append(f'"{n}" [label="{lab}", fillcolor="#eef2f7", color="#8fa4bd"];')
    for n in B.HAZARDS:
        dot.append(f'"{n}" [label="{B.LABEL[n]}\\nYes {100*base[n]:.1f}%", fillcolor="#fdf0d5", color="#c9a227"];')
    for n, lab in (("Pipeline_Failure", "PIPELINE LEAK"), ("Riser_Failure", "RISER LEAK")):
        dot.append(f'"{n}" [label="{lab}\\nYes {100*base[n]:.1f}%", fillcolor="#f6d5d5", '
                   f'color="#a83232", penwidth=1.8, fontsize=15];')
    for row in (ROW1, ROW2):
        dot.append("{rank=same; " + " ".join(f'"{n}"' for n in row) + "}")
        dot.append(" -> ".join(f'"{n}"' for n in row) + " [style=invis];")
    dot.append(f'"{ROW1[0]}" -> "{ROW2[0]}" [style=invis, weight=100];')
    for n in ROW2:
        dot.append(f'"{n}" -> "External_Corrosion_" [style=invis, weight=50];')
    for ch, d in nodes.items():
        for p in d["parents"]:
            dot.append(f'"{p}" -> "{ch}";')
    dot.append('{rank=same; "Pipeline_Failure" "Riser_Failure"}')
    dot.append('"Pipeline_Failure" -> "Riser_Failure" [style=invis, minlen=3];')
    dot.append("}")
    os.makedirs("figures", exist_ok=True)
    open("figures/Figure_2.dot", "w").write("\n".join(dot))
    if not shutil.which("dot"):
        raise SystemExit("Graphviz 'dot' not found; the .dot file was written.")
    subprocess.run(["dot", "-Tpdf", "figures/Figure_2.dot", "-o", "figures/Figure_2.pdf"], check=True)
    print("wrote figures/Figure_2.pdf")

if __name__ == "__main__":
    main()
