#!/usr/bin/env python3
"""Value of information for the Dhaka gas pipeline BN model.

For each background factor R and each computed node T, this reports the
expected reduction in the Shannon entropy of T from observing R:

    VOI(R, T) = H(T) - sum_s P(R = s) H(T | R = s)

This asks which factors are worth measuring, which is a different question
from which parameters the output is sensitive to (build_model.py) and from
which interventions reduce risk (also build_model.py). Nothing here perturbs
any parameter: the model is used exactly as parameterised.

The computation reproduces the ranking given by GeNIe's diagnostic value, and
its magnitudes to within a constant factor, so the result does not depend on
an interactive session with the software.

Requires build_model.py and Pipeline_Failure.xdsl in the same
directory.

Usage:  python3 value_of_information.py
"""
import numpy as np

import build_model as B


def entropy_bits(p):
    """Shannon entropy in bits of a binary variable with P(yes) = p."""
    q = np.array([p, 1.0 - p])
    q = q[q > 0.0]
    return float(-(q * np.log2(q)).sum())


def voi_matrix(nodes, links):
    """Expected entropy reduction, in bits, for every (root, target) pair."""
    roots = [n for n, d in nodes.items() if not d["parents"]]
    out = {}
    for tgt in B.ORDER:
        h0 = entropy_bits(B.marginals(nodes, links)[tgt])
        out[tgt] = {}
        for r in roots:
            expected = 0.0
            for i, state in enumerate(nodes[r]["states"]):
                p = B.marginals(nodes, links, {r: state})[tgt]
                expected += nodes[r]["probs"][i] * entropy_bits(p)
            # clamp: exact zeros can come out as -1e-17
            out[tgt][r] = max(0.0, h0 - expected)
    return roots, out


def main():
    _, nodes = B.load(B.SRC)
    links = {c: {p: dict(t) for p, t in s["links"].items()}
             for c, s in B.PARAMS.items()}
    B.apply_elicited(links)
    B.calibrate(nodes, links)

    roots, M = voi_matrix(nodes, links)
    term = M["Pipeline_Failure"]
    order = sorted(roots, key=lambda r: -term[r])

    print("Absolute expected entropy reduction on the terminal node (bits)\n")
    print(f"  {'background factor':<28}{'bits':>12}{'relative':>10}")
    top = max(term.values())
    for r in order:
        print(f"  {B.LABEL[r]:<28}{term[r]:>12.9f}{100 * term[r] / top:>9.1f}")

    print("\n\nRelative value of information, each column scaled to its own "
          "maximum = 100\n")
    head = "".join(f"{B.LABEL[t][:11]:>12}" for t in B.ORDER)
    print(f"  {'background factor':<28}{head}")
    for r in order:
        row = "".join(f"{100 * M[t][r] / max(M[t].values()):>12.1f}"
                      for t in B.ORDER)
        print(f"  {B.LABEL[r]:<28}{row}")

    print("\nA zero means the factor is d-separated from that target, or "
          "\n influences it only through a path with no uncertainty to resolve.")


if __name__ == "__main__":
    main()
