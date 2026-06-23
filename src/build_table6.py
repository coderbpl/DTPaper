"""
darktrace_phase1/src/build_table6.py

Assemble the full manuscript Table 6 (classification performance) from the
per-experiment CSV fragments produced by exp_traffic.py and exp_text.py.
Adds the prior-work reference rows (clearly marked as cited, not reproduced here)
so the table mirrors manuscript Section 8.17 layout.

Run after both experiments:
    python -m src.build_table6
"""
from __future__ import annotations
import os
import pandas as pd

TABLES = "results/tables"

# prior-work rows quoted from the manuscript references (NOT measured by us).
# These mirror Table 6's cited baselines; values carry the source marker.
PRIOR_ROWS = [
    {"Model_dataset": "DarknetSec (CIC-Darknet2020) [2] (reported)",
     "Accuracy": 0.9222, "Macro_F1": 0.9210, "AUC": "NA", "FPR": "NA"},
    {"Model_dataset": "Stacking ensemble (CIC-Darknet2020) [4] (reported)",
     "Accuracy": 0.97, "Macro_F1": "NA", "AUC": "NA", "FPR": "NA"},
]


def main():
    frags = []
    for f in ("table6_text.csv", "table6_traffic.csv"):
        p = os.path.join(TABLES, f)
        if os.path.exists(p):
            frags.append(pd.read_csv(p))
    if not frags:
        print("No fragments found. Run exp_text and exp_traffic first.")
        return
    ours = pd.concat(frags, ignore_index=True)
    prior = pd.DataFrame(PRIOR_ROWS)
    full = pd.concat([prior, ours], ignore_index=True)
    out = os.path.join(TABLES, "table6_full.csv")
    full.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(full.to_string(index=False))
    # note on provenance
    print("\nNOTE: rows marked '(reported)' are cited from prior work [2],[4]; "
          "rows marked '(SYNTHETIC)' are pipeline-test outputs and must be "
          "replaced with real-data runs before reporting.")


if __name__ == "__main__":
    main()
