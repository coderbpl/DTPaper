"""
darktrace_phase1/src/build_all_tables.py

Aggregate every phase's REAL result fragment into one manuscript-ready summary.
Run after the phases have produced real (non-smoke) outputs in results/tables/.

Collects:
  Table 6  - Phase 1 traffic + text classification   (table6_traffic.csv, table6_text.csv)
  Table 7  - Phase 2 multilingual per-language        (table7_multilingual.csv)
  Table 8  - Phase 4 blockchain sealing integrity     (table8_blockchain.csv)
  Table 9  - Phase 5 STIX/TAXII export                (table9_stix.csv)
  Table 10 - Phase 3 explainable scoring              (table10_scoring.csv)

Only REAL fragments are included; any *_SMOKETEST.csv is explicitly skipped, so the
summary can never accidentally contain non-reportable numbers. Missing fragments
are listed so you know which phases still need a real run.

Run:
    python -m src.build_all_tables
"""
from __future__ import annotations
import os, glob
import pandas as pd

TABLES = "results/tables"

PHASES = {
    "Table 6 - Classification (Phase 1)": ["table6_traffic.csv", "table6_text.csv"],
    "Table 7 - Multilingual (Phase 2)": ["table7_multilingual.csv"],
    "Table 8 - Blockchain integrity (Phase 4)": ["table8_blockchain.csv"],
    "Table 9 - STIX/TAXII export (Phase 5)": ["table9_stix.csv"],
    "Table 10 - Explainable scoring (Phase 3)": ["table10_scoring.csv"],
}


def main():
    os.makedirs(TABLES, exist_ok=True)
    summary_lines = ["# DarkTrace — Consolidated Real Results\n"]
    present, missing = [], []

    # guard: warn loudly if any smoke-test fragment exists
    smoke = glob.glob(os.path.join(TABLES, "*_SMOKETEST.csv"))
    if smoke:
        summary_lines.append(
            "> WARNING: smoke-test fragments present and EXCLUDED "
            f"({len(smoke)}): {[os.path.basename(s) for s in smoke]}\n")

    for title, frags in PHASES.items():
        rows = []
        for frag in frags:
            path = os.path.join(TABLES, frag)
            if os.path.exists(path):
                try:
                    rows.append(pd.read_csv(path))
                except Exception as e:
                    summary_lines.append(f"## {title}\n(error reading {frag}: {e})\n")
        if rows:
            df = pd.concat(rows, ignore_index=True)
            present.append(title)
            summary_lines.append(f"## {title}\n")
            summary_lines.append(df.to_markdown(index=False))
            summary_lines.append("")
        else:
            missing.append(title)
            summary_lines.append(f"## {title}\n_(no real fragment yet — run this phase on real data)_\n")

    summary_lines.append("\n## Status\n")
    summary_lines.append(f"- Phases with real results: {len(present)}/{len(PHASES)}")
    if missing:
        summary_lines.append(f"- Still needing a real run: {', '.join(missing)}")

    out_md = os.path.join(TABLES, "ALL_RESULTS.md")
    with open(out_md, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Wrote consolidated summary -> {out_md}")
    print(f"Phases with real results: {len(present)}/{len(PHASES)}")
    for t in present:
        print(f"  [done] {t}")
    for t in missing:
        print(f"  [pending] {t}")


if __name__ == "__main__":
    main()
