"""
run_all.py — DarkTrace: run every phase end-to-end and summarise.

Runs all experiments in dependency order, then figures and the consolidated
tables, printing a timed PASS/FAIL summary at the end. Real-data-only: a phase
whose dataset is missing FAILS clearly (it does not fabricate data). Cross-domain
Hindi/Arabic (Phase 2b) is treated as OPTIONAL — if its corpora are absent it is
SKIPPED, not failed, so the core pipeline still completes.

Usage (from repo root, e.g. /kaggle/working/DTPaper):
    python run_all.py
    python run_all.py --skip exp_multilingual exp_ablation   # skip slow phases
"""
import argparse, subprocess, sys, time, os

# (module, config, required?) — required=False means SKIP (not FAIL) if data absent
PHASES = [
    ("src.exp_traffic",                 "configs/traffic.json",                True),
    ("src.exp_text",                    "configs/text.json",                   True),
    ("src.exp_text_transformer",        "configs/text_transformer.json",       False),
    ("src.exp_multilingual",            "configs/multilingual.json",           True),
    ("src.exp_multilingual_crossdomain","configs/multilingual_crossdomain.json", False),
    ("src.exp_scoring",                 "configs/scoring.json",                True),
    ("src.exp_blockchain",              "configs/blockchain.json",             True),
    ("src.exp_stix",                    "configs/stix.json",                   True),
    ("src.exp_ablation",                "configs/ablation.json",               True),
]

POST = [
    ("src.make_figures", None),
    ("src.make_scoring_figures", None),
    ("src.build_all_tables", None),
    ("src.exp_stats", None),   # statistical validation (Section 8.13): CIs + tests
]


def _run(module, config):
    cmd = [sys.executable, "-m", module]
    if config:
        cmd += ["--config", config]
    t0 = time.time()
    proc = subprocess.run(cmd)
    return proc.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[],
                    help="module short-names to skip, e.g. exp_ablation")
    args = ap.parse_args()
    skip = set(args.skip)

    results = []
    for module, config, required in PHASES:
        short = module.split(".")[-1]
        if short in skip:
            results.append((short, "SKIPPED(user)", 0.0)); 
            print(f"\n=== SKIP {short} (user) ===")
            continue
        print(f"\n{'='*60}\n=== RUN {short} ===\n{'='*60}")
        rc, dt = _run(module, config)
        if rc == 0:
            status = "PASS"
        elif rc == 2 and not required:
            status = "SKIPPED(no data)"
        else:
            status = f"FAIL(rc={rc})"
        results.append((short, status, dt))

    # post-processing (figures + consolidated tables) — best-effort
    for module, config in POST:
        short = module.split(".")[-1]
        print(f"\n=== {short} ===")
        rc, dt = _run(module, config)
        results.append((short, "PASS" if rc == 0 else f"FAIL(rc={rc})", dt))

    print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}")
    for short, status, dt in results:
        print(f"  {short:32} {status:18} {dt:7.1f}s")
    n_fail = sum(1 for _, s, _ in results if s.startswith("FAIL"))
    print(f"\n{'ALL OK' if n_fail == 0 else str(n_fail) + ' FAILED'}")
    print("\nResults in results/tables/ALL_RESULTS.md")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
