"""
darktrace_phase1/tests/test_smoke.py

Fast smoke test: runs both experiments on the synthetic fallback and checks
that outputs are well-formed. Does NOT validate scientific correctness (that
requires real data); it validates that the pipeline executes end-to-end.

Run:
    python -m pytest tests/ -q      (if pytest available)
    python tests/test_smoke.py      (plain run)
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from src import exp_traffic, exp_text
from src.metrics_stats import mcnemar_test, bootstrap_ci, holm_bonferroni
import numpy as np


def test_stats_core():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    a = np.array([0, 1, 2, 0, 1, 1, 0, 1])   # 1 error
    b = np.array([0, 0, 2, 0, 0, 2, 0, 1])   # 2 errors
    p, stat = mcnemar_test(y, a, b)
    assert 0.0 <= p <= 1.0
    lo, hi = bootstrap_ci(y, a, "macro_f1", n_boot=100, seed=0)
    assert lo <= hi
    hb = holm_bonferroni({"x": 0.001, "y": 0.20, "z": 0.04})
    assert hb["x"][1] is True
    print("test_stats_core OK")


def test_traffic_runs():
    cfg = load_config("configs/traffic.json")
    cfg["bootstrap"] = 50; cfg["cv_folds"] = 3   # speed up
    from src.utils import get_logger
    logger = get_logger("test_traffic", cfg["paths"]["logs"])
    res = exp_traffic.run(cfg, logger)
    assert res["synthetic"] is True
    for name, r in res["models"].items():
        assert 0.0 <= r["test"]["macro_f1"] <= 1.0
        assert r["test_macro_f1_ci95"][0] <= r["test_macro_f1_ci95"][1]
    print("test_traffic_runs OK")


def test_text_runs():
    cfg = load_config("configs/text.json")
    cfg["bootstrap"] = 50; cfg["cv_folds"] = 3
    from src.utils import get_logger
    logger = get_logger("test_text", cfg["paths"]["logs"])
    res = exp_text.run(cfg, logger)
    assert res["synthetic"] is True
    # with 12% label noise the synthetic task must NOT be perfectly separable
    for name, r in res["models"].items():
        assert r["test"]["macro_f1"] < 0.999, "synthetic text too easy; add noise"
    print("test_text_runs OK")


if __name__ == "__main__":
    test_stats_core()
    test_traffic_runs()
    test_text_runs()
    print("\nALL SMOKE TESTS PASSED")
