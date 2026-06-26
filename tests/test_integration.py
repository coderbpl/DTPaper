"""
darktrace_phase1/tests/test_integration.py

Real-data integration checks. This pipeline is REAL-DATA-ONLY — there is no
synthetic fallback — so these tests verify two things:

  1. Import/During-construction sanity: every module imports and its pure helper
     functions (hashing, STIX bundle building, label normalisation) work on small
     in-test inputs. These run anywhere.

  2. Full experiment runs: executed only when the real datasets are present at the
     configured paths. If a dataset is missing, that phase's test is SKIPPED (not
     failed), because we never fabricate data to make a test pass.

Run:
    python tests/test_integration.py
    DARKTRACE_DATA=1 python tests/test_integration.py   # force-require real data
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_logger


def _has(path):
    return os.path.exists(path)


def test_pure_helpers():
    """Cryptography + STIX building + label-normalisation logic (no dataset)."""
    # blockchain hashing + chain integrity on tiny in-memory input
    from src.exp_blockchain import make_manifest, LocalHashChainLedger
    import tempfile, os as _os
    tmp = tempfile.mktemp(suffix=".jsonl")
    ledger = LocalHashChainLedger(tmp)
    m1 = make_manifest("evidence one", "http://x.onion/1", "tester")
    m2 = make_manifest("evidence two", "http://x.onion/2", "tester")
    ledger.seal(m1); ledger.seal(m2)
    assert ledger.verify_chain() is True
    # tamper -> must be detected
    ledger.blocks[0]["manifest"]["content_hash"] = "deadbeef" * 8
    assert ledger.verify_chain() is False
    _os.remove(tmp)

    # STIX bundle building + validation on tiny in-memory findings
    from src.exp_stix import build_bundle, validate_bundle
    logger = get_logger("test", "results/logs")
    items = [{"category": "Arms", "text": "weapon listing", "source_url": "http://x.onion/a", "risk": 3},
             {"category": "Drugs", "text": "gram listing", "source_url": "http://x.onion/b", "risk": 2}]
    bundle = build_bundle(items, logger, link_evidence=False)
    ok, n_ind = validate_bundle(bundle, logger)
    assert ok is True and n_ind == 2

    # traffic label normalisation (case-variant merge)
    from src.exp_traffic import _normalize_labels
    import numpy as np
    y = np.array(["Audio-Streaming", "AUDIO-STREAMING", "Chat", "chat"], dtype=object)
    out = _normalize_labels(y, logger)
    assert len(set(out)) == 2  # merged to 2 canonical classes
    print("test_pure_helpers OK")


def _run_if_data(name, config, module_run):
    cfg = load_config(config)
    csv = cfg["data"].get("csv_path", "")
    if not _has(csv):
        if os.environ.get("DARKTRACE_DATA"):
            raise AssertionError(f"{name}: real data required but missing at {csv}")
        print(f"test_{name} SKIPPED (no real data at {csv})")
        return
    logger = get_logger(f"test_{name}", cfg["paths"]["logs"])
    res = module_run(cfg, logger)
    assert res.get("reportable") is True
    print(f"test_{name} OK (real data)")


def test_traffic():
    from src import exp_traffic
    _run_if_data("traffic", "configs/traffic.json", exp_traffic.run)


def test_text():
    from src import exp_text
    _run_if_data("text", "configs/text.json", exp_text.run)


def test_scoring():
    from src import exp_scoring
    _run_if_data("scoring", "configs/scoring.json", exp_scoring.run)


def test_multilingual():
    from src import exp_multilingual
    _run_if_data("multilingual", "configs/multilingual.json", exp_multilingual.run)


def test_blockchain():
    from src import exp_blockchain
    _run_if_data("blockchain", "configs/blockchain.json", exp_blockchain.run)


def test_stix():
    from src import exp_stix
    _run_if_data("stix", "configs/stix.json", lambda c, l: exp_stix.run(c, l, push=False))



def test_multilingual_crossdomain():
    from src import exp_multilingual_crossdomain as xd
    cfg = load_config("configs/multilingual_crossdomain.json")
    hi = cfg["data"]["hi"]["csv_path"]; ar = cfg["data"]["ar"]["csv_path"]
    if not (_has(hi) or _has(ar)):
        if os.environ.get("DARKTRACE_DATA"):
            raise AssertionError("crossdomain: real data required but missing")
        print("test_multilingual_crossdomain SKIPPED (no Hindi/Arabic corpus)")
        return
    logger = get_logger("test_xd", cfg["paths"]["logs"])
    res = xd.run(cfg, logger)
    assert res.get("reportable") is True
    assert len(res["per_language"]) >= 1
    print("test_multilingual_crossdomain OK (real data)")


if __name__ == "__main__":
    test_pure_helpers()
    test_traffic()
    test_text()
    test_scoring()
    test_multilingual()
    test_blockchain()
    test_stix()
    test_multilingual_crossdomain()
    print("\nALL INTEGRATION CHECKS DONE")
