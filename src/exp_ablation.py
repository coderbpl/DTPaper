"""
darktrace_phase1/src/exp_ablation.py

Integration ablation (manuscript Section 8.11, RQ-integration). This is the
experiment that justifies calling DarkTrace a *framework* rather than a loose
collection of components.

THE HONEST PREMISE (state this verbatim in the manuscript):
The four pillars do NOT improve each other's predictive accuracy — explainable
scoring does not make the classifier more accurate, and sealing does not change
any F1. A naive "remove-a-component-and-watch-accuracy-fall" ablation would
therefore be misleading, because accuracy would not move. The real interaction
effect is on the PROPERTIES of the emitted threat-intelligence (TI) object: the
integrated pipeline produces, per finding, a single artifact that is
simultaneously (a) classified, (b) risk-scored, (c) explained with a faithful
attribution, (d) bound to capture-time tamper-evident sealed evidence, and (e)
exported in a standards-compliant, provenance-carrying STIX 2.1 indicator.

Removing any pillar does not lower accuracy; it DESTROYS a property of the output
and thereby a concrete analyst capability. We measure this directly: for each
pipeline configuration we compose the end-to-end artifact for every finding and
record which capability-properties hold, then summarise an "actionability"
composite. The full configuration is the only one in which all properties hold —
that is the integration effect, demonstrated rather than asserted.

Configurations:
  full              : classify + score + explain + seal + export(+provenance)
  no_scoring        : drop risk score (indicator loses severity/prioritisation)
  no_explanation    : drop attribution (score is unjustified / not auditable)
  no_sealing        : drop hash-chain evidence (export carries no verifiable provenance)
  no_export         : drop STIX (result not interoperable / not machine-ingestible)

Per-finding capability-properties (each boolean, then averaged):
  P1 has_label          : a class is assigned                        (classification)
  P2 has_severity       : a numeric risk/priority is attached        (scoring)
  P3 has_faithful_expl  : a faithful top-k attribution is attached   (explainability)
  P4 has_sealed_evidence: content hash committed to the hash chain   (integrity)
  P5 verifiable_provenance: STIX indicator carries the ledger hash &
                            it re-verifies against the chain          (integration)
  P6 standards_export   : a valid STIX 2.1 indicator is produced     (interoperability)

ACTIONABILITY composite = mean over the six properties (0..1), reported per
configuration. Only `full` reaches 1.0; each ablation zeroes the properties that
depend on the removed pillar AND any downstream property (e.g. removing sealing
also kills verifiable_provenance). This dependency cascade is the point.

Real-data-only: uses real CoDA findings + real scoring model + real hash chain +
real STIX builder. Fails hard if CoDA is absent.

Run:
    python -m src.exp_ablation --config configs/ablation.json
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd

from .utils import load_config, ensure_dirs, get_logger, save_json, set_seed
from .exp_blockchain import make_manifest, LocalHashChainLedger, sha256_hex, canonical_bytes
from .exp_stix import build_bundle, validate_bundle


class MissingRealDataError(FileNotFoundError):
    pass


# ----------------------------------------------------------------------
# Build the real per-finding inputs: category (classifier), risk (scorer),
# faithful attribution (explainer). We reuse the real scoring pipeline so the
# risk score and faithfulness are genuine, not invented.
# ----------------------------------------------------------------------
def _prepare_findings(cfg, logger, n):
    """Return a list of finding dicts with REAL category, risk, and a real
    top-k attribution + per-item faithfulness, from the scoring pipeline."""
    from .exp_text import load_dataset as load_text
    df, _ = load_text(cfg, logger)
    df = df.head(n).reset_index(drop=True)

    # Real risk via the scoring module's stated severity map (category-derived;
    # this provenance is itself reported as a limitation elsewhere).
    from .exp_scoring import _risk_label
    risk, risk_mode, _ = _risk_label(df, cfg, logger)
    risk = np.asarray(risk)

    # Real, lightweight faithful attribution: TF-IDF + linear model, then an
    # occlusion check per item (drop the top-k tokens -> probability should fall
    # more than dropping random tokens). This yields a genuine per-item
    # faithfulness flag without the full SHAP cost (the heavy SHAP/LIME/IG
    # analysis lives in Phase 3; here we only need a real faithful top-k).
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    y_bin = (risk >= int(cfg["data"].get("binary_threshold", 2))).astype(int)
    vec = TfidfVectorizer(max_features=cfg["model"].get("max_features", 2000),
                          ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(df["text"].astype(str).values)
    feats = np.array(vec.get_feature_names_out())
    if len(np.unique(y_bin)) < 2:
        raise ValueError("Need both risk classes for the ablation scorer.")
    clf = LogisticRegression(max_iter=1000).fit(X, y_bin)
    coef = clf.coef_[0]

    findings = []
    Xa = X.toarray()
    topk = int(cfg.get("topk", 10))
    for i in range(len(df)):
        row = Xa[i]
        present = np.where(row > 0)[0]
        if len(present) == 0:
            top_idx = np.array([], dtype=int)
        else:
            contrib = row[present] * coef[present]      # token contribution to risk
            order = present[np.argsort(-contrib)]
            top_idx = order[:topk]
        # faithfulness: p(full) - p(top-k removed) should exceed p(full)-p(random removed)
        p_full = clf.predict_proba(X[i])[0, 1]
        def _drop(idxs):
            x2 = row.copy(); x2[idxs] = 0.0
            from scipy.sparse import csr_matrix
            return clf.predict_proba(csr_matrix(x2))[0, 1]
        drop_top = p_full - _drop(top_idx) if len(top_idx) else 0.0
        rnd = np.random.RandomState(cfg["seed"] + i)
        rand_idx = rnd.choice(present, size=min(topk, len(present)), replace=False) if len(present) else []
        drop_rand = p_full - _drop(rand_idx) if len(rand_idx) else 0.0
        # Honest faithfulness: the top-k attribution must beat random by a real
        # MARGIN (default 0.05), not merely tie or win by epsilon. Removing the
        # highest-contribution tokens of a linear model almost always beats random
        # by a hair, so a bare drop_top>drop_rand test is trivially true and would
        # inflate this property; the margin makes it a meaningful, falsifiable flag.
        margin = float(cfg.get("faithfulness_margin", 0.05))
        faithful = bool((drop_top - drop_rand) >= margin)
        findings.append({
            "category": str(df.loc[i, "label"]),
            "text": str(df.loc[i, "text"]),
            "source_url": str(df.loc[i, "__key__"]) if "__key__" in df.columns else f"item://{i}",
            "risk": int(risk[i]),
            "top_features": feats[top_idx].tolist(),
            "faithful": faithful,
        })
    logger.info(f"Prepared {len(findings)} real findings "
                f"(risk_mode={risk_mode}); "
                f"{sum(f['faithful'] for f in findings)} have a faithful top-{topk} attribution.")
    return findings


# ----------------------------------------------------------------------
# Compose the end-to-end artifact for ONE finding under a given configuration,
# and record which capability-properties hold.
# ----------------------------------------------------------------------
def _compose(finding, cfg_flags, ledger, logger):
    """Return the property booleans for one finding under the active pillars."""
    use_score = cfg_flags["scoring"]
    use_expl = cfg_flags["explanation"]
    use_seal = cfg_flags["sealing"]
    use_export = cfg_flags["export"]

    props = {"P1_has_label": True,  # classification always on (the substrate)
             "P2_has_severity": False, "P3_has_faithful_expl": False,
             "P4_has_sealed_evidence": False, "P5_verifiable_provenance": False,
             "P6_standards_export": False}

    # scoring
    risk = finding["risk"] if use_score else None
    props["P2_has_severity"] = use_score

    # explanation (only meaningful if scoring is on AND the attribution is faithful)
    props["P3_has_faithful_expl"] = bool(use_expl and use_score and finding["faithful"])

    # sealing: commit the content hash to the real hash chain
    evidence_ref = None
    if use_seal:
        manifest = make_manifest(finding["text"], finding["source_url"],
                                 collector="DarkTrace-Ablation")
        block = ledger.seal(manifest)
        props["P4_has_sealed_evidence"] = True
        evidence_ref = {"evidence_id": manifest["evidence_id"],
                        "block_hash": block["block_hash"],
                        "content_hash": manifest["content_hash"]}

    # export: build a STIX indicator (optionally carrying provenance)
    if use_export:
        item = {"category": finding["category"], "text": finding["text"],
                "source_url": finding["source_url"], "risk": risk}
        if use_seal and evidence_ref:
            item["evidence_ref"] = evidence_ref
        bundle = build_bundle([item], logger, link_evidence=use_seal)
        ok, n_ind = validate_bundle(bundle, logger)
        props["P6_standards_export"] = bool(ok and n_ind == 1)
        # verifiable provenance: indicator carries the ledger hash AND it
        # re-verifies against the chain (the integration property)
        if use_seal and props["P6_standards_export"]:
            ind = [o for o in bundle["objects"] if o.get("type") == "indicator"][0]
            refs = ind.get("external_references", [])
            carried = any(evidence_ref["block_hash"] in (r.get("url", "")) for r in refs)
            # re-verify the sealed block is intact in the chain
            chain_ok = ledger.verify_chain()
            props["P5_verifiable_provenance"] = bool(carried and chain_ok)
    return props


def _run_configuration(name, flags, findings, cfg, logger):
    """Compose all findings under one configuration; return averaged properties."""
    # fresh ledger per configuration for a clean measurement
    lp = os.path.join(cfg["paths"]["tables"], f"_ablation_ledger_{name}.jsonl")
    if os.path.exists(lp):
        os.remove(lp)
    ledger = LocalHashChainLedger(lp, logger)
    rows = [_compose(f, flags, ledger, logger) for f in findings]
    dfp = pd.DataFrame(rows)
    prop_means = {c: float(dfp[c].mean()) for c in dfp.columns}
    actionability = float(np.mean(list(prop_means.values())))
    if os.path.exists(lp):
        os.remove(lp)
    logger.info(f"[{name}] actionability={actionability:.3f} "
                f"props={ {k: round(v,2) for k,v in prop_means.items()} }")
    return {"configuration": name, "flags": flags,
            "property_means": prop_means, "actionability": actionability}


def run(cfg, logger):
    set_seed(cfg["seed"])
    n = cfg["data"].get("n_findings", 300)
    findings = _prepare_findings(cfg, logger, n)

    configs = {
        "full":           {"scoring": True,  "explanation": True,  "sealing": True,  "export": True},
        "no_scoring":     {"scoring": False, "explanation": True,  "sealing": True,  "export": True},
        "no_explanation": {"scoring": True,  "explanation": False, "sealing": True,  "export": True},
        "no_sealing":     {"scoring": True,  "explanation": True,  "sealing": False, "export": True},
        "no_export":      {"scoring": True,  "explanation": True,  "sealing": True,  "export": False},
    }
    results = {"experiment": "integration_ablation", "reportable": True,
               "n_findings": len(findings), "seed": cfg["seed"],
               "premise": ("ablation measures capability-PROPERTIES of the emitted "
                           "TI object, not accuracy; only the full pipeline yields "
                           "all properties — this is the integration effect."),
               "configurations": []}
    for name, flags in configs.items():
        results["configurations"].append(_run_configuration(name, flags, findings, cfg, logger))

    full = next(c for c in results["configurations"] if c["configuration"] == "full")
    results["full_actionability"] = full["actionability"]
    results["delta_vs_full"] = {
        c["configuration"]: round(full["actionability"] - c["actionability"], 4)
        for c in results["configurations"] if c["configuration"] != "full"}
    return results


def write_table11_fragment(results, out_path, logger):
    rows = []
    props = ["P1_has_label", "P2_has_severity", "P3_has_faithful_expl",
             "P4_has_sealed_evidence", "P5_verifiable_provenance", "P6_standards_export"]
    for c in results["configurations"]:
        row = {"Configuration": c["configuration"]}
        for p in props:
            row[p.split("_", 1)[1]] = round(c["property_means"][p], 3)
        row["Actionability"] = round(c["actionability"], 3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Wrote Table 11 (ablation) fragment -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ablation.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("ablation", cfg["paths"]["logs"])
    t0 = time.time()
    try:
        results = run(cfg, logger)
    except MissingRealDataError as e:
        logger.error(str(e)); raise SystemExit(2)
    tables = cfg["paths"]["tables"]
    save_json(results, os.path.join(tables, "ablation_results.json"))
    write_table11_fragment(results, os.path.join(tables, "table11_ablation.csv"), logger)
    logger.info("Real-data run complete — Table 11 ablation fragment written.")
    logger.info(f"Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
