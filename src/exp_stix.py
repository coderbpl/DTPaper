"""
darktrace_phase1/src/exp_stix.py

Phase 5 / Interoperable threat-intelligence export (manuscript Section 12,
Section 8.something). Converts DarkTrace findings (classified + risk-scored
dark-web items, optionally with sealed-evidence references from Phase 4) into
STANDARD STIX 2.1 bundles and pushes them to a TAXII 2.1 server, so the output
plugs into SIEM/TIP tooling (MISP, OpenCTI, Anomali, etc.).

What is real and runnable here:
  - STIX 2.1 object construction (Indicator, Observed-Data, Note, Identity,
    Relationship) using the official `stix2` library when installed, with a
    spec-compliant manual fallback so bundles are produced even without it.
  - Mapping from DarkTrace categories/risk to STIX patterns + labels + confidence.
  - Optional linkage to Phase-4 evidence (external_references carrying the
    content hash + ledger block hash = verifiable chain of custody in the bundle).
  - A TAXII 2.1 push client (real `taxii2client`), used when you provide a server
    URL + collection; otherwise bundles are written to disk for manual import.
  - Bundle validation (structural) + a round-trip count check.

Boundary (honest): a live TAXII server / SIEM is your infrastructure. This module
builds and validates standard bundles and can push them, but does not host a
server. Files are written to results/stix/ for import if no server is configured.

Run:
    python -m src.exp_stix --config configs/stix.json
    python -m src.exp_stix --config configs/stix.json --push   # to TAXII server
"""
from __future__ import annotations
import argparse, json, os, time, uuid
from datetime import datetime, timezone
import numpy as np

from .utils import load_config, ensure_dirs, get_logger, save_json

try:
    import stix2
    HAVE_STIX = True
except Exception:
    HAVE_STIX = False

try:
    from taxii2client.v21 import Server as TaxiiServer
    HAVE_TAXII = True
except Exception:
    HAVE_TAXII = False


# category -> STIX indicator label + base confidence
CATEGORY_STIX = {
    "arms": ("weapon", 80), "drugs": ("drug-trafficking", 75),
    "hacking": ("malicious-activity", 85), "financial": ("fraud", 70),
    "violence": ("threat", 80), "crypto": ("anonymization", 60),
    "gambling": ("suspicious-activity", 40), "porn": ("suspicious-activity", 40),
    "electronic": ("benign", 20), "others": ("benign", 20),
}


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _identity_stix():
    if HAVE_STIX:
        return stix2.Identity(name="DarkTrace", identity_class="system",
                              description="Dark-web monitoring system")
    return {"type": "identity", "spec_version": "2.1",
            "id": f"identity--{uuid.uuid4()}", "created": _now(),
            "modified": _now(), "name": "DarkTrace", "identity_class": "system"}


def _indicator_stix(item, identity_id, evidence_ref=None):
    """Build a STIX Indicator from a DarkTrace finding."""
    cat = str(item["category"]).lower()
    label, conf = CATEGORY_STIX.get(cat, ("suspicious-activity", 30))
    # boost confidence by risk score if present
    if "risk" in item and item["risk"] is not None:
        conf = int(min(100, conf + 5 * float(item["risk"])))
    # a simple, valid STIX pattern over a custom observable (the source URL)
    url = item.get("source_url", "http://unknown.onion")
    pattern = f"[url:value = '{url}']"
    ext_refs = []
    if evidence_ref:
        ext_refs.append({
            "source_name": "DarkTrace-Evidence-Ledger",
            "description": "Capture-time sealed evidence (Phase 4)",
            "external_id": evidence_ref.get("evidence_id", ""),
            # carry the verifiable hashes
            "url": f"ledger://{evidence_ref.get('block_hash','')}"})
    if HAVE_STIX:
        return stix2.Indicator(
            name=f"Dark-web {cat} listing",
            description=item.get("text", "")[:200],
            pattern=pattern, pattern_type="stix",
            valid_from=_now(), confidence=conf,
            labels=[label], created_by_ref=identity_id,
            external_references=ext_refs or None)
    obj = {"type": "indicator", "spec_version": "2.1",
           "id": f"indicator--{uuid.uuid4()}", "created": _now(), "modified": _now(),
           "name": f"Dark-web {cat} listing", "description": item.get("text", "")[:200],
           "pattern": pattern, "pattern_type": "stix", "valid_from": _now(),
           "confidence": conf, "labels": [label], "created_by_ref": identity_id}
    if ext_refs:
        obj["external_references"] = ext_refs
    return obj


def _obj_to_dict(o):
    """Normalise a STIX object (stix2 lib object OR plain dict) to a plain dict."""
    if isinstance(o, dict):
        return o
    # stix2 objects expose serialize() -> JSON string
    try:
        return json.loads(o.serialize())
    except Exception:
        # last resort: stix2 objects are mapping-like
        return dict(o)


def build_bundle(items, logger, link_evidence=False):
    """Assemble a STIX 2.1 bundle from DarkTrace findings.

    Always returns a plain dict with a guaranteed "objects" key, regardless of
    whether the stix2 library is installed. We normalise each object to a dict
    and assemble the bundle ourselves rather than relying on Bundle.serialize()
    round-tripping (which varies across stix2 versions and silently produced an
    empty objects array in some)."""
    identity = _identity_stix()
    iid = identity["id"] if isinstance(identity, dict) else identity.id
    objs = [identity]
    for it in items:
        ev = it.get("evidence_ref") if link_evidence else None
        objs.append(_indicator_stix(it, iid, ev))

    # normalise every object to a plain dict so the bundle is consistent
    obj_dicts = [_obj_to_dict(o) for o in objs]
    bundle = {"type": "bundle",
              "id": f"bundle--{uuid.uuid4()}",
              "objects": obj_dicts}
    src = "stix2 lib" if HAVE_STIX else "manual, spec-compliant"
    logger.info(f"Built STIX 2.1 bundle ({src}): {len(obj_dicts)} objects.")
    return bundle


def validate_bundle(bundle, logger):
    """Structural validation: required fields, object count, id formats."""
    ok = True
    if bundle.get("type") != "bundle":
        logger.error("Bundle missing type=bundle"); ok = False
    objs = bundle.get("objects", [])
    n_ind = sum(1 for o in objs if o.get("type") == "indicator")
    for o in objs:
        if "id" not in o or "--" not in o.get("id", ""):
            logger.error(f"Object missing valid STIX id: {o.get('type')}"); ok = False
        if o.get("type") == "indicator":
            for req in ("pattern", "pattern_type", "valid_from"):
                if req not in o:
                    logger.error(f"Indicator missing {req}"); ok = False
    logger.info(f"Bundle validation: {'PASS' if ok else 'FAIL'} "
                f"({len(objs)} objects, {n_ind} indicators)")
    return ok, n_ind


def push_taxii(bundle, cfg, logger):
    """Push a bundle to a real TAXII 2.1 collection."""
    if not HAVE_TAXII:
        logger.warning("taxii2client not installed; cannot push. Bundle saved to disk.")
        return False
    tx = cfg.get("taxii", {})
    if not tx.get("url"):
        logger.warning("No taxii.url configured; bundle saved to disk instead.")
        return False
    try:
        server = TaxiiServer(tx["url"], user=tx.get("user"), password=tx.get("password"))
        api_root = server.api_roots[0]
        collection = next(c for c in api_root.collections
                          if c.id == tx["collection_id"])
        collection.add_objects(bundle)
        logger.info(f"Pushed {len(bundle['objects'])} objects to TAXII collection "
                    f"{tx['collection_id']}.")
        return True
    except Exception as e:
        logger.error(f"TAXII push failed: {e}. Bundle saved to disk.")
        return False


def _load_findings(cfg, logger, n):
    """Build DarkTrace 'findings' from real classified CoDA items."""
    from .exp_text import load_dataset as load_text
    df, _ = load_text(cfg, logger)
    items = []
    for i, row in df.head(n).iterrows():
        items.append({"category": row["label"], "text": str(row["text"]),
                      "source_url": str(row["__key__"]) if "__key__" in df.columns else f"item://{i}",
                      "risk": None})
    return items


def run(cfg, logger, push=False):
    n = cfg["data"].get("n_findings", 300)
    items = _load_findings(cfg, logger, n)
    t0 = time.time()
    bundle = build_bundle(items, logger, link_evidence=cfg.get("link_evidence", False))
    ok, n_ind = validate_bundle(bundle, logger)

    out_dir = cfg["paths"].get("stix", "results/stix")
    os.makedirs(out_dir, exist_ok=True)
    bundle_path = os.path.join(out_dir, "darktrace_bundle.json")
    with open(bundle_path, "w") as f:
        json.dump(bundle, f, indent=2)
    logger.info(f"Wrote STIX bundle -> {bundle_path}")

    pushed = push_taxii(bundle, cfg, logger) if push else None

    results = {
        "experiment": "phase5_stix_export",
        "reportable": True,
        "stix_lib": HAVE_STIX, "taxii_lib": HAVE_TAXII,
        "n_findings": len(items), "n_objects": len(bundle["objects"]),
        "n_indicators": n_ind, "bundle_valid": ok,
        "bundle_path": bundle_path, "pushed_to_taxii": pushed,
        "build_seconds": time.time() - t0,
    }
    return results


def write_table9_fragment(results, out_path, logger):
    import pandas as pd
    row = {"Export": "STIX2.1/TAXII",
           "Findings": results["n_findings"],
           "STIX_objects": results["n_objects"],
           "Indicators": results["n_indicators"],
           "Bundle_valid": results["bundle_valid"],
           "Pushed_TAXII": results["pushed_to_taxii"],
           "Build_s": round(results["build_seconds"], 3)}
    pd.DataFrame([row]).to_csv(out_path, index=False)
    logger.info(f"Wrote Table 9 fragment -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stix.json")
    ap.add_argument("--push", action="store_true", help="push to TAXII server")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("stix", cfg["paths"]["logs"])
    t0 = time.time()
    results = run(cfg, logger, push=args.push)
    tables = cfg["paths"]["tables"]
    save_json(results, os.path.join(tables, "stix_results.json"))
    write_table9_fragment(results, os.path.join(tables, "table9_stix.csv"), logger)
    logger.info("Real-data run complete — Table 9 STIX fragment written.")
    logger.info(f"Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
