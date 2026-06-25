# Phase 5 — Interoperable Threat-Intelligence Export (STIX 2.1 / TAXII)

Converts DarkTrace findings (classified + risk-scored items, optionally with
Phase-4 sealed-evidence references) into standard **STIX 2.1** bundles and pushes
them to a **TAXII 2.1** server, so the output plugs into SIEM/TIP tools (MISP,
OpenCTI, Anomali, etc.). Manuscript Section 12.

## What runs now

```bash
python -m src.exp_stix --config configs/stix.json
```

Builds a STIX 2.1 bundle from real classified CoDA findings, validates it
structurally, and writes it to `results/stix/darktrace_bundle.json` for manual
SIEM import. Uses the official `stix2` library when installed; otherwise it emits
spec-compliant STIX dicts directly (so it runs without the lib for testing).

## Push to a real TAXII 2.1 server

```bash
pip install taxii2-client stix2
# set taxii.url + taxii.collection_id (+ user/password) in configs/stix.json
python -m src.exp_stix --config configs/stix.json --push
```

The push uses `taxii2client.v21`. If no server is configured, the bundle is saved
to disk instead and the run reports `pushed_to_taxii: false`.

## Mapping

Each finding becomes a STIX **Indicator** with:
- a valid STIX pattern over the source URL,
- a `label` mapped from the DarkTrace category (arms→weapon, hacking→
  malicious-activity, ...),
- a `confidence` derived from the category and (if present) the Phase-3 risk score,
- optional `external_references` carrying the Phase-4 evidence id + ledger block
  hash — i.e. the exported intel item is linked to verifiable sealed evidence.

## Cross-pillar integration (the integration novelty)

With `link_evidence: true` and Phase-4 evidence refs attached to findings, the
STIX indicators carry the ledger block hash. This is the four-pillars-connected
story the manuscript's integration claim needs: a single exported, standards-
compliant intel object that is classified (Phase 1), risk-scored and explained
(Phase 3), and backed by capture-time-sealed, tamper-evident evidence (Phase 4).
Demonstrate this end-to-end path in the ablation/integration section.

## Boundary

This builds, validates, and can push standard bundles, but does not host a TAXII
server or SIEM — that is your infrastructure. Bundle generation and validation are
fully reportable now; the push result depends on your server.

## How to report it

- Report bundle validity, object/indicator counts, and (if pushed) successful
  ingestion into your SIEM/TIP.
- The contribution is *interoperability*: that DarkTrace output is consumable by
  standard tooling without custom adapters. A screenshot of the indicators in
  MISP/OpenCTI is strong evidence for the manuscript.
- If you link Phase-4 evidence, highlight that the exported intel is independently
  verifiable against the evidence ledger — that linkage is the novel part.
