# Integration Ablation — Justifying "Framework" (not just "pipeline")

This experiment provides the evidence that DarkTrace is an integrated *framework*
rather than a loose bundle of independent tools. Manuscript Section 8.11 / the
integration research question.

## The honest premise (state this in the paper)

The four pillars do **not** improve each other's predictive accuracy. Explainable
scoring does not raise classifier F1; sealing does not change any metric. So a
conventional "remove a component, watch accuracy drop" ablation would be
**misleading** — accuracy would not move, and a reviewer would (correctly) call
the integration cosmetic.

The real interaction effect is on the **capability-properties of the emitted
threat-intelligence (TI) object**. The integrated pipeline produces, per finding,
a single artifact that is simultaneously: classified, risk-scored, explained with
a *faithful* attribution, bound to capture-time tamper-evident sealed evidence,
and exported as a standards-compliant STIX 2.1 indicator that *carries verifiable
provenance*. Removing any pillar does not lower accuracy — it **destroys a
property of the output**, and often downstream properties too.

## What is measured

For each configuration we compose the end-to-end artifact for every real CoDA
finding and record six boolean capability-properties, then average them into an
**actionability** composite (0–1):

| Property | Depends on | Meaning |
|---|---|---|
| has_label | classification | a class is assigned |
| has_severity | scoring | a risk/priority is attached |
| has_faithful_expl | scoring + explanation | a *faithful* top-k attribution is attached (margin-tested, not trivial) |
| has_sealed_evidence | sealing | content hash committed to the hash chain |
| verifiable_provenance | sealing + export | the STIX indicator carries the ledger hash AND it re-verifies against the chain |
| standards_export | export | a valid STIX 2.1 indicator is produced |

Configurations: `full`, `no_scoring`, `no_explanation`, `no_sealing`, `no_export`.

## The result and why it matters

Only `full` yields all properties (actionability ~1.0). Each ablation zeroes the
removed pillar's property **and any downstream property that depends on it** — the
*dependency cascade*:

- `no_scoring` also loses `has_faithful_expl` (you cannot explain a score that
  doesn't exist) → actionability ~0.67.
- `no_sealing` also loses `verifiable_provenance` (no hash to carry) → ~0.67.
- `no_export` also loses `verifiable_provenance` (no STIX object to carry it) → ~0.67.
- `no_explanation` loses only faithfulness → ~0.83.

This cascade is the integration effect, **demonstrated rather than asserted**: the
pillars are interdependent in producing an actionable, verifiable TI object. That
is the defensible basis for the word "framework."

## Honesty safeguards built in

- The faithfulness property uses a **margin** (default 0.05): the top-k attribution
  must beat random by a real margin, not by epsilon. Without this, removing the
  top tokens of a linear model trivially "wins" and inflates the property to ~100%.
  With it, weakly-explained items correctly fail.
- Risk is the **category-derived severity** label (same provenance caveat as
  Phase 3); the ablation measures *property survival*, which does not depend on
  the label being analyst-rated.
- Sealing and provenance are verified against the **real** hash chain
  (re-verification on each item), not asserted.

## Run

```bash
python -m src.exp_ablation --config configs/ablation.json
```

Outputs: `results/tables/ablation_results.json` and `table11_ablation.csv`
(per-configuration property means + actionability + delta-vs-full).

## How to report it

Lead with the table and the cascade. State plainly: *"The ablation does not measure
accuracy — by design, integration does not change accuracy. It measures which
analyst-relevant properties of the emitted threat-intelligence object survive when
a pillar is removed. Only the full pipeline yields a classified, scored, faithfully
explained, tamper-sealed, standards-exported object with verifiable provenance;
removing any pillar removes that property and its dependents. This interdependence
is what distinguishes DarkTrace from a co-located set of tools."*

Do **not** claim the ablation shows accuracy gains — it does not, and saying so
would be the kind of overclaim this experiment exists to avoid.
