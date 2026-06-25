# Phase 4 — Blockchain-Backed Evidence Integrity (real-data, no external service)

Implements capture-time forensic sealing (manuscript Section 11, Section 8.10):
each collected dark-web item is hashed at acquisition and committed to an
append-only hash-chain ledger, making any later tampering detectable and the
chain of custody cryptographically verifiable. Runs entirely locally — no
external blockchain service required, and no synthetic data.

## What it does

A real append-only hash chain:
`block_hash = H(prev_hash || manifest_hash || index || timestamp)`. Each block
references the previous block's hash, so altering any sealed manifest breaks the
chain from that point; `verify_chain()` recomputes every hash and detects it.
The ledger is persisted as `results/ledger.jsonl`.

It seals REAL collected evidence (the CoDA corpus items), so the throughput,
latency, and tamper-detection results are all measured on real data.

## Run

```bash
python -m src.exp_blockchain --config configs/blockchain.json
```

Reports:
- sealing throughput (items/s) and per-item latency (p50/p95) on real evidence,
- clean-chain verification (must PASS),
- a tamper-detection test: a sealed block is mutated and the chain verification
  must catch it (the integrity guarantee).

Outputs: `results/tables/blockchain_results.json`, `table8_blockchain.csv`.

## How to report it

- Report sealing throughput + latency (operational feasibility) and the
  tamper-detection result (the integrity guarantee).
- Frame the contribution as *capture-time* sealing — sealing at acquisition, not
  post-hoc — which is the manuscript's Section 11 novelty.
- The integrity property is cryptographic (SHA-256 hash chain) and is verified by
  recomputation, independent of any third-party service.
