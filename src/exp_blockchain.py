"""
darktrace_phase1/src/exp_blockchain.py

Phase 4 / Blockchain-backed forensic evidence integrity (manuscript Section 11,
Section 8.10). Implements CAPTURE-TIME sealing: each piece of collected dark-web
evidence is hashed at the moment of acquisition and the hash is committed to an
append-only ledger, so any later tampering is detectable and the chain of custody
is cryptographically verifiable.

What is real and runnable here:
  - SHA-256 content hashing + canonical evidence manifest (who/what/when/where).
  - An append-only HASH-CHAIN ledger (each block references the previous block's
    hash), so sealing + verification + tamper-detection run with no external
    service. This is a genuine cryptographic integrity structure, not a mock.
  - An evaluation that seals REAL collected evidence (CoDA items), measures sealing
    throughput/latency, verifies the clean chain, and proves that tampering with
    any sealed manifest is detected (the manuscript's evidence-integrity claim).

Run:
    python -m src.exp_blockchain --config configs/blockchain.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, uuid
from datetime import datetime, timezone
import numpy as np

from .utils import load_config, ensure_dirs, get_logger, save_json


# ----------------------------------------------------------------------
# Evidence manifest + hashing
# ----------------------------------------------------------------------
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(obj) -> bytes:
    """Deterministic JSON serialisation so the same content always hashes equal."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def make_manifest(content: str, source_url: str, collector: str,
                  captured_at: str | None = None, extra: dict | None = None) -> dict:
    """Build a capture-time evidence manifest. The content_hash binds the exact
    bytes collected; the manifest binds the provenance metadata."""
    captured_at = captured_at or datetime.now(timezone.utc).isoformat()
    manifest = {
        "evidence_id": str(uuid.uuid4()),
        "content_hash": sha256_hex(content.encode("utf-8")),
        "content_length": len(content.encode("utf-8")),
        "source_url": source_url,
        "collector": collector,
        "captured_at": captured_at,
    }
    if extra:
        manifest["extra"] = extra
    # the manifest hash binds ALL provenance fields together
    manifest["manifest_hash"] = sha256_hex(canonical_bytes(
        {k: manifest[k] for k in manifest if k != "manifest_hash"}))
    return manifest


# ----------------------------------------------------------------------
# Ledger backends
# ----------------------------------------------------------------------
class LedgerBackend:
    def seal(self, manifest: dict) -> dict: raise NotImplementedError
    def get(self, evidence_id: str) -> dict | None: raise NotImplementedError
    def verify_chain(self) -> bool: raise NotImplementedError
    def all_blocks(self) -> list: raise NotImplementedError


class LocalHashChainLedger(LedgerBackend):
    """Append-only hash chain: block_hash = H(prev_hash || manifest_hash || index).
    Tampering with any sealed manifest breaks the chain from that point forward,
    which verify_chain() detects. Persisted as JSONL so it survives a run."""

    GENESIS = "0" * 64

    def __init__(self, path: str, logger=None):
        self.path = path
        self.logger = logger
        self.blocks = []
        if os.path.exists(path):
            with open(path) as f:
                self.blocks = [json.loads(line) for line in f if line.strip()]

    def _block_hash(self, prev_hash, manifest_hash, index, ts):
        return sha256_hex(canonical_bytes(
            {"prev": prev_hash, "mh": manifest_hash, "i": index, "ts": ts}))

    def seal(self, manifest: dict) -> dict:
        prev = self.blocks[-1]["block_hash"] if self.blocks else self.GENESIS
        index = len(self.blocks)
        ts = datetime.now(timezone.utc).isoformat()
        bh = self._block_hash(prev, manifest["manifest_hash"], index, ts)
        block = {"index": index, "prev_hash": prev, "sealed_at": ts,
                 "evidence_id": manifest["evidence_id"],
                 "manifest_hash": manifest["manifest_hash"],
                 "manifest": manifest, "block_hash": bh}
        self.blocks.append(block)
        with open(self.path, "a") as f:
            f.write(json.dumps(block) + "\n")
        return block

    def get(self, evidence_id: str):
        for b in self.blocks:
            if b["evidence_id"] == evidence_id:
                return b
        return None

    def verify_chain(self) -> bool:
        prev = self.GENESIS
        for i, b in enumerate(self.blocks):
            if b["prev_hash"] != prev:
                if self.logger: self.logger.error(f"Chain break at block {i}: prev mismatch.")
                return False
            # recompute manifest hash (detects content tampering)
            recomputed_mh = sha256_hex(canonical_bytes(
                {k: b["manifest"][k] for k in b["manifest"] if k != "manifest_hash"}))
            if recomputed_mh != b["manifest_hash"]:
                if self.logger: self.logger.error(f"Manifest tampering at block {i}.")
                return False
            # recompute block hash (detects ledger tampering)
            recomputed_bh = self._block_hash(b["prev_hash"], b["manifest_hash"],
                                             b["index"], b["sealed_at"])
            if recomputed_bh != b["block_hash"]:
                if self.logger: self.logger.error(f"Block-hash tampering at block {i}.")
                return False
            prev = b["block_hash"]
        return True

    def all_blocks(self): return list(self.blocks)


def get_backend(cfg, logger):
    ledger_path = cfg["data"].get("ledger_path", "results/ledger.jsonl")
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    return LocalHashChainLedger(ledger_path, logger)


# ----------------------------------------------------------------------
# Experiment: seal real evidence, measure throughput, prove tamper-detection
# ----------------------------------------------------------------------
def _load_evidence(cfg, logger, n):
    """Use real collected text (CoDA) as the evidence items to seal."""
    from .exp_text import load_dataset as load_text
    df, _ = load_text(cfg, logger)
    items = []
    for i, row in df.head(n).iterrows():
        url = row["__key__"] if "__key__" in df.columns else f"item://{i}"
        items.append((str(row["text"]), str(url)))
    return items


def run(cfg, logger):
    n = cfg["data"].get("n_evidence", 500)
    items = _load_evidence(cfg, logger, n)
    ledger = get_backend(cfg, logger)
    logger.info(f"Sealing {len(items)} evidence items via local hash-chain ledger...")

    # seal all items, measuring per-item latency
    t0 = time.time(); latencies = []
    sealed = []
    for content, url in items:
        s = time.time()
        manifest = make_manifest(content, url, collector="DarkTrace-Phase4")
        block = ledger.seal(manifest)
        latencies.append(time.time() - s)
        sealed.append((manifest, block))
    total = time.time() - t0
    lat = np.array(latencies)
    logger.info(f"Sealed {len(items)} items in {total:.2f}s "
                f"({len(items)/total:.1f} items/s); "
                f"latency ms p50={np.percentile(lat,50)*1000:.2f} "
                f"p95={np.percentile(lat,95)*1000:.2f}")

    # integrity verification (clean chain should pass)
    clean_ok = ledger.verify_chain()
    logger.info(f"Chain verification (untampered): {'PASS' if clean_ok else 'FAIL'}")

    # tamper-detection test: mutate a sealed manifest, confirm the chain catches it
    tamper_detected = None
    if ledger.blocks:
        import copy
        saved = copy.deepcopy(ledger.blocks)
        victim = len(ledger.blocks) // 2
        ledger.blocks[victim]["manifest"]["content_hash"] = "deadbeef" * 8
        tamper_detected = not ledger.verify_chain()
        ledger.blocks = saved  # restore in-memory
        logger.info(f"Tamper-detection test: modified block {victim} -> "
                    f"{'DETECTED' if tamper_detected else 'MISSED (BUG)'}")

    results = {
        "experiment": "phase4_blockchain_sealing",
        "reportable": True,
        "backend": "local_hash_chain", "n_evidence": len(items),
        "throughput_items_per_s": len(items) / total if total else None,
        "latency_ms": {"p50": float(np.percentile(lat, 50) * 1000),
                       "p95": float(np.percentile(lat, 95) * 1000),
                       "mean": float(lat.mean() * 1000)},
        "chain_verified_clean": clean_ok,
        "tamper_detected": tamper_detected,
    }
    return results


def write_table8_fragment(results, out_path, logger):
    import pandas as pd
    row = {
        "Backend": results["backend"],
        "N_evidence": results["n_evidence"],
        "Throughput_items_per_s": round(results["throughput_items_per_s"], 1)
            if results["throughput_items_per_s"] else "NA",
        "Latency_p50_ms": round(results["latency_ms"]["p50"], 3),
        "Latency_p95_ms": round(results["latency_ms"]["p95"], 3),
        "Chain_verified": results["chain_verified_clean"],
        "Tamper_detected": results["tamper_detected"],
    }
    pd.DataFrame([row]).to_csv(out_path, index=False)
    logger.info(f"Wrote Table 8 fragment -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/blockchain.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("blockchain", cfg["paths"]["logs"])
    # fresh ledger for a clean measurement unless appending is intended
    lp = cfg["data"].get("ledger_path", "results/ledger.jsonl")
    if not cfg["data"].get("append_ledger", False) and os.path.exists(lp):
        os.remove(lp)
    t0 = time.time()
    results = run(cfg, logger)
    tables = cfg["paths"]["tables"]
    save_json(results, os.path.join(tables, "blockchain_results.json"))
    write_table8_fragment(results, os.path.join(tables, "table8_blockchain.csv"), logger)
    logger.info("Real-data run complete — Table 8 blockchain fragment written.")
    logger.info(f"Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
