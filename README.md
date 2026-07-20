# Benchmarking Vision Foundation Models for Endoscopic Video Understanding

**Research question:** Does frame-level linear-probe accuracy predict the temporal
reliability of frozen vision foundation model features on endoscopic video?

See [`reports/abstract_v0.md`](reports/abstract_v0.md) for the pre-registered abstract,
predictions, and stopping rules. That document is the scope contract.

## Status
Week 1 of 8 (started 2026-07-20). Infrastructure only; no results yet.

## Compute
WashU SLURM cluster. Embeddings cached under `$BIGDIR`, not in this repo.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.extract --config configs/dinov2_cholecseg8k.yaml
```

## Layout
| Path | Purpose |
|---|---|
| `src/encoders/` | One file per model, uniform `Encoder` interface |
| `src/cache/` | Memory-mapped embedding store with SHA256 manifests |
| `src/data/` | Loaders and **video-level** splits (never frame-level) |
| `src/metrics/` | Temporal reliability suite |
| `src/corruptions/` | Endoscopy-specific corruption suite |
| `src/experiments/` | Pure config -> results.json entrypoints |
| `reports/` | Abstract, figures, technical report |
| `LEDGER.md` | Running log of failures and dead ends |
