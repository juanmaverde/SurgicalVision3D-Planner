# SurgicalVision3D Planner Reproducibility Package

This package was generated in deterministic sequence mode for reviewer and validation workflows.

## What this package contains

- `manifest.json`: package metadata, provenance, artifact listing, and warnings.
- `schemas/`: schema and layout references for this package format.
- `benchmarks/`: benchmark definitions and benchmark resources when included.
- `validation/`: benchmark and validation run summaries when available.
- `cohorts/`: cohort definitions and cohort summary outputs when available.
- `study_analytics/`: study-level analytics summaries when available.
- `reports/`: report-oriented artifacts when available.
- `exports/`: copied export bundle artifacts and provenance tables.
- `canonical_json/`: canonical machine-readable summaries.

## Determinism and integrity notes

- Package folders are sequence-versioned: `<base>_<sequence>`.
- Optional artifacts that were requested but unavailable are listed in `manifest.json` warnings.
- File integrity fields include size bytes and lightweight SHA256 hashes when file size is within the configured cheap-hash limit.

## Reviewer checklist

1. Open `manifest.json` and review package mode, sequence, and warnings.
2. Confirm required artifacts for your review objective are present.
3. Use CSV and JSON files directly for downstream analysis in Python/R/Excel/LaTeX.
4. Keep this folder read-only for frozen supplement archives.
