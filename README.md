# SurgicalVision3D-Planner

SurgicalVision3D-Planner is a 3D Slicer scripted module for research-oriented ablation planning.
It helps users move from trajectory definition to quantitative evaluation, scenario exploration,
and reproducible export workflows.

This README is written for both:
- non-expert users who need a clear operational workflow
- technical users who need deterministic outputs for testing and study work

## What This Project Does

At a high level, the module supports:
- trajectory planning from entry/target point pairs
- synthetic probe placement and merged ablation geometry
- margin, safety, coverage, feasibility, and coordination analysis
- scenario snapshots and candidate generation/filtering/recommendation
- explicit candidate promotion back to the working plan (no silent overwrite)
- benchmark and cohort/study workflows for repeatable evaluation
- structured exports, reports, and reproducibility packages

## Who This Is For

- Clinical/research planning teams exploring ablation strategy variants
- Technical collaborators validating deterministic behavior across cases
- Developers extending the module in a phase-based, reviewable way

## Main Module Files

- Module root: `app/SurgicalVision3D_Planner/`
- Python entry point: `app/SurgicalVision3D_Planner/SurgicalVision3D_Planner.py`
- Qt UI file: `app/SurgicalVision3D_Planner/Resources/UI/SurgicalVision3D_Planner.ui`
- Tests: `app/SurgicalVision3D_Planner/Testing/Python/SurgicalVision3D_PlannerPhase1Test.py`
- Phase notes: `app/SurgicalVision3D_Planner/PHASE1_REFACTOR_NOTES.md`
- End-user guide: `app/SurgicalVision3D_Planner/USAGE_GUIDE.md`

## Quick Start (First Run in 3D Slicer)

1. Open 3D Slicer.
2. Add the local module path in Additional module paths:
   - `.../SurgicalVision3D-Planner/app`
3. Restart Slicer or reload modules.
4. Open module: `SurgicalVision3D Planner`.
5. In **Inputs**, select:
   - reference probe segmentation
   - endpoints markups (entry/target pairs)
   - tumor segmentation
6. Run:
   - `Place Probes`
   - `Merge Translated Probes`
   - `Evaluate Margins`
7. Review generated `SV3D ...` tables in the Data module.

## Key Terms (Plain Language)

- **Reference probe segmentation**:
  A template probe/applicator geometry duplicated and placed along each trajectory.
- **Endpoints markups**:
  Ordered points that define trajectories in pairs: `entry1,target1,entry2,target2,...`.
- **Working plan**:
  The current active planning state in the scene.
- **Scenario**:
  A saved planning snapshot used for comparison and recommendation workflows.
- **Candidate**:
  A generated scenario variant from deterministic perturbation rules.
- **Promotion**:
  Explicitly applying a selected scenario/candidate to the working plan.
- **Cohort/Study**:
  Batch execution across multiple case members for aggregate analysis.

## Workflow Overview

1. Define trajectories and generate probe placements.
2. Merge generated probes into an ablation-zone representation.
3. Evaluate quantitative metrics (margin, safety, coverage, etc.).
4. Create scenarios and compare alternatives.
5. Generate/filter candidates and review recommendations.
6. Promote one selected candidate explicitly if needed.
7. Export structured outputs and reproducibility packages.

## Deterministic Output Philosophy

The module creates deterministic owned outputs (`SV3D ...` tables/models/segmentations)
and reuses them across reruns where possible.
This reduces stale duplicates and improves reproducibility/regression tracking.

## Export and Reproducibility

The module supports structured bundle exports and reproducibility package generation with:
- deterministic folder naming with sequence suffixes
- machine-readable CSV/JSON artifacts
- manifest-based provenance
- optional inclusion of benchmark/cohort/study/report artifacts when present

## Repository Structure

- `app/`
  Active module code, resources, tests, and documentation.
- `Legacy/`
  Historical code kept for migration/reference context.

## Testing and Validation

Primary scripted tests are located at:
- `app/SurgicalVision3D_Planner/Testing/Python/SurgicalVision3D_PlannerPhase1Test.py`

These tests focus on deterministic helper behavior and table-generation logic.
Some runtime checks still require manual validation inside Slicer.

## Scope and Limitations

This is a research planning platform, not a real-time navigation system and not a validated physics simulator.
Current models are conservative and geometry/rule based for transparency and auditability.

## Credits

- Juan Verde