# SurgicalVision3D-Planner

`SurgicalVision3D-Planner` is a 3D Slicer scripted extension for research-oriented ablation planning. The repository combines interactive planning workflows, beginner-guided single-applicator steps, deterministic trajectory-array generation, geometric margin and safety analysis, cohort summaries, structured exports, and reproducibility package assembly.

The active extension lives under `app/`. Historical migration code is kept under `Legacy/` for reference.

## License and ownership

This repository is proprietary and restricted. It is not open source and is not licensed for public use, execution, modification, redistribution, research use, clinical use, benchmarking, or machine-learning use without prior written authorization from SurgicalVision3D.

- Owner: SurgicalVision3D
- Website: https://surgicalvision3d.com/en
- Corporate representative: Baptiste Savard, CEO
- Scientific and R&D lead: Juan M. Verde, MD, MSc. (Business, Engineering and Surgical Technology Transfer), Research Scientist (Translational Research), Preclinical Research, and Innovation Manager

See [LICENSE.md](LICENSE.md) for the full restricted proprietary license.

## What the repo contains

- `app/SurgicalVision3D_Planner/SurgicalVision3D_Planner.py`: main scripted module source
- `app/SurgicalVision3D_Planner/Resources/UI/SurgicalVision3D_Planner.ui`: base Qt UI
- `app/SurgicalVision3D_Planner/Resources/Geometries/`: bundled STL applicator/ablation templates
- `app/SurgicalVision3D_Planner/Resources/Cohorts/`: sample case and cohort resources
- `app/SurgicalVision3D_Planner/Resources/Reproducibility/`: package schemas and templates
- `app/SurgicalVision3D_Planner/Testing/Python/`: Slicer Python tests
- `LICENSE.md`: restricted proprietary no-use license
- `DOCUMENTATION.md`: detailed repository documentation

## Quick start in Slicer

1. Open 3D Slicer.
2. Add `.../SurgicalVision3D-Planner/app` to `Edit -> Application Settings -> Modules -> Additional module paths`.
3. Restart Slicer.
4. Open `SurgicalVision3D Planner`.
5. Optionally load the bundled `CRLM-1001` sample case from the module UI.
6. Select a reference probe segmentation, endpoint markups, and tumor segmentation.
7. Run `Place Probes`, `Merge Translated Probes`, and `Evaluate Margins`.

## Main workflows

- General planning: paired entry/endpoint trajectories, probe placement, merged ablation, margin and safety evaluation.
- Beginner workflow: import, segment, define one master trajectory, validate, optionally auto-adjust endpoint, evaluate MAM, lock plan, compute coaxial guidance.
- Multi-trajectory planning: generate deterministic parallel arrays around one master trajectory.
- Batch and portability: cohort summaries, export bundles, and reproducibility packages.

## Documentation map

- [Detailed repository documentation](DOCUMENTATION.md)
- [Release guide](RELEASE.md)
- [Prerequisites](app/SurgicalVision3D_Planner/PREREQUISITES.md)
- [Usage guide](app/SurgicalVision3D_Planner/USAGE_GUIDE.md)
- [Beginner step-by-step guide](app/SurgicalVision3D_Planner/BEGINNER_STEP_BY_STEP.md)
- [Refactor and phase notes](app/SurgicalVision3D_Planner/PHASE1_REFACTOR_NOTES.md)
- [Derived trajectory array specification](app/SurgicalVision3D_Planner/MULTI_TRAJECTORY_ARRAY_SPEC.md)

## Testing

The repo includes Slicer Python tests for:

- trajectory extraction and vector math
- deterministic node reuse and repeated-run behavior
- margin, safety, and coordination summaries
- endpoint auto-adjust logic
- export bundle generation
- cohort aggregation
- reproducibility package assembly

Primary test entry points are:

- `app/SurgicalVision3D_Planner/Testing/Python/SurgicalVision3D_PlannerPhase1Test.py`
- the embedded unittest registration in `app/SurgicalVision3D_Planner/SurgicalVision3D_Planner.py`

## Current scope

This is a research planning extension, not a real-time navigation platform and not a physics-validated simulator. Most computations are deliberate geometric approximations chosen for transparency, deterministic behavior, and reviewability.

Repository access does not grant permission to use the software. The governing restrictions are in [LICENSE.md](LICENSE.md).
