# SurgicalVision3D-Planner

`SurgicalVision3D-Planner` is a proprietary 3D Slicer extension for research-oriented ablation planning. On `dev/beginner-module`, the module emphasizes a guided beginner workflow for single-applicator planning while retaining the broader planning, margin-analysis, export, cohort, and reproducibility infrastructure in the same codebase.

The active extension project lives under `app/`. Historical migration context is kept in notes such as `PHASE1_REFACTOR_NOTES.md`; this branch does not contain a checked-in `Legacy/` source tree.

## Repository at a glance

- `app/`: Slicer extension project loaded by Slicer or built with CMake.
- `app/SurgicalVision3D_Planner/SurgicalVision3D_Planner.py`: main scripted module implementation.
- `app/SurgicalVision3D_Planner/Resources/UI/`: Qt UI definition and icons.
- `app/SurgicalVision3D_Planner/Resources/Geometries/`: bundled STL applicator and ablation templates.
- `app/SurgicalVision3D_Planner/Resources/Cohorts/`: bundled sample case assets, cohort schemas, and example study definitions.
- `app/SurgicalVision3D_Planner/Resources/Reproducibility/`: reproducibility package schemas and templates.
- `app/SurgicalVision3D_Planner/Testing/Python/`: Slicer runtime tests.
- `QUICK_START.md`: first-run setup guide.
- `QUICK_USAGE.md`: short workflow guide for everyday use.
- `INSTRUCTIONS_FOR_USE.md`: detailed operator instructions for the module.
- `SOP.md`: standard operating procedure for controlled research use.
- `DOCUMENTATION.md`: detailed architecture and repository notes.
- `RELEASE.md`: release checklist and tag guidance.

## Main capabilities

- Beginner single-applicator planning: import a case, segment structures, define one master trajectory, validate it, optionally auto-adjust the endpoint, evaluate MAM, lock the plan, and compute coaxial guidance.
- General planning workflow: paired entry/endpoint trajectories, probe placement, merged ablation-zone generation, margin evaluation, safety checks, and coordination summaries.
- Deterministic derived trajectory arrays around one validated master trajectory.
- Export bundles, cohort summaries, and reproducibility package assembly.
- Bundled `CRLM-1001` sample resources for repeatable demo and regression workflows.

## Quick start

1. Open 3D Slicer.
2. Add `<repo>/app` to `Edit -> Application Settings -> Modules -> Additional module paths`.
3. Restart Slicer and open `SurgicalVision3D Planner`.
4. Load the bundled `CRLM-1001` sample case or import a local case folder.
5. In beginner mode, work in this order: segment structures, place exactly two points in `entry -> endpoint` order, validate, place the applicator, merge, evaluate MAM, lock the plan, and compute coaxial guidance.

Use these docs for the short path:

- [Quick start](QUICK_START.md)
- [Quick usage](QUICK_USAGE.md)
- [Instructions for use](INSTRUCTIONS_FOR_USE.md)
- [Standard operating procedure](SOP.md)

## Development and testing

Development is centered on the `app/` extension tree. The repository can be loaded directly into Slicer from source or built as a normal Slicer extension through [app/CMakeLists.txt](app/CMakeLists.txt).

The main automated test entry points are:

- `app/SurgicalVision3D_Planner/Testing/Python/SurgicalVision3D_PlannerPhase1Test.py`
- the embedded unittest registration in `app/SurgicalVision3D_Planner/SurgicalVision3D_Planner.py`

The tests cover trajectory extraction, deterministic node reuse, margin and safety summaries, endpoint auto-adjust behavior, export bundle generation, cohort aggregation, and reproducibility packaging.

## Documentation map

- [Quick start](QUICK_START.md)
- [Quick usage](QUICK_USAGE.md)
- [Instructions for use](INSTRUCTIONS_FOR_USE.md)
- [Standard operating procedure](SOP.md)
- [Detailed repository documentation](DOCUMENTATION.md)
- [Release guide](RELEASE.md)
- [Prerequisites](app/SurgicalVision3D_Planner/PREREQUISITES.md)
- [Usage guide](app/SurgicalVision3D_Planner/USAGE_GUIDE.md)
- [Beginner step-by-step guide](app/SurgicalVision3D_Planner/BEGINNER_STEP_BY_STEP.md)
- [Refactor and phase notes](app/SurgicalVision3D_Planner/PHASE1_REFACTOR_NOTES.md)
- [Derived trajectory array specification](app/SurgicalVision3D_Planner/MULTI_TRAJECTORY_ARRAY_SPEC.md)

## License and ownership

This repository is proprietary and restricted. It is not open source and it is not licensed for public use, execution, modification, redistribution, research use, clinical use, benchmarking, or machine-learning use without prior written authorization from SurgicalVision3D.

- Owner: SurgicalVision3D
- Website: https://surgicalvision3d.com/en
- Corporate representative: Baptiste Savard, CEO
- Scientific and R&D lead: Juan M. Verde, MD, MSc. (Business, Engineering and Surgical Technology Transfer), Research Scientist (Translational Research), Preclinical Research, and Innovation Manager

This is a research planning extension, not a real-time navigation platform and not a physics-validated simulator. Most computations are deliberate geometric approximations chosen for transparency, deterministic behavior, and reviewability.

See [LICENSE.md](LICENSE.md) for the full restricted proprietary license.
