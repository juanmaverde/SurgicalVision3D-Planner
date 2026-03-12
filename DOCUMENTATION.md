# SurgicalVision3D-Planner Repository Documentation

## 1. Overview

`SurgicalVision3D-Planner` is a 3D Slicer scripted extension for research-oriented ablation planning. The active codebase lives under `app/` and is centered around one scripted module, `SurgicalVision3D_Planner`, that combines:

- trajectory definition from markup point pairs
- probe/applicator placement from reusable geometry templates
- merged ablation-zone generation
- geometric margin and safety analysis
- deterministic multi-trajectory array generation
- beginner-oriented single-master planning and coaxial guidance
- cohort-style batch summaries
- export bundles and reproducibility package assembly
- lightweight git status operations inside the module UI

This repository is not a generic Python package and not a web application. It is a Slicer extension project with CMake entry points, MRML scene objects, Qt UI resources, bundled case assets, and Slicer runtime tests.

The repository is proprietary and restricted. Repository access does not grant a public-use license. Use, execution, modification, redistribution, clinical use, research use, benchmarking, or machine-learning use require prior written authorization from SurgicalVision3D.

Two principles show up throughout the code:

1. Deterministic outputs: generated nodes, tables, export folders, and reproducibility packages are reused or sequence-numbered deliberately.
2. Explicit ownership: module-created nodes are tagged with ownership attributes so repeated runs can clear or reuse them without silently mutating user-owned data.

## 2. Repository Structure

### Top level

```text
.
|-- INSTRUCTIONS_FOR_USE.md
|-- QUICK_START.md
|-- QUICK_USAGE.md
|-- SOP.md
|-- app/
|   |-- CMakeLists.txt
|   `-- SurgicalVision3D_Planner/
|       |-- SurgicalVision3D_Planner.py
|       |-- CMakeLists.txt
|       |-- PREREQUISITES.md
|       |-- USAGE_GUIDE.md
|       |-- BEGINNER_STEP_BY_STEP.md
|       |-- PHASE1_REFACTOR_NOTES.md
|       |-- MULTI_TRAJECTORY_ARRAY_SPEC.md
|       |-- Resources/
|       |   |-- UI/
|       |   |-- Icons/
|       |   |-- Geometries/
|       |   |-- Cohorts/
|       |   `-- Reproducibility/
|       `-- Testing/
|           `-- Python/
|-- LICENSE.md
|-- RELEASE.md
|-- SurgicalVision3D_Planner.png
|-- README.md
`-- DOCUMENTATION.md
```

### What each major path is for

| Path | Purpose |
| --- | --- |
| `app/CMakeLists.txt` | Extension-level CMake entry point used by Slicer. |
| `app/SurgicalVision3D_Planner/CMakeLists.txt` | Registers the scripted module, resources, and tests. |
| `app/SurgicalVision3D_Planner/SurgicalVision3D_Planner.py` | Main module source: constants, dataclasses, parameter node, widget, logic, and some tests. |
| `app/SurgicalVision3D_Planner/Resources/UI/SurgicalVision3D_Planner.ui` | Qt Designer `.ui` file for the non-beginner/base UI. |
| `app/SurgicalVision3D_Planner/Resources/Geometries/` | STL ablation/applicator templates used as reference probe geometries. |
| `app/SurgicalVision3D_Planner/Resources/geometry_catalog.json` | Beginner workflow geometry catalog with labels, template files, active-element length, and optional axial offsets. |
| `app/SurgicalVision3D_Planner/Resources/Cohorts/` | Sample case assets, cohort schema/catalog metadata, and example cohort definition JSON. |
| `app/SurgicalVision3D_Planner/Resources/Reproducibility/` | Schema/layout/config templates for reproducibility package generation. |
| `app/SurgicalVision3D_Planner/Testing/Python/` | Dedicated Slicer Python tests for logic behavior and deterministic output rules. |
| `INSTRUCTIONS_FOR_USE.md` | Detailed operator-focused instructions for loading, running, reviewing, and exporting planning sessions. |
| `QUICK_START.md` | Short first-run setup guide for loading the extension in Slicer and completing one beginner plan. |
| `QUICK_USAGE.md` | Short operational guide for the beginner and general planning workflows. |
| `SOP.md` | Standard operating procedure for repeatable research execution, review, and record-keeping. |
| `LICENSE.md` | Restricted proprietary no-use license and ownership statement. |
| `RELEASE.md` | Release checklist, validation expectations, and version/tag guidance. |

## 3. Runtime Architecture

The active module is implemented almost entirely in `app/SurgicalVision3D_Planner/SurgicalVision3D_Planner.py`. That file is intentionally large because it follows the standard Slicer scripted-module pattern in one place.

### 3.1 Main layers

| Layer | Role |
| --- | --- |
| `SurgicalVision3D_Planner` | Slicer module metadata and sample-data registration hooks. |
| `SurgicalVision3D_PlannerParameterNode` | Typed persistent state for selected nodes, numeric settings, toggles, JSON summaries, and generated-node IDs. |
| `SurgicalVision3D_PlannerWidget` | UI orchestration, button callbacks, state gating, beginner workflow assembly, and git dashboard wiring. |
| `SurgicalVision3D_PlannerLogic` | Core geometric, export, cohort, and reproducibility logic. |
| `SurgicalVision3D_PlannerTest` plus `Testing/Python/SurgicalVision3D_PlannerPhase1Test.py` | Runtime tests for math, scene lifecycle, determinism, export, cohort, and reproducibility behavior. |

### 3.2 Architectural style

The widget does not carry long-term planning state in ad hoc instance variables. Instead:

- most persistent state is stored in the parameter node
- the widget binds UI elements to the parameter node
- the logic layer consumes MRML nodes and plain dataclasses
- outputs are materialized back into the scene as MRML nodes and tables

This matters because scenes can be saved, reloaded, batch-processed, and inspected through Slicer tools outside this module.

### 3.3 Output ownership model

Generated outputs are tagged with attributes such as:

- `SurgicalVision3D_Planner.GeneratedProbe`
- `SurgicalVision3D_Planner.GeneratedTrajectoryLine`
- `SurgicalVision3D_Planner.GeneratedCombinedProbe`
- `SurgicalVision3D_Planner.GeneratedMarginModel`
- `SurgicalVision3D_Planner.GeneratedResultTable`

The logic then uses `createOrReuseOwnedOutputNode(...)`, `removeNodeIfOwned(...)`, and related helpers to:

- reuse existing module-owned nodes when safe
- avoid duplicating stale outputs on repeated runs
- avoid overwriting user-owned nodes that happen to be selected in the UI

This is one of the core maintainability rules in the repository.

## 4. Core Functional Areas

### 4.1 Case loading and sample data

The module supports two common entry paths:

- bundled sample scenes under `Resources/Cohorts/`
- direct folder import for a case directory

Case-folder import is conservative:

- it resolves the selected folder path
- it loads the largest non-segmentation `.nrrd` as the main volume
- it loads all `.seg.nrrd` files in the folder
- it loads all markups files matching `*.mrk.json` or `*.fcsv`
- it ensures segmentation reference image geometry after loading
- it then ensures the reference probe templates are available in the scene

Bundled scene loading includes fallback logic:

- if direct `.mrml` scene loading is noisy or incomplete, the code falls back to case-asset loading from the scene folder
- if a referenced `.seg.vtm` file is missing, the loader attempts repair from a matching STL source when possible

Included sample resources currently center on the `CRLM-1001` demo case.

### 4.2 Probe placement workflow

The general planning workflow expects:

- a reference probe segmentation
- an endpoints markups node
- a tumor segmentation

Trajectory extraction rule:

- control points are interpreted as `entry1, endpoint1, entry2, endpoint2, ...`
- odd point counts can be rejected strictly or truncated safely depending on the call site

Probe placement rule:

- the reference geometry is assumed to point along local negative Z
- the code computes a rigid transform that aligns `[0, 0, -1]` to each trajectory direction
- the transformed instance is placed at the entry point

Optional helpers:

- create visible trajectory lines
- clear previous generated probes/lines before a new run
- merge translated probes into a single combined ablation segmentation

### 4.3 Beginner workflow

The repository contains a strong beginner-oriented layer for a linear, single-master workflow. In practical terms that means:

1. Import a case folder or bundled sample.
2. Confirm tumor and critical-structure segmentations.
3. Define exactly one master trajectory using exactly two points:
   - point 1 = entry
   - point 2 = applicator endpoint
4. Validate that master trajectory against critical structures.
5. If validation fails, optionally run `Auto-adjust Endpoint`.
6. Choose an applicator geometry from `Resources/geometry_catalog.json`.
7. Place the applicator, merge outputs, and evaluate MAM.
8. Lock the validated plan.
9. Compute coaxial guidance for `PullBack` or `PushThrough`.

The beginner workflow is stateful and gated. Buttons are enabled only when upstream conditions are met, for example:

- exactly one trajectory is present
- validation passed
- MAM has been evaluated before lock/coaxial steps

### 4.4 Master-trajectory validation and endpoint rescue

Critical-structure validation is geometry-based:

- each usable critical-structure segment is converted/prepared as closed surface
- the line from entry to endpoint is checked for intersection
- minimum line-to-surface distance is computed
- summary rows and a pass/fail summary are produced

Auto-adjust endpoint is deterministic and intentionally conservative:

- entry point stays fixed
- trajectory length stays fixed
- the endpoint must remain inside the tumor surface
- the search explores shells up to `15 mm` by default
- shell step is `1.5 mm` by default
- azimuth sampling is `36` directions by default
- only zero-intersection candidates are accepted
- if no safe candidate exists within the cap, the endpoint is left unchanged

This is designed as a rescue heuristic, not an optimizer.

### 4.5 Derived trajectory arrays

The module also supports deterministic symmetric trajectory bundles derived from one validated master trajectory.

Supported controls:

- `trajectoryPlanningMode`: `Single` or `MultiTrajectoryArray`
- derived trajectory count
- radial offset in mm
- angular offset in degrees
- include/exclude the master trajectory from the bundle

Key implementation rules:

- all child trajectories remain parallel to the master
- all child trajectories preserve master length
- child axes are translated in the plane orthogonal to the master direction
- the orthogonal basis is deterministic in patient space
- for four derived trajectories, roles are labeled `North`, `East`, `South`, `West`
- for other counts, children are labeled `Derived 01`, `Derived 02`, and so on

The beginner UI can preview and clear this array without rewriting the master markup node.

### 4.6 Margin evaluation and MAM

Margin evaluation converts:

- the combined probe segmentation into a model
- the tumor segmentation into a model

It then computes a signed distance field on the tumor surface relative to the ablation geometry and publishes:

- `SV3D Signed Margin Model`
- `SV3D Signed Margin Table`
- `SV3D Plan Summary`
- `SV3D Margin Threshold Summary`

For beginner MAM assessment:

- achieved margins are computed by negating the stored signed-margin values
- default MAM is `10 mm`
- color buckets are:
  - red: below `0.5 x MAM`
  - orange: between `0.5 x MAM` and `MAM`
  - green: at least `MAM`

Important implication:

- the raw signed-distance model and the beginner MAM coloring are related but not identical visualizations
- the code keeps a backup copy of the original signed distances so recolor and reset operations are reversible

### 4.7 Structure safety and probe coordination

Optional structures-at-risk evaluation:

- iterates all valid risk-structure segments
- computes signed closest-point distances against the combined ablation zone
- negative values indicate overlap/collision
- publishes summary and threshold tables

Probe coordination evaluates pairwise arrangement quality using trajectory geometry:

- inter-probe distance
- entry-point spacing
- target-point spacing
- axis-angle difference
- conservative overlap proxy
- optional no-touch check for entry points outside tumor

Outputs include:

- settings table
- pairwise summary table
- plan-level coordination summary
- optional no-touch summary

### 4.8 Registration

Registration support uses Slicer's `fiducialregistration` CLI in rigid mode. The workflow is:

- select native and registered fiducial nodes
- compute rigid transform
- apply the transform to the tumor segmentation
- optionally harden the transform

The repository documentation under `PREREQUISITES.md` explicitly calls out this dependency.

### 4.9 Export bundles

The export subsystem creates deterministic sequence-numbered folders:

- folder name pattern: `<exportBaseName>_<sequence4>`
- if the target folder already exists, the sequence increments

The bundle includes:

- `manifest.json`
- `plan_summary.json`
- optional `scenario_summary.json`
- `tables/*.csv`
- provenance JSON when scenario/recommendation tables exist

The export layer opportunistically includes other in-scene tables too, such as cohort, reproducibility, coverage, scenario, or recommendation outputs when present.

This design lets the exporter work across different module phases without hard-failing when optional tables do not exist.

### 4.10 Cohort evaluation

The cohort subsystem is table-driven, not full raw-image replay.

It loads a JSON study definition with:

- `studyId`
- optional metadata
- a list of case definitions

Supported case input references:

- `ScenarioID`
- `CurrentWorkingPlan`

The cohort run:

- resolves metrics from existing in-scene summary tables
- records per-case success or failure without aborting the full batch
- aggregates descriptive statistics
- groups results by preset ID for comparison summaries

This is useful for deterministic study-style review, but it is not yet a full batch planner that replays every case from original imaging assets.

### 4.11 Reproducibility package

The reproducibility subsystem assembles a frozen package with explicit layout folders:

- `schemas`
- `benchmarks`
- `validation`
- `cohorts`
- `study_analytics`
- `reports`
- `exports`
- `canonical_json`

Package assembly rules:

- package folders are sequence-numbered deterministically
- artifacts are copied when source files already exist
- CSV/JSON artifacts can be regenerated from in-scene tables or summaries
- missing optional artifacts become manifest warnings instead of silent failures
- integrity metadata includes size and lightweight SHA256 hashes

This is intended for reviewer supplements, validation archives, and internal handoff snapshots.

### 4.12 Git dashboard

The widget includes a git dashboard panel that can:

- refresh repository status
- stage all changes
- commit with a message
- push the current branch
- append operator notes to `git_agent_log.md`

This is a convenience layer inside Slicer rather than a replacement for normal git tooling. It is most useful for quick provenance notes and basic working-tree inspection during interactive planning sessions.

## 5. Data Models and Persistent State

The repository uses dataclasses heavily to keep domain state explicit.

### 5.1 Important dataclasses

| Dataclass | Purpose |
| --- | --- |
| `ProbeTrajectory` | Normalized internal representation of a single trajectory. |
| `GeometryCatalogEntry` | Catalog row for bundled applicator geometry. |
| `ProbeCoordinationConstraintSettings` | Pairwise/plan-level spacing and feasibility settings. |
| `PlanExportConfig` / `PlanExportManifest` | Export request parameters and manifest schema. |
| `CohortCaseMember`, `CohortStudyDefinition`, `CohortExecutionConfig`, `CohortCaseResult` | Table-driven cohort execution model. |
| `ReproducibilityPackageConfig`, `ReproducibilityArtifactEntry`, `ReproducibilityManifest` | Reproducibility packaging model. |
| `LockedMasterPlanSnapshot` | Frozen beginner master-plan snapshot used before coaxial guidance. |
| `CoaxialPlanSummary` | Navigation target and explanatory notes for coaxial planning. |
| `DerivedTrajectoryArrayConfig`, `DerivedTrajectoryDescriptor`, `DerivedTrajectoryArraySummary` | Derived-array planning configuration and summary model. |
| `EndpointAutoAdjustResult` | Deterministic endpoint rescue result and search statistics. |

### 5.2 Parameter node categories

The parameter node stores a large amount of persistent state. It can be understood in groups:

| Group | Examples |
| --- | --- |
| Scene inputs | case folder path, selected geometry, reference probe segmentation, endpoints markups, tumor segmentation, critical structures, risk structures, fiducials |
| Scene outputs | combined ablation segmentation, margin model, result table, validation tables, cohort/export/repro tables, coaxial target |
| Planning controls | create trajectory lines, clear previous probes, planning mode, array count/radius/angle, include master |
| Beginner controls | MAM threshold, lock state, coaxial technique, coaxial spare |
| Coordination controls | min/max spacing, angle threshold, overlap threshold, no-touch toggle, gate toggles |
| Export controls | export mode, scenario ID, base name, directory, include flags, last sequence |
| Cohort controls | study definition path, execution mode, include flags, max cases |
| Reproducibility controls | package mode, include toggles, output directory, package base name, last sequence |
| Serialized summaries | generated node ID JSON strings and JSON summary blobs for validation, MAM, locked plan, coaxial plan, and bundle summaries |

Because this state is stored on the MRML parameter node, scenes can restore a substantial portion of workflow context.

## 6. Resources and Bundled Content

### 6.1 UI resources

The base GUI is defined in:

- `app/SurgicalVision3D_Planner/Resources/UI/SurgicalVision3D_Planner.ui`

The widget class then augments that UI in Python by:

- wiring callbacks
- configuring tooltips
- adding beginner-specific sections
- creating the git dashboard
- enabling/disabling sections according to workflow state

### 6.2 Geometry resources

The geometry library is defined by:

- STL files under `Resources/Geometries/`
- metadata in `Resources/geometry_catalog.json`

Current catalog entries include Emprint and Cool-tip presets. Each entry carries:

- stable ID
- display label
- template file path
- active-element length in mm
- optional axial placement offset in mm

When extending the catalog, keep file-path casing consistent with `Resources/Geometries/`. This matters for case-sensitive installs and packaged releases.

### 6.3 Cohort resources

Bundled cohort resources include:

- `cohort_schema_v1.json`
- `cohort_catalog_v1.json`
- `studies/example_cohort_v1.json`
- the `CRLM-1001` sample case directory with scene, volume, segmentations, markups, and some private reference artifacts

### 6.4 Reproducibility resources

Bundled reproducibility assets include:

- `reproducibility_package_schema_v1.json`
- `reproducibility_package_layout_v1.json`
- `example_reviewer_package_config_v1.json`
- `README_package_template.md`

These resources are copied into generated reproducibility packages where appropriate.

### 6.5 Historical migration context

This branch does not include a checked-in `Legacy/` source tree. Historical migration context is instead captured in notes such as `app/SurgicalVision3D_Planner/PHASE1_REFACTOR_NOTES.md`.

## 7. Key Output Nodes and Tables

The module relies heavily on named MRML tables for auditability and downstream export.

### 7.1 Common output names

| Output | Meaning |
| --- | --- |
| `SV3D Combined Ablation Zone` | Merged segmentation from placed probes. |
| `SV3D Signed Margin Model` | Model storing signed-distance field data for tumor-surface margin analysis. |
| `SV3D Signed Margin Table` | Raw table view of signed-margin results. |
| `SV3D Trajectory Summary` | Per-trajectory metrics. |
| `SV3D Derived Trajectory Bundle Summary` | Summary rows for master plus derived trajectories. |
| `SV3D Plan Summary` | Aggregate signed-margin metrics for the current plan. |
| `SV3D Margin Threshold Summary` | Bucketed signed-margin threshold counts. |
| `SV3D Structure Safety Summary` | Per-structure safety distance summary. |
| `SV3D Structure Safety Threshold Summary` | Per-structure safety threshold counts. |
| `SV3D Probe Coordination Constraint Settings` | Persisted coordination rule settings. |
| `SV3D Probe Pair Coordination Summary` | Pairwise feasibility metrics. |
| `SV3D Probe Coordination Summary` | Plan-level coordination gate summary. |
| `SV3D NoTouch Summary` | Entry-point no-touch result. |
| `SV3D Master Trajectory Validation` | Beginner master validation rows. |
| `SV3D Locked Master Plan` | Frozen master-plan snapshot. |
| `SV3D Coaxial Plan Summary` | Coaxial planning summary. |
| `SV3D Export Summary` / `SV3D Export Manifest Preview` | Last export operation preview tables. |
| `SV3D Cohort ...` tables | Batch execution, case, aggregate, and comparison summaries. |
| `SV3D Reproducibility ...` tables | Reproducibility package status, manifest preview, and artifact index. |

### 7.2 Why these tables matter

They are used for three different purposes:

- human inspection inside Slicer
- deterministic regression targets in tests
- machine-readable inputs to export and reproducibility packaging

## 8. Development Notes

### 8.0 Ownership and licensing

This repository is governed by the restricted proprietary license in `LICENSE.md`.

Key points:

- owner: SurgicalVision3D
- website: https://surgicalvision3d.com/en
- corporate representative: Baptiste Savard, CEO
- scientific and R&D lead named in the license: Juan M. Verde, MD, MSc. (Business, Engineering and Surgical Technology Transfer), Research Scientist (Translational Research), Preclinical Research, and Innovation Manager

Practical effect:

- this repository is not open source
- repository visibility or access does not grant permission to use the software
- no research, development, clinical, educational, benchmarking, redistribution, or machine-learning use is allowed without prior written authorization

### 8.1 How the extension is loaded

The simplest development path is to point Slicer at the `app/` directory:

1. Open 3D Slicer.
2. Go to `Edit -> Application Settings -> Modules`.
3. Add the repository's `app` directory to `Additional module paths`.
4. Restart Slicer.
5. Open the `SurgicalVision3D Planner` module.

The CMake files also support building this as a normal Slicer extension project.

### 8.2 Where to make changes

| If you want to change... | Start here |
| --- | --- |
| Buttons, selectors, labels in the base UI | `Resources/UI/SurgicalVision3D_Planner.ui` |
| Beginner-specific workflow behavior | `SurgicalVision3D_PlannerWidget` methods in `SurgicalVision3D_Planner.py` |
| Core geometry, validation, export, or packaging behavior | `SurgicalVision3D_PlannerLogic` in `SurgicalVision3D_Planner.py` |
| Geometry presets | `Resources/geometry_catalog.json` and `Resources/Geometries/` |
| Sample cases or cohort definitions | `Resources/Cohorts/` |
| Reproducibility package templates and schemas | `Resources/Reproducibility/` |
| Tests | `Testing/Python/SurgicalVision3D_PlannerPhase1Test.py` and embedded module tests |

### 8.3 Existing design conventions

Important conventions already established in the code:

- prefer deterministic names that begin with `SV3D`
- reuse owned output nodes instead of creating duplicates
- keep user-authored nodes separate from generated nodes
- use typed dataclasses for exported/derived state rather than anonymous dicts when the data has domain meaning
- keep optional workflows tolerant of missing upstream tables; record warnings instead of failing silently

### 8.4 Recommended change strategy

When adding a feature:

1. decide whether the state belongs in the parameter node
2. add or update a dataclass if the feature has structured domain state
3. keep generated nodes explicitly owned
4. add tests for deterministic behavior and repeated-run lifecycle
5. update user docs under `app/SurgicalVision3D_Planner/` if the workflow changes

### 8.5 Release preparation

Current release expectations for this repository:

- `app/SurgicalVision3D_Planner/CMakeLists.txt` is the packaging source of truth for runtime assets. If you add geometry templates, demo-case files, or packaged module docs, update that file in the same change.
- Keep the bundled template directory spelled as `Resources/Geometries/` in code, docs, and catalog metadata.
- Keep a stable demo thumbnail at `Resources/Cohorts/CRLM-1001/DemoScene.png` because sample-data registration expects that filename.
- Follow `RELEASE.md` for the final validation pass and git tag workflow.

## 9. Testing

### 9.1 Test entry points

The repository registers tests in two places:

- `app/SurgicalVision3D_Planner/CMakeLists.txt`
  - registers the main module script as a unittest entry
- `app/SurgicalVision3D_Planner/Testing/Python/CMakeLists.txt`
  - registers `SurgicalVision3D_PlannerPhase1Test.py`

### 9.2 What is covered

The test suite covers a meaningful amount of logic behavior, especially for a Slicer scripted module. Covered topics include:

- vector normalization and robust rotation matrices
- trajectory extraction from point pairs
- derived trajectory basis generation and spacing
- generated-node reuse policy
- repeated probe-line creation and merge behavior
- signed-margin summaries and threshold calculations
- structure-safety summaries
- no-touch and coordination logic
- endpoint auto-adjust determinism
- export bundle creation and sequence handling
- cohort definition loading and aggregation
- reproducibility package assembly and manifest generation
- parameter-node restoration
- beginner MAM summary behavior
- coaxial push-through offset behavior

### 9.3 What still needs manual validation

Automated tests do not replace runtime validation inside Slicer. Manual checks are still important for:

- UI ergonomics and state gating
- Segment Editor interactions
- visual display correctness of placed segmentations/models
- scene save/load behavior across Slicer versions
- actual clinical/research interpretation of geometric outputs

## 10. Current Boundaries and Limitations

This repository is capable, but it is still intentionally conservative.

- It is a research planning tool, not a real-time navigation system.
- Margin, safety, and validation are geometric; they are not thermal, perfusion, or physics simulations.
- Beginner validation assumes exactly one master trajectory.
- Some export/reproducibility hooks look for tables such as scenario, benchmark, report, or study outputs that may only exist when companion workflows populate them.
- Cohort execution is currently table-driven and summary-oriented, not raw-case replay from imaging assets.
- Probe placement assumes the reference geometry is oriented along local negative Z.
- Several workflows default to first-segment or preferred-name policies instead of rich explicit segment-picking UIs.

These are not accidents. Most of them are tradeoffs in favor of deterministic, reviewable behavior.

## 11. Recommended Reading Order

For a new developer or reviewer, the most useful reading order is:

1. `README.md`
2. `QUICK_START.md`
3. `QUICK_USAGE.md`
4. `INSTRUCTIONS_FOR_USE.md`
5. `SOP.md`
6. `app/SurgicalVision3D_Planner/PREREQUISITES.md`
7. `app/SurgicalVision3D_Planner/BEGINNER_STEP_BY_STEP.md`
8. `app/SurgicalVision3D_Planner/USAGE_GUIDE.md`
9. `app/SurgicalVision3D_Planner/SurgicalVision3D_Planner.py`
10. `app/SurgicalVision3D_Planner/Testing/Python/SurgicalVision3D_PlannerPhase1Test.py`
11. `app/SurgicalVision3D_Planner/PHASE1_REFACTOR_NOTES.md`
12. `app/SurgicalVision3D_Planner/MULTI_TRAJECTORY_ARRAY_SPEC.md`

That sequence moves from user-facing behavior to implementation detail to historical design context.
