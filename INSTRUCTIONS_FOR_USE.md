# SurgicalVision3D-Planner Instructions for Use

## 1. Purpose

This document provides detailed operating instructions for `SurgicalVision3D-Planner`, a proprietary 3D Slicer extension for research-oriented ablation planning. It is written for authorized users who need a structured guide for loading the module, preparing inputs, running the beginner workflow, using the broader planning tools, reviewing outputs, and exporting results.

This repository and software are restricted and proprietary. Repository access does not grant permission to use the software beyond the governing authorization and license terms in `LICENSE.md`.

## 2. Scope

These instructions cover:

- loading the extension from source into 3D Slicer
- preparing required imaging, segmentation, and markups inputs
- executing the beginner single-applicator workflow on `dev/beginner-module`
- using the general planning workflow for paired trajectories
- reviewing generated outputs, tables, and summaries
- saving and exporting planning results

These instructions do not certify the software for clinical use. The current module is a research planning tool and not a real-time navigation system, physics simulator, or validated treatment-delivery platform.

## 3. Intended user profile

The module is intended for authorized research, engineering, and technical users who:

- can operate 3D Slicer
- understand segmentation and markup placement concepts
- can review geometric planning outputs critically
- can distinguish software-generated planning suggestions from validated procedural decisions

## 4. Operational context and limitations

Users must understand the following before operation:

- the module is deterministic, but deterministic does not mean clinically validated
- margin and safety calculations are geometric approximations
- the beginner workflow assumes exactly one master trajectory
- probe placement assumes the source geometry points along local negative Z
- optional workflows may expose outputs only when the required upstream nodes or tables exist

If any output appears anatomically, geometrically, or procedurally inconsistent, stop and investigate the input data, point order, segmentation quality, and geometry orientation before proceeding.

## 5. Required software and environment

Minimum operational environment:

- a recent 3D Slicer build
- built-in Slicer modules `Segment Editor` and `Fiducial Registration`
- local access to this repository

Recommended operator setup:

- enough disk space for scenes, exported bundles, screenshots, and reproducibility artifacts
- a stable local working copy of the repository
- familiarity with Slicer segmentations, models, tables, and markups

See `app/SurgicalVision3D_Planner/PREREQUISITES.md` for the short prerequisite checklist.

## 6. Repository and module locations

The active extension project is under `app/`.

Key paths:

- `app/CMakeLists.txt`: extension entry point
- `app/SurgicalVision3D_Planner/SurgicalVision3D_Planner.py`: main module implementation
- `app/SurgicalVision3D_Planner/Resources/UI/SurgicalVision3D_Planner.ui`: base UI
- `app/SurgicalVision3D_Planner/Resources/Geometries/`: bundled STL templates
- `app/SurgicalVision3D_Planner/Resources/Cohorts/`: sample case and cohort resources
- `app/SurgicalVision3D_Planner/Testing/Python/`: runtime tests

## 7. Load the extension into Slicer

Perform this once per workstation or whenever the module path changes.

1. Open 3D Slicer.
2. Go to `Edit -> Application Settings -> Modules`.
3. Add `<repo>/app` to `Additional module paths`.
4. Confirm the setting.
5. Restart Slicer.
6. Open `SurgicalVision3D Planner` from the module selector.

Expected result:

- the module appears in the selector
- the module UI loads without import errors
- the beginner workflow sections are visible on `dev/beginner-module`

If the module does not appear:

- confirm the path points to the repository `app/` directory, not the repo root
- confirm the repository files are present
- restart Slicer after changing the path

## 8. Data requirements

### 8.1 Required input categories

For the beginner flow:

- one tumor segmentation
- one critical-structures segmentation
- one endpoints markups node with exactly two points
- one selected geometry from the beginner catalog

For the general planning flow:

- one reference probe segmentation
- one endpoints markups node with paired points
- one tumor segmentation

### 8.2 Supported case-folder content

The case import path expects a folder that may contain:

- one main `.nrrd` volume
- optional `.seg.nrrd` segmentations
- optional markups files such as `*.mrk.json` or `*.fcsv`

Import behavior is conservative:

- the largest non-segmentation `.nrrd` is used as the primary volume
- available segmentations are loaded
- available markups are loaded
- segmentation reference image geometry is aligned when possible

### 8.3 Markups point order

Point order is critical.

Beginner flow:

- point 1 = entry
- point 2 = endpoint

General flow:

- `entry1, endpoint1, entry2, endpoint2, ...`

If the order is wrong, the resulting trajectory direction, placement, validation, and coaxial outputs will also be wrong.

## 9. Pre-run checklist

Before starting a planning run, confirm all of the following:

- the module is loaded from the correct repository copy
- the intended case is loaded
- the visible volume and segmentations belong to the same case
- the tumor segmentation is present and not empty
- the critical structures segmentation is present if you are using the beginner validation workflow
- the endpoints markups node contains the correct number of points
- the operator understands whether they are using beginner mode or the general planning path

## 10. Beginner workflow: detailed operating instructions

The beginner flow is the primary guided workflow on `dev/beginner-module`.

### 10.1 Step 1: Import case

Purpose:

- load the case volume, available segmentations, and available markups

Operator actions:

1. Open the `Step 1 - Import Case` section.
2. Choose either a local case folder or the bundled sample case.
3. If using a local case folder, browse to the folder and click `Import`.
4. If using the bundled sample list, choose the case and click `Load`.

Expected outputs:

- one CT or other primary volume is visible
- relevant segmentations appear in the scene
- existing markups, if any, appear in the scene

Failure conditions and operator response:

- if no primary volume loads, verify the selected folder contains a valid non-segmentation `.nrrd`
- if segmentations are missing, confirm they exist in the folder and reload
- if scene loading is incomplete, try the folder-import path instead of relying on a saved `.mrml` scene

### 10.2 Step 2: Segment structures

Purpose:

- ensure the target and avoidance structures exist and are usable for validation and margin review

Operator actions:

1. Select the tumor segmentation node.
2. Select the critical structures segmentation node.
3. Open `Segment Editor`.
4. Review, create, or correct the necessary segments.
5. Return to the module after segmentation review.

Minimum quality expectation:

- the tumor segment is anatomically appropriate for the intended planning task
- each critical structure segment intended for avoidance is present and non-empty
- segmentation geometry is aligned to the case volume

Optional registration:

If registration is required for the case:

1. Select native fiducials.
2. Select registered fiducials.
3. Click `Register Tumor`.
4. If appropriate, click `Harden Tumor Transform`.

Stop conditions:

- do not proceed if the tumor segment is absent or obviously incorrect
- do not proceed if critical structures required for validation have not been segmented

### 10.3 Step 3: Define master trajectory

Purpose:

- define one candidate applicator path for the beginner workflow

Operator actions:

1. Select or create one endpoints markups node.
2. Place exactly two points.
3. Ensure the first point is the entry point.
4. Ensure the second point is the applicator endpoint.
5. Preview the trajectory if the preview action is available.

Expected result:

- one visible trajectory representation exists
- the planned direction runs from entry toward endpoint

Critical operator notes:

- do not place more than two points in beginner mode
- if the order is reversed, delete and replace the points
- do not assume imported markups already follow the correct order; verify them explicitly

### 10.4 Step 4: Validate trajectory against critical structures

Purpose:

- check whether the current master trajectory intersects or approaches critical structures in a disallowed way

Operator actions:

1. Click `Validate Against Critical Structures`.
2. Review the validation result text.
3. Review any validation tables generated in the scene.

Expected result:

- a clear pass or fail status
- a validation table such as `SV3D Master Trajectory Validation`

If validation passes:

- proceed to applicator planning

If validation fails:

1. Review the current trajectory visually.
2. Decide whether a manual point adjustment is appropriate.
3. Optionally run `Auto-adjust Endpoint`.
4. Re-run validation after any change.

### 10.5 Auto-adjust endpoint behavior

Purpose:

- attempt a conservative endpoint rescue after a failed beginner validation

What the tool does:

- keeps the entry point fixed
- keeps trajectory length fixed
- searches candidate endpoints within a bounded shell
- only accepts candidates that stay inside the tumor and avoid critical-structure intersection

Important operating expectations:

- this is a rescue heuristic, not a global optimizer
- if no safe candidate exists within the configured search envelope, the endpoint remains unchanged
- the operator must still review the updated trajectory before proceeding

### 10.6 Step 5: Single applicator plan

Purpose:

- place the selected geometry along the validated trajectory and evaluate margin performance

Operator actions:

1. Select a geometry from the catalog-backed list.
2. Keep planning mode aligned with the beginner workflow.
3. Click `Place Single Applicator`.
4. Click `Merge Translated Probes`.
5. Click `Evaluate MAM`.

Expected outputs:

- one placed applicator or probe segmentation
- `SV3D Combined Ablation Zone`
- margin model and margin summary tables
- MAM assessment result

Interpretation notes:

- MAM evaluation is based on signed-margin geometry
- default MAM threshold is `10 mm`
- beginner margin coloring is intended as a fast visual aid, not a substitute for table review

If MAM does not pass:

- do not lock the plan
- revise the trajectory or geometry choice
- re-run validation, placement, merge, and MAM evaluation

### 10.7 Step 6: Lock plan and compute coaxial guidance

Purpose:

- freeze the validated master plan and derive a coaxial navigation target

Operator actions:

1. Click `Lock Validated Master Plan`.
2. Select `PullBack` or `PushThrough`.
3. Set `Spare (mm)` if required.
4. Click `Compute Coaxial Plan`.

Expected outputs:

- `SV3D Locked Master Plan`
- `SV3D Coaxial Plan Summary`
- a generated coaxial navigation target or line

Important note:

- do not compute coaxial guidance on an unvalidated or unlocked plan
- if the master trajectory changes, reset the lock state and recompute from validation forward

### 10.8 Step 7: Save and export

Purpose:

- preserve the case state and generate a machine-readable export bundle

Recommended operator actions:

1. Save the Slicer scene.
2. Open the export section if you need a bundle.
3. Set the export base name and directory.
4. Choose include flags appropriate to the current run.
5. Click `Export Bundle`.

Expected outputs:

- a sequence-numbered export folder
- manifest JSON
- selected CSV tables
- plan summary JSON and related metadata

## 11. General planning workflow: detailed operating instructions

Use this workflow when planning with paired entry/endpoint trajectories and a reference probe segmentation instead of the beginner single-applicator path.

### 11.1 Select inputs

1. Select `Reference Probe Segmentation`.
2. Select `Endpoints Markups`.
3. Confirm the markups node contains paired points in the correct order.
4. Select the tumor segmentation.
5. Optionally select risk structures segmentation.

### 11.2 Place probes

1. Click `Place Probes`.
2. Optionally create trajectory lines.
3. Review all generated placed-probe nodes.

Expected result:

- one placed probe node per trajectory
- optional generated trajectory line nodes

### 11.3 Merge and evaluate

1. Click `Merge Translated Probes`.
2. Confirm `SV3D Combined Ablation Zone` exists.
3. Click `Evaluate Margins`.
4. Review the margin model, raw table, and summary tables.

### 11.4 Optional downstream workflows

After the primary placement and margin evaluation, operators may run:

- structure safety analysis
- probe coordination analysis
- derived trajectory generation
- cohort evaluation
- export bundle generation
- reproducibility package assembly

Use these only when the corresponding inputs and outputs are present.

## 12. Output review and interpretation

Minimum review set for a normal planning session:

- visible trajectory path
- visible placed applicator or combined ablation zone
- tumor segmentation overlay
- validation summary if using beginner mode
- `SV3D Plan Summary`
- `SV3D Margin Threshold Summary`

Review expectations:

- confirm the trajectory follows the intended anatomy
- confirm the placed geometry is aligned and not obviously rotated incorrectly
- confirm margin outcomes are interpreted from both visuals and tables
- confirm critical-structure interaction results are acceptable for the intended research analysis

## 13. Common output nodes and tables

Common named outputs include:

- `SV3D Combined Ablation Zone`
- `SV3D Signed Margin Model`
- `SV3D Signed Margin Table`
- `SV3D Plan Summary`
- `SV3D Margin Threshold Summary`
- `SV3D Master Trajectory Validation`
- `SV3D Locked Master Plan`
- `SV3D Coaxial Plan Summary`

These outputs are module-owned and are designed to be reused or regenerated deterministically.

## 14. Data saving and record retention

Recommended minimum artifacts to save for a completed run:

- the Slicer scene
- the final endpoints markups node
- the tumor and critical structures segmentations used during the run
- relevant summary tables
- the export bundle, if generated

For study-style review, also preserve:

- case identifier
- operator name or initials
- run date
- geometry preset used
- MAM threshold used
- coaxial technique selected
- any material deviations from the standard flow

## 15. Troubleshooting

### 15.1 Buttons remain disabled

Likely causes:

- required node not selected
- wrong node type selected
- markups count does not match the workflow
- lock-state or validation-state gating has not been satisfied

### 15.2 Probe placement looks wrong

Likely causes:

- entry/endpoint order reversed
- source geometry orientation inconsistent with the expected local negative Z convention
- mismatched case coordinate context between markups and segmentation nodes

### 15.3 Validation fails repeatedly

Likely causes:

- trajectory actually intersects a critical structure
- critical structures segmentation is too broad or misregistered
- endpoint remains outside an acceptable tumor position after manual or automatic rescue

### 15.4 Export bundle is incomplete

Likely causes:

- optional tables were never generated in the scene
- export include flags excluded them
- target output directory was not writable

## 16. Quality controls and operator checks

Before accepting a run as complete, verify:

- the correct case was used
- the correct tumor and critical structures segmentations were used
- the trajectory point order is correct
- the final plan was validated if you used beginner mode
- the final margin evaluation was reviewed
- the saved outputs correspond to the final accepted scene state

## 17. Stop conditions

Stop the workflow and resolve the issue before proceeding if:

- the wrong patient or case is loaded
- the tumor segmentation is missing, empty, or obviously incorrect
- the trajectory order is uncertain
- validation results conflict with the visual scene and cannot be explained
- placed geometry is visibly misaligned
- export or saved outputs do not match the accepted on-screen plan

## 18. Related documents

- `QUICK_START.md`
- `QUICK_USAGE.md`
- `DOCUMENTATION.md`
- `app/SurgicalVision3D_Planner/PREREQUISITES.md`
- `app/SurgicalVision3D_Planner/USAGE_GUIDE.md`
- `app/SurgicalVision3D_Planner/BEGINNER_STEP_BY_STEP.md`

