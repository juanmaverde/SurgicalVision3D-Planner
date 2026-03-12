# SurgicalVision3D-Planner Quick Usage

This guide is a short operational reference once the module is already loaded in Slicer.

## Required inputs

- `Tumor Segmentation`: target structure used for margin and coverage evaluation.
- `Endpoints Markups`: planned points used to define trajectories.
- `Reference Probe Segmentation` for the general workflow, or a geometry choice from the beginner catalog.

Optional inputs:

- `Critical Structures Segmentation` for beginner validation.
- `Risk Structures Segmentation` for safety-distance summaries.
- `Native Fiducials` and `Registered Fiducials` for rigid registration.

## Beginner workflow

Use this path on `dev/beginner-module` for a single applicator:

1. Import a case or load the bundled sample.
2. Open Segment Editor and prepare tumor plus critical structures.
3. Place exactly two points in `entry -> endpoint` order.
4. Run `Validate Against Critical Structures`.
5. If needed, run `Auto-adjust Endpoint`.
6. Choose a geometry and click `Place Single Applicator`.
7. Click `Merge Translated Probes`.
8. Click `Evaluate MAM`.
9. If the plan passes, lock it and compute the coaxial plan.

## General planning workflow

Use this when working with paired point trajectories and a reference probe template:

1. Select `Reference Probe Segmentation`.
2. Select `Endpoints Markups` with an even number of control points:
   - `entry1, endpoint1, entry2, endpoint2, ...`
3. Click `Place Probes`.
4. Optionally create trajectory lines.
5. Click `Merge Translated Probes`.
6. Click `Evaluate Margins`.
7. Optionally run structure-safety, coordination, cohort, export, or reproducibility workflows.

## Common outputs

- `SV3D Combined Ablation Zone`: merged probe result.
- `SV3D Signed Margin Model` and `SV3D Signed Margin Table`: signed margin outputs.
- `SV3D Plan Summary`: aggregate margin summary.
- `SV3D Master Trajectory Validation`: beginner validation output.
- `SV3D Coaxial Plan Summary`: coaxial guidance result after locking.

## Operating notes

- The module is designed to be deterministic: repeated runs with unchanged inputs should reuse module-owned outputs when possible.
- Probe placement assumes the reference geometry points along local negative Z.
- The tool is research-oriented. It is not a real-time navigation system and not a physics-validated simulator.

## If results look wrong

- Re-check point order in the markups node.
- Re-check segmentation selection and segment content.
- Re-check the reference probe orientation.
- Re-run the workflow from validation forward after changing trajectory points.

For full parameter and workflow details, see [app/SurgicalVision3D_Planner/USAGE_GUIDE.md](app/SurgicalVision3D_Planner/USAGE_GUIDE.md).
