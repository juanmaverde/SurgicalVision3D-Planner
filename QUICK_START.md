# SurgicalVision3D-Planner Quick Start

This guide is the fastest path from clone to first successful run in 3D Slicer.

## 1. Prerequisites

- A recent 3D Slicer stable or preview build.
- The built-in Slicer modules `Segment Editor` and `Fiducial Registration`.
- A local clone of this repository.

If you need the fuller checklist, see [app/SurgicalVision3D_Planner/PREREQUISITES.md](app/SurgicalVision3D_Planner/PREREQUISITES.md).

## 2. Load the extension from source

1. Open 3D Slicer.
2. Go to `Edit -> Application Settings -> Modules`.
3. Add `<repo>/app` to `Additional module paths`.
4. Restart Slicer.
5. Open `SurgicalVision3D Planner` from the module selector.

## 3. Choose a first case

You have two easy starting points:

- `Bundled sample`: load the `CRLM-1001` demo case from the module UI.
- `Local case folder`: import a folder that contains one main `.nrrd` volume plus any available `.seg.nrrd` and `*.mrk.json` files.

## 4. First beginner run

1. Confirm or create the tumor segmentation.
2. Confirm or create the critical-structures segmentation.
3. Select or create one endpoint markups node.
4. Place exactly two points in this order:
   - point 1 = `entry`
   - point 2 = `endpoint`
5. Click `Validate Against Critical Structures`.
6. If validation fails, adjust the points or run `Auto-adjust Endpoint`.
7. Choose an applicator geometry.
8. Click `Place Single Applicator`.
9. Click `Merge Translated Probes`.
10. Click `Evaluate MAM`.
11. If the margin result is acceptable, click `Lock Validated Master Plan`.
12. Choose `PullBack` or `PushThrough`, then click `Compute Coaxial Plan`.

## 5. Expected outputs

After a successful first run you should see:

- a validated master trajectory
- a placed applicator segmentation
- `SV3D Combined Ablation Zone`
- `SV3D Signed Margin Model` and summary tables
- a locked-plan summary and coaxial target

## 6. Common blockers

- Disabled buttons usually mean a required node is missing.
- Beginner mode expects exactly two points, not a longer point list.
- Point order matters: `entry` first, `endpoint` second.
- Lock and coaxial actions stay gated until validation and MAM evaluation are complete.

For a slightly broader workflow summary, see [QUICK_USAGE.md](QUICK_USAGE.md).
