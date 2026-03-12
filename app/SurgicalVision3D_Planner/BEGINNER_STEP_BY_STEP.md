# SurgicalVision3D Planner Beginner Step-by-Step (Simple Guide)

This is the short, plain-language version.

Use this when you want the exact click order without technical details.

## Before You Start

1. Open **3D Slicer**.
2. Load the `SurgicalVision3D Planner` module.
3. Make sure you can see the beginner workflow sections:
   1. `Step 1 - Import Case`
   2. `Step 2 - Segment Structures`
   3. `Step 3 - Define Master Trajectory`
   4. `Step 4 - Validate Trajectory`
   5. `Step 5 - Applicator Plan`
   6. `Step 6 - Lock + Coaxial Plan`

## Step 1 - Import Case

Choose one method:

1. `Case folder` method:
   1. Click `Browse`.
   2. Pick a folder that has one main `.nrrd` scan file.
   3. Click `Import`.
2. `Bundled sample` method:
   1. Pick a case in `Bundled sample`.
   2. Click `Load`.

Goal after Step 1:

1. You should have a CT volume loaded.
2. You should have tumor/critical structure nodes available (or ready to create in Step 2).

## Step 2 - Segment Structures

1. In `Tumor segmentation`, select the tumor segmentation node.
2. In `Critical structures`, select the structures-to-avoid segmentation node.
3. Click `Open Segment Editor`.
4. In Segment Editor:
   1. Create or fix tumor segment.
   2. Create or fix critical structure segments.
5. Return to this module.

Optional registration (only if needed):

1. Fill `Native fiducials` and `Registered fiducials` (at least 3 matching points each).
2. Click `Register Tumor`.
3. Click `Harden Tumor Transform` if you want to bake transform.

Goal after Step 2:

1. Tumor and critical structures are correctly segmented.

## Step 3 - Define Master Trajectory

1. In `Endpoint markups`, select or create a markups node.
2. Place exactly **2 points** in this exact order:
   1. Point 1 = **entry point** (skin/liver entry)
   2. Point 2 = **applicator endpoint** (target area)
3. Click `Preview Master Trajectory` (or `Preview Planned Trajectories` if you changed mode).

Important:

1. Beginner mode expects exactly 2 points.
2. If order is wrong, delete points and place again in the correct order.

Goal after Step 3:

1. One valid master trajectory line is visible.

## Step 4 - Validate Trajectory

1. Click `Validate Against Critical Structures`.
2. Read the status text:
   1. `Trajectory valid` means continue.
   2. `Trajectory invalid` means move your 2 points in Step 3 or run `Auto-adjust Endpoint`, then validate again.

Goal after Step 4:

1. Validation must pass before lock/coaxial workflow.

## Step 5 - Applicator Plan

1. In `Geometry`, choose one ablation geometry.
2. Keep `Planning mode` as `Single trajectory` for beginner workflow.
3. Click `Place Single Applicator`.
4. Click `Merge Translated Probes`.
5. Click `Evaluate MAM`.
6. Read MAM status:
   1. If not satisfied, adjust trajectory in Step 3 and repeat from Step 4.
   2. If satisfied, continue to Step 6.

Goal after Step 5:

1. You have a combined ablation zone.
2. MAM pass is achieved.

## Step 6 - Lock + Coaxial Plan

1. Click `Lock Validated Master Plan`.
2. Choose `Coaxial technique`:
   1. `PullBack` or
   2. `PushThrough`
3. Set `Spare (mm)` if needed (default is fine for first use).
4. Click `Compute Coaxial Plan`.

Goal after Step 6:

1. Coaxial target is generated.
2. Plan is locked for execution guidance.

## If Something Is Blocked

If a button is disabled, check these in order:

1. Tumor segmentation selected.
2. Critical structures segmentation selected.
3. Exactly 2 endpoint points placed.
4. Step 4 validation passed.
5. Step 5 MAM evaluated and passed (for lock step).
6. Plan is not already locked when trying to edit trajectory.

## Fast Reset

If the workflow feels stuck:

1. In Step 6, click `Reset Lock`.
2. Go back to Step 3 and redefine the 2 points.
3. Repeat Step 4, Step 5, Step 6 in order.

## One-Line Checklist

Import -> Segment -> 2 points (entry then endpoint) -> Validate pass -> Place -> Merge -> Evaluate MAM pass -> Lock -> Compute Coaxial Plan.
