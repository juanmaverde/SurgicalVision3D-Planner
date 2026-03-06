# SurgicalVision3D Planner Usage Guide

This guide explains the day-to-day workflow for using the module inside 3D Slicer.

## 1. What This Module Does

SurgicalVision3D Planner supports:

- trajectory-based applicator planning
- ablation-zone geometry generation
- signed-margin and safety evaluation
- probe coordination checks
- cohort/study batch summaries
- deterministic export bundles (JSON + CSV + manifest)

The module is deterministic and conservative:

- no silent automatic plan overwrite
- owned output nodes/tables are reused when possible
- repeated runs with the same inputs produce stable outputs

## 2. Before You Start

Prepare these inputs:

- reference probe/applicator segmentation
- endpoint markups (entry/target pairs)
- tumor segmentation

Optional:

- structures-at-risk segmentation
- fiducials for rigid registration

Important endpoint rule:

- endpoint markups must have an even number of points:
  - entry1, target1, entry2, target2, ...

### 2.1 What each required input means

#### `Probe segmentation` (very important)

This is **not** the tumor and not the final merged ablation.  
It is the **reference geometry template** that the module duplicates and places on each trajectory.

Practical meaning:

- one segmentation node that contains the probe/ablation template geometry
- typically a single segment used as the source shape
- this source shape is copied, rotated, and translated for each entry-target pair

How the module uses it:

- during `Place Probes`, the module creates per-trajectory probe instances from this template
- during `Merge Translated Probes`, those instances are combined into `SV3D Combined Ablation Zone`

Geometry assumption:

- the reference probe template is assumed to be oriented along **negative Z** in local coordinates
- if your template orientation is inconsistent, placements can appear misaligned

Good sanity check:

- run with one simple straight trajectory first
- confirm placed probe orientation/direction matches the expected entry-to-target direction

#### `Trajectory endpoint markups`

This node defines planned trajectories as ordered point pairs:

- point 1 = entry for trajectory 1
- point 2 = target for trajectory 1
- point 3 = entry for trajectory 2
- point 4 = target for trajectory 2
- etc.

Rules:

- must be even number of points
- pair ordering matters
- do not mix entry/target order within a pair

If odd count is present, trajectory-based actions are blocked or fail with a guard error.

#### `Tumor segmentation`

This is the planning target used for:

- signed-margin evaluation
- safety/coverage context
- optional no-touch checks (entry points outside tumor)

Use the clinically relevant target segment and keep segment IDs/names stable if you plan cohort/export workflows.

#### Optional `Risk structures segmentation`

If provided, the module computes distance-based safety summaries between the ablation zone and each valid structure segment.

If omitted:

- main planning still works
- structure safety tables are simply not generated for that run

## 3. Core Single-Case Workflow

### Step A: Inputs

In **Inputs**:

1. Select `Probe segmentation`:
   - choose the reference probe/ablation template segmentation
   - this is the geometry that gets copied onto each trajectory
2. Select `Trajectory endpoint markups`.
3. Optionally enable:
   - `Create trajectory lines on placement`
   - `Clear previous generated probes`

### Step B: Place and Merge

1. Click `Place Probes`.
2. Optionally click `Create Trajectory Lines`.
3. Click `Merge Translated Probes`.

Expected outputs:

- generated probe instances
- optional line markups
- `SV3D Combined Ablation Zone`

### Step C: Registration (Optional)

In **Tumor Registration**:

1. Select tumor and fiducials.
2. Click `Register Tumor`.
3. If desired, click `Harden Tumor Transform`.

### Step D: Margin and Safety Evaluation

In **Ablation Margin Evaluation**:

1. Confirm tumor and combined ablation are selected.
2. Optionally select structures-at-risk segmentation.
3. Click `Evaluate Margins`.

Optional recoloring:

- adjust low/mid/high thresholds
- click `Recolor Margins` or `Reset Margin Colors`

Expected summary tables include:

- `SV3D Trajectory Summary`
- `SV3D Plan Summary`
- `SV3D Margin Threshold Summary`
- `SV3D Structure Safety Summary` (if risk structures selected)
- `SV3D Structure Safety Threshold Summary` (if risk structures selected)

## 4. Probe Coordination Workflow

In the coordination block (under evaluation):

1. Set spacing/angle/overlap thresholds.
2. Enable or disable individual rules.
3. Optionally enable no-touch check.
4. Click `Evaluate Probe Coordination`.

Expected outputs:

- `SV3D Probe Coordination Constraint Settings`
- `SV3D Probe Pair Coordination Summary`
- `SV3D Probe Coordination Summary`
- `SV3D NoTouch Summary` (when no-touch is checked)

## 5. Cohort / Study Evaluation Workflow

In **Cohort / Study Evaluation**:

1. Set `Cohort definition path` (JSON file).
2. Choose execution mode.
3. Choose metric groups to include.
4. Set max cases (`0` = all).
5. Click `Run Cohort Evaluation`.

Expected outputs:

- `SV3D Cohort Execution Summary`
- `SV3D Cohort Case Summary`
- `SV3D Cohort Aggregate Metrics`
- `SV3D Cohort Comparison Summary`

Included sample definition:

- `Resources/Cohorts/studies/example_cohort_v1.json`

## 6. Export Workflow

In **Export**:

1. Choose export mode.
2. Set base name and destination directory.
3. Set include/exclude toggles.
4. Click `Export Bundle`.

Bundle behavior:

- output folder name is deterministic: `<base>_<sequence>`
- manifest + JSON summaries + selected CSV tables are written
- export does not mutate the current plan

When present, cohort outputs are also exported.

## 7. Troubleshooting

### Buttons stay disabled

Check required upstream inputs:

- no endpoints => no placement
- odd endpoint count => no valid pair extraction
- no merged/generated ablation + no tumor => no margin evaluation
- wrong node type in `Probe segmentation` selector => no placement

### Probe placement looks wrong

Check:

- reference probe template orientation (expected local negative Z forward direction)
- endpoint ordering (entry/target must be paired correctly)
- coordinate conventions of your markups and segmentation

### Cohort run fails

Check:

- cohort JSON path is valid
- referenced scenario IDs exist in current scene tables
- expected source tables are present

### Export missing optional files

Optional CSVs are exported only when source tables exist.
This is expected behavior.

## 8. Practical Tips

- Save scene snapshots before major comparisons.
- Keep scenario IDs stable for cohort/benchmark references.
- Use deterministic naming for markups/segmentations to simplify QA.
- Re-run the same workflow with unchanged inputs to verify reproducibility.
