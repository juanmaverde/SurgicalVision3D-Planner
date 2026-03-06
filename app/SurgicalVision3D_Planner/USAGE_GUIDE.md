# SurgicalVision3D Planner Usage Guide

This guide is written for both:

- non-expert users who need plain-language instructions
- technical users who need exact definitions and deterministic behavior notes

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
- module-owned output nodes/tables are reused when possible
- repeated runs with the same inputs produce stable outputs

## 2. Terminology (Non-Expert + Technical)

Use this glossary when reading the UI and outputs.

| Term | Non-expert explanation | Technical definition |
|---|---|---|
| Probe segmentation | The template shape of your treatment tool/zone. | A `vtkMRMLSegmentationNode` used as the source geometry for per-trajectory probe instance placement. |
| Segment | One labeled structure inside a segmentation. | A segmentation entry identified by a segment ID and name. |
| Tumor segmentation | The structure you want to treat. | The target segmentation node used for margin and coverage-related computations. |
| Risk structures segmentation | Nearby structures you want to avoid. | Optional segmentation node containing one or more structures-at-risk for distance checks. |
| Markups endpoint node | A list of points that define trajectories. | A `vtkMRMLMarkupsFiducialNode` whose control points are parsed in entry/target pairs. |
| Control point | A single planned point you place manually. | A fiducial point in markups with index and RAS coordinates. |
| Entry point | Where a probe enters tissue. | First point of each pair. |
| Target point | Where a probe aims inside/near the target. | Second point of each pair. |
| Trajectory | One planned path from entry to target. | Vector/path derived from one entry-target pair. |
| Trajectory line | A visible line for a trajectory. | Generated `vtkMRMLMarkupsLineNode` owned by the module. |
| Combined ablation zone | The union of all placed probe zones. | `SV3D Combined Ablation Zone` segmentation created from generated instances. |
| Registration | Aligning structures from one space to another. | Rigid transform computed from native and registered fiducials. |
| Harden transform | Make transform permanent in geometry. | Apply transform to node geometry and clear transform reference. |
| Signed margin | How far ablation is from tumor boundary (with sign). | Signed point-to-surface distance where negative indicates under-coverage/inside relation depending on convention. |
| Safety distance | How far ablation is from risk structures. | Distance metrics between combined ablation and each selected risk segment. |
| Threshold | A numeric cutoff to classify results. | Rule value used for recoloring, feasibility, or bucket summaries. |
| Feasibility | Whether a plan satisfies enabled rules. | Boolean gate result from enabled constraints. |
| No-touch check | A conservative geometric rule. | Optional rule: all entry points must remain outside the tumor segment. |
| Coordination | Probe-to-probe arrangement quality checks. | Pairwise spacing/angle/overlap rules plus plan-level gate summary. |
| Scenario | A stored planning state for comparison. | Snapshot record used in scenario/comparison/recommendation flows. |
| Cohort | Multiple cases analyzed together. | Batch definition with case members and execution configuration. |
| Study analytics | Grouped descriptive summaries over cohort outputs. | Second-layer aggregate analytics over deterministic case-level rows. |
| Deterministic | Same inputs produce same outputs. | Stable naming, ordering, and values under unchanged scene/settings. |
| Manifest | A file that lists what was exported. | Structured JSON metadata describing bundle contents and provenance. |
| Export bundle | Folder containing machine-readable outputs. | Deterministic package with JSON summaries, CSV tables, and manifest. |
| JSON | Human-readable structured data format. | Text-based key/value format for manifests and structured summaries. |
| CSV | Spreadsheet-friendly table text format. | Delimited row/column export for analysis tools. |
| Provenance | Traceability of source and settings. | IDs, names, modes, and metadata linking outputs to their origin. |

## 3. Before You Start

Required inputs:

- reference probe/applicator segmentation
- endpoint markups (entry/target pairs)
- tumor segmentation

Optional inputs:

- structures-at-risk segmentation
- fiducials for rigid registration

Important endpoint rule:

- endpoint markups must have an even number of points:
  - entry1, target1, entry2, target2, ...

## 4. Input Selection Guide (Detailed)

### 4.1 Probe segmentation (most misunderstood input)

What it is:

- a reusable geometry template for the applicator/ablation shape
- copied onto every trajectory during probe placement

What it is not:

- not the tumor segmentation
- not the final merged output
- not a per-case result table

Technical behavior:

- `Place Probes` duplicates this source segment for each trajectory
- `Merge Translated Probes` combines generated instances into `SV3D Combined Ablation Zone`

Orientation assumption:

- the source template is expected to point along local negative Z
- if orientation is inconsistent, placed probes may appear rotated/misaligned

Simple example:

- if you define 3 trajectories and your probe template has one segment, the module places 3 transformed instances from that one source segment

### 4.2 Trajectory endpoint markups

How to enter points:

- point 1 = entry for trajectory 1
- point 2 = target for trajectory 1
- point 3 = entry for trajectory 2
- point 4 = target for trajectory 2

Do not:

- leave odd number of points
- swap entry/target order for some pairs

### 4.3 Tumor segmentation

Used for:

- margin metrics
- no-touch checks (if enabled)
- target-context summaries

### 4.4 Risk structures segmentation (optional)

Used for:

- per-structure safety distance summaries
- threshold bucket safety summaries

If missing:

- planning still runs
- safety-specific tables are not generated for that run

## 5. Core Single-Case Workflow

### Step A: Inputs

In **Inputs**:

1. Select `Probe segmentation` (source template).
2. Select `Trajectory endpoint markups` (entry/target pairs).
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

### Step C: Registration (optional)

In **Tumor Registration**:

1. Select tumor and fiducials.
2. Click `Register Tumor`.
3. If desired, click `Harden Tumor Transform`.

### Step D: Margin and Safety

In **Ablation Margin Evaluation**:

1. Confirm tumor and combined ablation are selected.
2. Optionally select risk structures segmentation.
3. Click `Evaluate Margins`.

Optional recoloring:

- set low/mid/high threshold values
- click `Recolor Margins` or `Reset Margin Colors`

## 6. Probe Coordination Workflow

In the coordination block:

1. Set spacing/angle/overlap threshold values.
2. Enable or disable each rule.
3. Optionally enable no-touch check.
4. Click `Evaluate Probe Coordination`.

Expected outputs:

- `SV3D Probe Coordination Constraint Settings`
- `SV3D Probe Pair Coordination Summary`
- `SV3D Probe Coordination Summary`
- `SV3D NoTouch Summary` (if no-touch is enabled)

## 7. Cohort / Study Workflow

In **Cohort / Study Evaluation**:

1. Set `Cohort definition path` to a cohort JSON.
2. Choose execution mode.
3. Choose which metric groups to include.
4. Set max cases (`0` means all listed cases).
5. Click `Run Cohort Evaluation`.

Expected outputs:

- `SV3D Cohort Execution Summary`
- `SV3D Cohort Case Summary`
- `SV3D Cohort Aggregate Metrics`
- `SV3D Cohort Comparison Summary`

Included sample:

- `Resources/Cohorts/studies/example_cohort_v1.json`

## 8. Export Workflow

In **Export**:

1. Choose export mode.
2. Set base name and destination directory.
3. Choose include/exclude toggles.
4. Click `Export Bundle`.

Bundle behavior:

- output folder uses deterministic naming: `<base>_<sequence>`
- manifest + JSON summaries + selected CSV tables are written
- export does not mutate the current plan
- cohort tables are included when present

## 9. Output Table Reference (What each table means)

| Table name | Non-expert meaning | Technical meaning |
|---|---|---|
| `SV3D Trajectory Summary` | Basic details for each planned path. | Per-trajectory geometry metrics (entry/target/length/direction). |
| `SV3D Plan Summary` | Overall margin summary for the plan. | Aggregated signed-margin statistics. |
| `SV3D Margin Threshold Summary` | How much of margin falls into risk buckets. | Bucketed counts/percent by threshold cutoffs. |
| `SV3D Structure Safety Summary` | Distance from treatment zone to each risk structure. | Per-structure distance summary statistics. |
| `SV3D Structure Safety Threshold Summary` | Safety bucket rates per risk structure. | Threshold counts/percent by structure. |
| `SV3D Probe Pair Coordination Summary` | Pair-by-pair probe arrangement quality. | Pairwise spacing/angle/overlap rule results. |
| `SV3D Probe Coordination Summary` | Overall pass/fail of probe arrangement. | Plan-level aggregation of coordination constraints. |
| `SV3D NoTouch Summary` | Whether entry points stayed outside tumor. | Boolean no-touch result with failed indices. |
| `SV3D Cohort Case Summary` | Per-case results in a batch/study run. | Case-level execution status and selected metrics. |
| `SV3D Cohort Aggregate Metrics` | Study-level average/median style summary. | Deterministic aggregate stats over successful case rows. |
| `SV3D Cohort Comparison Summary` | Grouped case summary (for example by preset). | Group-wise descriptive comparison table. |
| `SV3D Export Summary` | Last export result overview. | Deterministic export status metadata. |
| `SV3D Export Manifest Preview` | What metadata went into the export manifest. | Key/value projection of manifest fields. |

## 10. Troubleshooting

### Buttons stay disabled

Check:

- missing required input nodes
- odd number of endpoint control points
- no generated/merged ablation when running margin evaluation
- wrong node type selected in probe selector

### Probe placement looks wrong

Check:

- source probe template orientation (expected local negative Z forward)
- entry/target ordering in markups
- coordinate consistency of markups and segmentations

### Cohort run fails

Check:

- cohort JSON path exists
- cohort JSON has valid `studyId` and `cases`
- referenced `scenarioId` values exist in scene tables
- expected source tables are present

### Export missing optional files

This is expected if source tables are absent.
Only available outputs are exported.

## 11. Practical Tips

- Save scene snapshots before major comparisons.
- Keep scenario IDs stable if cohort/study workflows depend on them.
- Use clear naming for markups/segmentations for auditability.
- Re-run unchanged workflows to verify reproducibility.
- When sharing results, include both bundle manifest and key source tables.
