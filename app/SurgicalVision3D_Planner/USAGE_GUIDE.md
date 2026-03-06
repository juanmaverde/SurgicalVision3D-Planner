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

## 12. Full GUI Parameter Reference (All Bound Parameters)

This section documents every GUI-bound persisted parameter (`SlicerParameterName`) in the module.
These values are stored in the module parameter node and restored with the scene.

### 12.1 Inputs, registration, and margin-display parameters

| Parameter key | UI control | What it controls | Expected value / example |
|---|---|---|---|
| `referenceProbeSegmentation` | `probeSegmentationSelector` | Source probe/applicator template geometry used for placement. | Segmentation node with a valid closed-surface segment. |
| `endpointsMarkups` | `endpointsMarkupsSelector` | Entry/target fiducial list used to define trajectories. | Markups fiducial node with even number of points. |
| `createTrajectoryLinesOnPlacement` | `createTrajectoryLinesOnPlacementCheckBox` | Auto-create line markups after probe placement. | `true` or `false`. |
| `clearPreviousGeneratedProbes` | `clearPreviousGeneratedProbesCheckBox` | Clear prior generated probes/lines before new placement. | `true` or `false`. |
| `tumorSegmentation` | `tumorSegmentationSelector` | Target/tumor segmentation used in evaluation. | Segmentation node with at least one valid target segment. |
| `nativeFiducials` | `nativeFiducialsSelector` | Native-space fiducials for rigid registration. | Markups fiducial node. |
| `registeredFiducials` | `registeredFiducialsSelector` | Target-space fiducials for rigid registration. | Markups fiducial node. |
| `tumorTransform` | `tumorTransformSelector` | Optional explicit transform reference used by registration/harden steps. | Transform node. |
| `combinedProbeSegmentation` | `combinedProbeSegmentationSelector` | Output combined ablation segmentation reference. | Module-owned segmentation node (`SV3D Combined Ablation Zone`). |
| `outputMarginModel` | `outputMarginModelSelector` | Output signed-margin model reference. | Module-owned model node (`SV3D Signed Margin Model`). |
| `resultTable` | `resultTableSelector` | Output raw signed-margin table reference. | Module-owned table node (`SV3D Signed Margin Table`). |
| `riskStructuresSegmentation` | `riskStructuresSegmentationSelector` | Optional segmentation of structures-at-risk for safety metrics. | Segmentation node with one or more risk segments. |
| `recolorThresholdLow` | `recolorThresholdLowSpinBox` | Low threshold for margin recolor bucketization. | mm value, e.g. `-10.0`. |
| `recolorThresholdMid` | `recolorThresholdMidSpinBox` | Mid threshold for margin recolor bucketization. | mm value, e.g. `-5.0`. |
| `recolorThresholdHigh` | `recolorThresholdHighSpinBox` | High threshold for margin recolor bucketization. | mm value, e.g. `-2.0`. |

### 12.2 Probe coordination parameters

| Parameter key | UI control | What it controls | Expected value / example |
|---|---|---|---|
| `minInterProbeDistanceMm` | `minInterProbeDistanceSpinBox` | Minimum allowed distance between trajectory centerline segments. | mm value, e.g. `5.0`. |
| `maxInterProbeDistanceMm` | `maxInterProbeDistanceSpinBox` | Maximum allowed distance between trajectory centerline segments. | mm value, e.g. `120.0`. |
| `minEntryPointSpacingMm` | `minEntryPointSpacingSpinBox` | Minimum spacing between probe entry points. | mm value, e.g. `5.0`. |
| `minTargetPointSpacingMm` | `minTargetPointSpacingSpinBox` | Minimum spacing between probe target points. | mm value, e.g. `3.0`. |
| `maxParallelAngleDeg` | `maxParallelAngleSpinBox` | Angle threshold for near-parallel rule. | degree value, e.g. `10.0`. |
| `maxAllowedOverlapPercentBetweenPerProbeVolumes` | `maxOverlapRedundancySpinBox` | Conservative overlap redundancy cap between probe volumes. | percent value, e.g. `80.0`. |
| `requireAllProbePairsFeasible` | `requireAllProbePairsFeasibleCheckBox` | If enabled, any failed pair fails plan-level coordination gate. | `true` or `false`. |
| `enableNoTouchCheck` | `enableNoTouchCheckBox` | Enables no-touch entry rule (entry points outside tumor). | `true` or `false`. |
| `enableInterProbeDistanceRule` | `enableInterProbeDistanceRuleCheckBox` | Enables min/max inter-probe distance rule. | `true` or `false`. |
| `enableEntrySpacingRule` | `enableEntrySpacingRuleCheckBox` | Enables minimum entry-point spacing rule. | `true` or `false`. |
| `enableTargetSpacingRule` | `enableTargetSpacingRuleCheckBox` | Enables minimum target-point spacing rule. | `true` or `false`. |
| `enableAngleRule` | `enableAngleRuleCheckBox` | Enables near-parallel angle rule. | `true` or `false`. |
| `enableOverlapRule` | `enableOverlapRuleCheckBox` | Enables overlap-redundancy rule. | `true` or `false`. |

### 12.3 Cohort/study evaluation parameters

| Parameter key | UI control | What it controls | Expected value / example |
|---|---|---|---|
| `cohortStudyDefinitionPath` | `cohortStudyDefinitionPathLineEdit` | Path to cohort definition JSON. | Relative path like `Resources/Cohorts/studies/example_cohort_v1.json` or absolute path. |
| `cohortExecutionMode` | `cohortExecutionModeComboBox` | Source mode for case execution. | `ScenarioRegistry` or `CurrentWorkingPlan`. |
| `cohortIncludeMarginMetrics` | `cohortIncludeMarginMetricsCheckBox` | Include margin metrics in cohort output rows. | `true` or `false`. |
| `cohortIncludeSafetyMetrics` | `cohortIncludeSafetyMetricsCheckBox` | Include safety metrics in cohort outputs. | `true` or `false`. |
| `cohortIncludeCoverageMetrics` | `cohortIncludeCoverageMetricsCheckBox` | Include coverage metrics in cohort outputs. | `true` or `false`. |
| `cohortIncludeFeasibilityMetrics` | `cohortIncludeFeasibilityMetricsCheckBox` | Include feasibility/gate metrics in cohort outputs. | `true` or `false`. |
| `cohortIncludeCoordinationMetrics` | `cohortIncludeCoordinationMetricsCheckBox` | Include coordination/no-touch metrics in cohort outputs. | `true` or `false`. |
| `cohortIncludeVerificationMetrics` | `cohortIncludeVerificationMetricsCheckBox` | Include planned-vs-actual verification metrics when available. | `true` or `false`. |
| `cohortIncludeRecommendationMetrics` | `cohortIncludeRecommendationMetricsCheckBox` | Include recommendation/composite-score metrics. | `true` or `false`. |
| `cohortMaxCases` | `cohortMaxCasesSpinBox` | Maximum number of cases to execute from cohort list. | Integer (`0` means all). |

### 12.4 Reproducibility package parameters

| Parameter key | UI control | What it controls | Expected value / example |
|---|---|---|---|
| `packageMode` | `packageModeComboBox` | Reproducibility package scope profile. | `ReviewerSupplement`, `ValidationArchive`, `InternalHandoff`. |
| `includeBenchmarkArtifacts` | `includeBenchmarkArtifactsCheckBox` | Include benchmark definitions and benchmark runtime tables when present. | `true` or `false`. |
| `includeScenarioRegistry` | `includeScenarioRegistryCheckBox` | Include scenario registry/recommendation provenance artifacts. | `true` or `false`. |
| `includeCohortStudyArtifacts` | `includeCohortStudyArtifactsCheckBox` | Include cohort resources and cohort outputs. | `true` or `false`. |
| `includeStudyAnalytics` | `includeStudyAnalyticsCheckBox` | Include study-analytics tables if available in-scene. | `true` or `false`. |
| `includeReports` | `includeReportsCheckBox` | Include report-oriented outputs when available. | `true` or `false`. |
| `includeCanonicalJson` | `includeCanonicalJsonCheckBox` | Include canonical machine-readable JSON summaries. | `true` or `false`. |
| `includeValidationResults` | `includeValidationResultsCheckBox` | Include validation/benchmark result tables when available. | `true` or `false`. |
| `packageBaseName` | `packageBaseNameLineEdit` | Root package name before sequence suffix. | Text, e.g. `SV3D_ReproducibilityPackage`. |
| `packageOutputDirectory` | `packageOutputDirectoryLineEdit` | Output folder for reproducibility packages. | Empty for default temp path or absolute directory path. |

### 12.5 Export parameters

| Parameter key | UI control | What it controls | Expected value / example |
|---|---|---|---|
| `exportMode` | `exportModeComboBox` | Export context selection. | `CurrentWorkingPlan`, `SelectedScenario`, `CurrentRecommendationContext`. |
| `selectedExportScenarioID` | `selectedExportScenarioIDLineEdit` | Scenario ID used when scenario-specific export mode is selected. | Scenario ID string, e.g. `S002`. |
| `exportBaseName` | `exportBaseNameLineEdit` | Export bundle base folder name before sequence suffix. | Text, e.g. `SV3D_Export`. |
| `lastExportDirectory` | `exportDirectoryLineEdit` | Export root directory for bundles. | Empty for default temp path or absolute directory path. |
| `includeWorkingPlan` | `includeWorkingPlanCheckBox` | Include current working-plan tables/summary. | `true` or `false`. |
| `includeSelectedScenario` | `includeSelectedScenarioCheckBox` | Include selected scenario summary payload if available. | `true` or `false`. |
| `includeScenarioComparison` | `includeScenarioComparisonCheckBox` | Include scenario comparison/delta/frontier tables when present. | `true` or `false`. |
| `includeRecommendationOutputs` | `includeRecommendationOutputsCheckBox` | Include recommendation outputs when present. | `true` or `false`. |
| `includeTrajectoryTables` | `includeTrajectoryTablesCheckBox` | Include trajectory-related tables. | `true` or `false`. |
| `includeSafetyTables` | `includeSafetyTablesCheckBox` | Include structure safety tables. | `true` or `false`. |
| `includeCoverageTables` | `includeCoverageTablesCheckBox` | Include coverage-related tables. | `true` or `false`. |
| `includeFeasibilityTables` | `includeFeasibilityTablesCheckBox` | Include feasibility and gating tables. | `true` or `false`. |
| `includeCoordinationTables` | `includeCoordinationTablesCheckBox` | Include probe-coordination/no-touch tables. | `true` or `false`. |

## 13. Tooltip Coverage Guarantee

All GUI-bound parameters listed in Section 12 now have explicit explanatory tooltip text in the module widget setup.
Tooltips are assigned during module setup and remain active for normal runtime usage.
