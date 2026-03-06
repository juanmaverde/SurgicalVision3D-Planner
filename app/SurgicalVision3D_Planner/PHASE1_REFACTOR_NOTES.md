# SurgicalVision3D_Planner Phase 1 Refactor Notes

## Migrated from `Legacy/AblationPlanner`
- Endpoint-pair trajectory extraction from markups control points.
- Probe duplication and rigid placement along each trajectory.
- Optional trajectory line generation.
- Probe merge workflow into a single ablation-zone segmentation.
- Tumor rigid registration via `fiducialregistration` (native vs registered fiducials).
- Tumor/ablation signed margin evaluation via `ModelToModelDistance`.
- Margin recolor and reset behavior on the signed-distance model.

## Refactored for current module architecture
- Replaced threshold-template logic with planner-specific logic while keeping:
  - `ScriptedLoadableModule` structure
  - `parameterNodeWrapper` state model
  - GUI auto-binding through `connectGui` / `disconnectGui`
- Introduced explicit `ProbeTrajectory` dataclass for internal trajectory handling.
- Replaced legacy ad hoc node-reference usage with typed parameter-node fields.
- Replaced anonymous UI control names with workflow-specific widget/button names.
- Added dedicated Phase 1 tests under `Testing/Python` and enabled registration.

## Legacy issues fixed during migration
- Removed fragile manual parameter-node synchronization.
- Removed malformed node reference key usage (legacy typo around native fiducials key).
- Removed off-by-one loops in scalar-array recolor/reset flows.
- Hardened rotation matrix generation for parallel and anti-parallel vectors.
- Reworked cleanup paths so missing nodes/IDs are handled safely.
- Avoided cumulative/invalid observer patterns by using explicit button-driven updates.

## Deferred to Phase 2
- Full multi-segment policy controls (currently first segment is used for probe/tumor model conversion).
- Rich reporting UI for margin statistics (current output is model + raw field-data table).
- Advanced auto-update behavior tied to markup interaction events.
- Expanded integration tests with real segmentation assets in CI-safe packaging.

## Reference probe geometry assumption
- Placement assumes the source probe geometry is oriented along negative Z in local coordinates.
- The rigid placement transform aligns source direction `[0, 0, -1]` to each endpoint-derived trajectory direction,
  then translates to the trajectory entry point.

## Phase 1.1 hardening pass
- Added deterministic generated-node lifecycle rules:
  - generated probe/trajectory nodes are tracked and reconciled against scene state
  - module-owned combined segmentation, margin model, and result table are reused deterministically
  - stale module-owned outputs are cleared when upstream data changes
- Hardened repeated-run behavior for:
  - `Place Probes`
  - `Create Trajectory Lines`
  - `Merge Translated Probes`
  - `Evaluate Margins`
  - repeated recolor/reset cycles
- Added stronger guard clauses and explicit user-facing errors for missing/invalid inputs.
- Added centralized first-segment helper (`getWorkingSegmentID`) so first-segment policy is explicit and validated.
- Added deterministic naming for generated outputs (`SV3D ...`).
- Added temporary margin-input model cleanup so repeated margin evaluations do not leave orphan helper models.

## Remaining for Phase 2
- Optional explicit user segment selection UI beyond first-segment policy.
- Live drag interaction updates for probe placement.
- Higher-level planning analytics and richer margin reporting UI.

## Phase 2A planning metrics and plan summary
- Added a metrics layer on top of Phase 1 geometry outputs:
  - per-trajectory metrics computed from `ProbeTrajectory`
  - signed-margin distribution summary metrics
  - threshold-based signed-margin bucket summary
- Added deterministic module-owned summary table outputs:
  - `SV3D Trajectory Summary`
  - `SV3D Plan Summary`
  - `SV3D Margin Threshold Summary`
- Workflow integration points:
  - trajectory summary table is updated after `Place Probes`
  - plan and threshold summary tables are updated after `Evaluate Margins`
- Repeated-run behavior:
  - summary tables are created/reused via owned-output helpers
  - stale plan/threshold summaries are cleared when merged probe geometry changes
  - summary computation validates empty/invalid signed-margin arrays and fails explicitly

## Deferred after Phase 2A
- Advanced planning intelligence (optimization and auto-planning).
- Device-specific ablation libraries and physics-driven coverage modeling.
- Critical-structure constraints and risk-aware planning.
- Rich UI dashboard beyond table-node outputs.

## Phase 2B structures-at-risk and safety distances
- Added optional structures-at-risk input:
  - `riskStructuresSegmentation` parameter-node field
  - UI selector in Ablation Margin Evaluation section
- Added geometry-based safety evaluation against all valid risk-structure segments:
  - iterates every valid segment in selected risk segmentation (no first-segment shortcut)
  - computes signed closest-point distance against combined ablation zone
  - uses signed convention where negative values indicate overlap/collision
- Added deterministic module-owned safety tables:
  - `SV3D Structure Safety Summary`
  - `SV3D Structure Safety Threshold Summary`
- Added safety summary metrics per structure:
  - minimum, mean, median, P20, P80 distances (mm)
- Added threshold metrics per structure:
  - count/percent `< 0 mm`, `< 2 mm`, `< 5 mm`, `>= 5 mm`
- Added repeated-run safety handling:
  - stale safety tables are cleared when upstream geometry changes
  - stale safety tables are cleared when risk-structures selection changes
  - temporary safety-evaluation models are removed after each run

## Deferred after Phase 2B
- Automated plan optimization with safety constraints.
- Structure-priority weighting and clinical risk scoring.
- Thermal/dose/perfusion modeling beyond geometric distance fields.
- Collision-aware trajectory search and real-time drag updates.

## Phase 6B multi-probe coordination, spacing rules, and no-touch constraints
- Added explicit probe-coordination constraint settings in the parameter node:
  - inter-probe distance min/max
  - entry/target spacing minimums
  - maximum parallel-angle threshold
  - maximum overlap-redundancy threshold (conservative proxy)
  - toggles for each rule, plus `requireAllProbePairsFeasible` and `enableNoTouchCheck`
- Added deterministic pairwise probe-coordination evaluation based on trajectory centerlines:
  - `InterProbeDistanceMm` is defined as the minimum distance between two trajectory line segments
  - entry spacing, target spacing, and axis-angle metrics are computed per pair
  - pair rows are deterministically ordered by `(ProbeAIndex, ProbeBIndex)`
- Added plan-level aggregation outputs:
  - feasible/infeasible pair counts
  - aggregated failed rule names
  - `CoordinationGatePass` and `CoordinationFailureSummary` for gating integration
- Added conservative no-touch rule (optional):
  - no-touch passes only if all probe entry points are outside the tumor closed surface
- Added deterministic module-owned output tables:
  - `SV3D Probe Coordination Constraint Settings`
  - `SV3D Probe Pair Coordination Summary`
  - `SV3D Probe Coordination Summary`
  - `SV3D NoTouch Summary`
- Workflow integration:
  - explicit `Evaluate Probe Coordination` button added in the evaluation section
  - coordination outputs are also refreshed from margin evaluation when endpoint pairs are valid
  - stale coordination outputs are cleared on upstream geometry resets
- Candidate-gating hook integration:
  - coordination results publish `AllPairsFeasible`, `InfeasiblePairCount`, `NoTouchPass`, and `CoordinationGatePass`
  - downstream candidate-feasibility layers can consume these fields without a separate filtering subsystem

## Deferred after Phase 6B
- Exact per-probe volumetric overlap intersection (current overlap is a conservative distance/length proxy).
- Advanced no-touch strategy synthesis and trajectory auto-adjustment.
- Automatic multi-probe optimization and procedural sequencing.

## Phase 7A export package, reporting outputs, and reproducible plan bundle
- Added export configuration state to the parameter node:
  - export mode (`CurrentWorkingPlan`, `SelectedScenario`, `CurrentRecommendationContext`)
  - selected scenario ID for export context
  - export base name and last-used export directory/sequence
  - include/exclude flags for table groups (trajectory, safety, coverage, feasibility, coordination, scenario/recommendation)
- Added deterministic export engine helpers:
  - deterministic bundle path builder using `<base>_<sequence>`
  - table-node-to-dictionary serializer
  - table-to-CSV export helper
  - structured JSON export helper
  - manifest builder and bundle export orchestrator
- Added deterministic export bundle structure:
  - `manifest.json`
  - `plan_summary.json`
  - `scenario_summary.json` when selected-scenario export is requested
  - `tables/*.csv` for available summary tables
  - `provenance/scenario_registry.json` and `provenance/recommendation_summary.json` when available
- Added deterministic module-owned export preview outputs:
  - `SV3D Export Summary`
  - `SV3D Export Manifest Preview`
- Added minimal export UI:
  - export mode selector
  - selected scenario ID field
  - export base name and export directory fields
  - include/exclude checkboxes for export categories
  - `Export Bundle` button and status label
- Export behavior is non-mutating:
  - no geometry or plan-state mutation occurs during export
  - optional outputs are exported only when source nodes exist
  - repeated exports use incrementing sequence folders to avoid silent overwrite

## Deferred after Phase 7A
- Rich document reporting (PDF/slide generation).
- External upload/sync or database export integrations.
- Styled human-facing report composition beyond structured JSON/CSV bundles.
