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
