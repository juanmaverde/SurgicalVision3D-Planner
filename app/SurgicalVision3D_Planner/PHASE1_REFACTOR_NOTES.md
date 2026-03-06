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
