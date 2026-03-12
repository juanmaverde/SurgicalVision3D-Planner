# SurgicalVision3D-Planner Standard Operating Procedure

## 1. Document control

- Document title: `SurgicalVision3D-Planner Standard Operating Procedure`
- Repository: `SurgicalVision3D-Planner`
- Applicable branch profile: `dev/beginner-module`
- Document type: internal operating procedure
- Status: working SOP template for authorized research use

## 2. Purpose

This SOP defines a consistent operational procedure for using `SurgicalVision3D-Planner` in a controlled research setting. It is intended to reduce operator variability, improve record quality, and ensure that planning runs are performed, reviewed, saved, and exported in a repeatable way.

## 3. Scope

This SOP applies to:

- loading the module into 3D Slicer
- preparing case inputs
- executing the beginner workflow
- performing quality checks on generated outputs
- saving and exporting results
- documenting deviations and run outcomes

This SOP does not authorize clinical use and does not replace institutional approval, data-governance requirements, or specialist review.

## 4. Roles and responsibilities

### 4.1 Operator

The operator is responsible for:

- loading the correct repository and case
- selecting the correct nodes
- placing trajectory points in the correct order
- reviewing all validation and margin outputs
- saving the scene and exports
- documenting deviations from the SOP

### 4.2 Reviewer or supervisor

When a second reviewer is required, that reviewer is responsible for:

- confirming the case identity and data integrity
- confirming the accepted final trajectory and geometry selection
- confirming that the recorded outputs match the reviewed scene

## 5. Required materials

- authorized local clone of this repository
- recent 3D Slicer build
- workstation with access to the case data
- segmentation and markups inputs required for the selected workflow

## 6. Safety and use restrictions

- use only authorized research data
- do not represent the software as clinically validated
- do not proceed if input identity or segmentation integrity is uncertain
- do not use unattended or unexplained auto-generated outputs without review

## 7. Definitions

- `Entry point`: first point of the beginner trajectory pair
- `Endpoint`: second point of the beginner trajectory pair
- `MAM`: minimal ablative margin assessment
- `Locked plan`: accepted beginner master plan snapshot used for coaxial planning
- `Deviation`: any intentional or unavoidable departure from this SOP

## 8. Pre-run procedure

Perform the following before every session.

### 8.1 Environment verification

1. Open 3D Slicer.
2. Confirm that `<repo>/app` is configured as an additional module path.
3. Open `SurgicalVision3D Planner`.
4. Confirm the module loads without runtime errors.

Acceptance criteria:

- module is visible and opens successfully
- beginner workflow sections are present

### 8.2 Repository and working-copy verification

1. Confirm the operator is working from the intended repository path.
2. Confirm the branch or local state is the intended one for the session.
3. Confirm the repository contains the expected documentation and resources.

Acceptance criteria:

- the active working copy is the approved one for the session

### 8.3 Case verification

1. Confirm the correct case identifier before import or load.
2. Confirm the case folder or bundled sample selection.
3. Confirm the operator understands whether the run is a demo, test, or study run.

Acceptance criteria:

- case identity is confirmed before planning begins

## 9. Main procedure

### 9.1 Load case

1. Import the case folder or load the bundled sample.
2. Confirm the primary volume is present.
3. Confirm relevant segmentation and markups nodes are present, if expected.

Acceptance criteria:

- one usable case volume is loaded
- expected supporting nodes are visible in the scene

Stop condition:

- stop if the wrong case or an incomplete case loads

### 9.2 Prepare segmentations

1. Select the tumor segmentation node.
2. Select the critical structures segmentation node.
3. Open Segment Editor.
4. Review and correct the required segments.
5. Return to the module.

Acceptance criteria:

- tumor segment exists and is usable
- critical structures intended for validation exist and are usable

Stop condition:

- stop if segmentation quality is inadequate for the planning task

### 9.3 Define master trajectory

1. Select or create one endpoints markups node.
2. Place exactly two points in this order:
   - point 1 = entry
   - point 2 = endpoint
3. Preview the trajectory if available.

Acceptance criteria:

- exactly two points are present
- point order is confirmed
- one master trajectory is visible or otherwise confirmed

Stop condition:

- stop if the point order cannot be confirmed

### 9.4 Validate trajectory

1. Click `Validate Against Critical Structures`.
2. Review the validation status.
3. Review the validation table if generated.

Acceptance criteria:

- the accepted trajectory must pass validation before lock and coaxial steps

Failure response:

- manually adjust the trajectory or run `Auto-adjust Endpoint`
- revalidate after each change

Stop condition:

- stop if repeated rescue attempts do not produce a reviewable valid trajectory

### 9.5 Plan applicator and evaluate MAM

1. Choose the intended geometry preset.
2. Click `Place Single Applicator`.
3. Click `Merge Translated Probes`.
4. Click `Evaluate MAM`.
5. Review the combined ablation zone, margin model, and summary tables.

Acceptance criteria:

- geometry is visibly aligned with the intended trajectory
- `SV3D Combined Ablation Zone` exists
- MAM output is reviewed and recorded

Failure response:

- revise the trajectory or geometry choice
- repeat validation and MAM workflow as needed

Stop condition:

- stop if placement is visibly inconsistent or if margin outputs cannot be trusted

### 9.6 Lock plan and compute coaxial guidance

1. Click `Lock Validated Master Plan`.
2. Choose the coaxial technique.
3. Set spare distance if needed.
4. Click `Compute Coaxial Plan`.
5. Review the coaxial target and summary.

Acceptance criteria:

- lock succeeds only after validation and MAM review
- coaxial target is generated and reviewed

Stop condition:

- stop if the locked plan does not match the reviewed accepted trajectory

### 9.7 Save and export

1. Save the Slicer scene.
2. Configure export settings.
3. Run `Export Bundle` if required.
4. Confirm the export directory and sequence-numbered bundle.

Acceptance criteria:

- scene is saved
- export bundle exists if required
- manifest and summary files correspond to the accepted run

## 10. Required run record

For each accepted run, record at minimum:

- date
- operator name or initials
- case identifier
- repository path or branch context
- geometry used
- whether validation passed directly or after endpoint rescue
- MAM threshold used
- coaxial technique used
- export directory or bundle name
- deviations, if any

## 11. Deviation handling

A deviation occurs when:

- the normal step order is changed
- required inputs are substituted
- validation is bypassed or manually interpreted outside the normal gating logic
- exported outputs are incomplete but still retained for review

Deviation procedure:

1. Record the deviation.
2. Record the reason.
3. Record the effect on the run outcome.
4. Record who approved continuation, if applicable.

## 12. Quality review checklist

Before closing the session, confirm:

- the correct case was used
- the final accepted trajectory is the one shown in the scene
- the final outputs match the final accepted plan
- no earlier invalid outputs are being mistaken for the accepted ones
- saved or exported files are present and readable

## 13. Post-run procedure

1. Save the final scene if not already saved.
2. Save any required screenshots or notes.
3. Archive the export bundle if generated.
4. Log the run outcome in the project tracking system, if applicable.

## 14. Nonconformance triggers

Escalate or quarantine the run if:

- case identity is uncertain
- segmentation quality is unacceptable
- the accepted trajectory cannot be reconstructed from saved data
- output files are missing or mismatched
- results appear inconsistent with the visible scene and the cause cannot be explained

## 15. Revision expectations

This SOP should be revised whenever:

- workflow gating changes
- beginner workflow step names change
- export behavior changes materially
- new required inputs or mandatory checks are introduced

## 16. Related documents

- `INSTRUCTIONS_FOR_USE.md`
- `QUICK_START.md`
- `QUICK_USAGE.md`
- `DOCUMENTATION.md`
- `app/SurgicalVision3D_Planner/USAGE_GUIDE.md`
- `app/SurgicalVision3D_Planner/BEGINNER_STEP_BY_STEP.md`

