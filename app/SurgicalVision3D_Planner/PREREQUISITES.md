# SurgicalVision3D Planner Prerequisites

Use this checklist before running the `SurgicalVision3D Planner` module in 3D Slicer.

## 1) Software Prerequisites

- 3D Slicer (recent stable or preview build).
- A Slicer build that includes the built-in modules:
  - `Segment Editor`
  - `Fiducial Registration` (CLI module key: `fiducialregistration`)
- Python packages bundled with Slicer:
  - `numpy`
  - `vtk`

## 2) Install and Load This Module

1. Open 3D Slicer.
2. Go to `Edit -> Application Settings -> Modules`.
3. In `Additional module paths`, add:
   - `<repo>/SurgicalVision3D-Planner/app`
4. Click `OK`.
5. Restart Slicer.
6. Open `SurgicalVision3D Planner` from the module selector.

## 3) Verify Required Runtime Modules

Open Slicer Python Interactor and run:

```python
required = ["segmenteditor", "fiducialregistration"]
missing = [name for name in required if not hasattr(slicer.modules, name)]
print("Missing:", missing)
```

Expected output:

```text
Missing: []
```

If `fiducialregistration` is missing, install/use a Slicer build that includes the Registration CLI modules.

## 4) Required Scene Inputs (Nodes You Must Create)

Before clicking planner actions, prepare these nodes:

- `Reference Probe Segmentation` (`vtkMRMLSegmentationNode`)
  - Must contain at least one valid segment with closed-surface representation.
- `Endpoints Markups` (`vtkMRMLMarkupsFiducialNode`)
  - Control points are interpreted in pairs: `entry1,endpoint1,entry2,endpoint2,...`
  - Use an even number of points for strict pairing.
- `Tumor Segmentation` (`vtkMRMLSegmentationNode`)
  - Must contain at least one valid segment with closed-surface representation.

Optional inputs:

- `Risk Structures Segmentation` for structure-safety analysis.
- `Native Fiducials` + `Registered Fiducials` for tumor registration workflow.

## 5) Legacy Note: ModelToModelDistance

Current code computes signed distances internally with VTK and does **not** require the `ModelToModelDistance` extension.

If you are running an older commit that still depends on `ModelToModelDistance`, install that extension from:

- `View -> Extensions Manager` in Slicer, then search `ModelToModelDistance`.

