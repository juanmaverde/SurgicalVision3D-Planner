# SurgicalVision3D-Planner Release Guide

## Scope

Use this checklist before cutting a release branch, publishing a build, or creating a git tag.

This repository currently has no established public version history. Choose the release version explicitly before tagging.

## Version and tag policy

- Use annotated tags, not lightweight tags.
- Prefer semantic-version style tags: `vX.Y.Z`.
- For pre-release drops, use `vX.Y.Z-rcN`.
- Do not tag from a dirty worktree.

Example:

```powershell
git tag -a v0.1.0 -m "SurgicalVision3D-Planner v0.1.0"
git push origin v0.1.0
```

## Release checklist

1. Verify repository docs are current:
   - `README.md`
   - `INSTRUCTIONS_FOR_USE.md`
   - `QUICK_START.md`
   - `QUICK_USAGE.md`
   - `SOP.md`
   - `DOCUMENTATION.md`
   - `LICENSE.md`
   - user-facing module docs under `app/SurgicalVision3D_Planner/`
2. Verify packaged resources are current in `app/SurgicalVision3D_Planner/CMakeLists.txt`:
   - `Resources/geometry_catalog.json`
   - all `Resources/Geometries/*.stl` templates
   - bundled `CRLM-1001` sample-case top-level assets
   - reproducibility schemas/templates
3. Run lightweight syntax validation:

```powershell
python -m py_compile app\SurgicalVision3D_Planner\SurgicalVision3D_Planner.py
python -m py_compile app\SurgicalVision3D_Planner\Testing\Python\SurgicalVision3D_PlannerPhase1Test.py
```

4. Run Slicer-side automated tests from a Slicer build or extension test environment.
5. Perform a manual smoke test in 3D Slicer:
   - load the module from `app/`
   - load the bundled `CRLM-1001` sample case
   - confirm reference geometries auto-load
   - place probes, merge, and evaluate margins
   - run beginner workflow validation and export one bundle
6. Confirm extension metadata in `app/CMakeLists.txt` is appropriate for the distribution target:
   - homepage
   - icon URL
   - screenshot URLs

## Tagging decision

Create the tag only after:

- the release commit exists
- the validation pass is complete
- the final version string is agreed

If those three conditions are not met, stop before tagging.
