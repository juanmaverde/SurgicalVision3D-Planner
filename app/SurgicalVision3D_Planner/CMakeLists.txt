#-----------------------------------------------------------------------------
set(MODULE_NAME SurgicalVision3D_Planner)

file(GLOB MODULE_GEOMETRY_RESOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  Resources/Geometries/*.stl
  )

file(GLOB MODULE_SAMPLE_CASE_RESOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  Resources/Cohorts/CRLM-1001/*.csv
  Resources/Cohorts/CRLM-1001/*.mrk.json
  Resources/Cohorts/CRLM-1001/*.mrml
  Resources/Cohorts/CRLM-1001/*.nrrd
  Resources/Cohorts/CRLM-1001/*.png
  )

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  BEGINNER_STEP_BY_STEP.md
  MULTI_TRAJECTORY_ARRAY_SPEC.md
  PHASE1_REFACTOR_NOTES.md
  PREREQUISITES.md
  USAGE_GUIDE.md
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/geometry_catalog.json
  Resources/Cohorts/cohort_schema_v1.json
  Resources/Cohorts/cohort_catalog_v1.json
  Resources/Cohorts/studies/example_cohort_v1.json
  Resources/Reproducibility/reproducibility_package_schema_v1.json
  Resources/Reproducibility/reproducibility_package_layout_v1.json
  Resources/Reproducibility/example_reviewer_package_config_v1.json
  Resources/Reproducibility/README_package_template.md
  ${MODULE_GEOMETRY_RESOURCES}
  ${MODULE_SAMPLE_CASE_RESOURCES}
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
