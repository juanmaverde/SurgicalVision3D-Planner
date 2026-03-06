#-----------------------------------------------------------------------------
set(MODULE_NAME SurgicalVision3D_Planner)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/Cohorts/cohort_schema_v1.json
  Resources/Cohorts/cohort_catalog_v1.json
  Resources/Cohorts/studies/example_cohort_v1.json
  Resources/Reproducibility/reproducibility_package_schema_v1.json
  Resources/Reproducibility/reproducibility_package_layout_v1.json
  Resources/Reproducibility/example_reviewer_package_config_v1.json
  Resources/Reproducibility/README_package_template.md
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
