cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(SurgicalVision3D_Planner)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://surgicalvision3d.com/en")
set(EXTENSION_CONTRIBUTORS "Juan Verde (Surgeon Scientist)")
set(EXTENSION_DESCRIPTION "Research-oriented 3D Slicer extension for ablation planning, beginner-guided master-trajectory workflows, geometric margin/safety analysis, and deterministic export/reproducibility packages.")
# Replace these with public image URLs before publishing through the Slicer Extensions Index.
set(EXTENSION_ICONURL "https://surgicalvision3d.com/en")
set(EXTENSION_SCREENSHOTURLS "https://surgicalvision3d.com/en")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SurgicalVision3D_Planner)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
