cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(SurgicalVision3D_Planner)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SurgicalVision3D_Planner")
set(EXTENSION_CONTRIBUTORS "Juan Verde (Surgeon Scientist)")
set(EXTENSION_DESCRIPTION "Phase 1 prototype for segmentation-based ablation probe planning and tumor margin evaluation.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SurgicalVision3D_Planner.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SurgicalVision3D_Planner/Screenshots/1.png")
set(EXTENSION_DEPENDS "ModelToModelDistance") # Specified as a list or "NA" if no dependencies

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
