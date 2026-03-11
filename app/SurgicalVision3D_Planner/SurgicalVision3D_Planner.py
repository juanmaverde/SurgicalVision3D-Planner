from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import subprocess
import shutil
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Sequence
from urllib.parse import unquote
from xml.etree import ElementTree as ET

import numpy as np
import vtk
from vtk.util import numpy_support

import ctk
import qt
import slicer
from slicer import (
    vtkMRMLMarkupsFiducialNode,
    vtkMRMLMarkupsLineNode,
    vtkMRMLModelNode,
    vtkMRMLSegmentationNode,
    vtkMRMLTableNode,
    vtkMRMLTransformNode,
)
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.parameterNodeWrapper import Choice, parameterNodeWrapper
from slicer.util import VTKObservationMixin


REFERENCE_PROBE_DIRECTION_RAS = np.array([0.0, 0.0, -1.0], dtype=float)
GENERATED_PROBE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedProbe"
GENERATED_TRAJECTORY_LINE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedTrajectoryLine"
GENERATED_COMBINED_PROBE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedCombinedProbe"
GENERATED_MARGIN_MODEL_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedMarginModel"
GENERATED_RESULT_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedResultTable"
GENERATED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedTrajectorySummaryTable"
GENERATED_PLAN_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedPlanSummaryTable"
GENERATED_MARGIN_THRESHOLD_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedMarginThresholdSummaryTable"
GENERATED_STRUCTURE_SAFETY_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedStructureSafetySummaryTable"
GENERATED_STRUCTURE_SAFETY_THRESHOLD_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedStructureSafetyThresholdSummaryTable"
GENERATED_PROBE_COORDINATION_SETTINGS_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedProbeCoordinationSettingsTable"
GENERATED_PROBE_PAIR_COORDINATION_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedProbePairCoordinationTable"
GENERATED_PROBE_COORDINATION_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedProbeCoordinationSummaryTable"
GENERATED_NO_TOUCH_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedNoTouchSummaryTable"
GENERATED_DERIVED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedDerivedTrajectoryBundleSummaryTable"
GENERATED_EXPORT_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedExportSummaryTable"
GENERATED_EXPORT_MANIFEST_PREVIEW_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedExportManifestPreviewTable"
GENERATED_COHORT_EXECUTION_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedCohortExecutionSummaryTable"
GENERATED_COHORT_CASE_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedCohortCaseSummaryTable"
GENERATED_COHORT_AGGREGATE_METRICS_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedCohortAggregateMetricsTable"
GENERATED_COHORT_COMPARISON_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedCohortComparisonSummaryTable"
GENERATED_REPRODUCIBILITY_PACKAGE_SUMMARY_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedReproducibilityPackageSummaryTable"
GENERATED_REPRODUCIBILITY_MANIFEST_PREVIEW_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedReproducibilityManifestPreviewTable"
GENERATED_REPRODUCIBILITY_ARTIFACT_INDEX_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedReproducibilityArtifactIndexTable"
GENERATED_MASTER_TRAJECTORY_VALIDATION_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedMasterTrajectoryValidationTable"
GENERATED_MASTER_PLAN_SNAPSHOT_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedMasterPlanSnapshotTable"
GENERATED_COAXIAL_PLAN_TABLE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedCoaxialPlanTable"
GENERATED_COAXIAL_TARGET_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedCoaxialNavigationTarget"
GENERATED_COAXIAL_LINE_ATTRIBUTE = "SurgicalVision3D_Planner.GeneratedCoaxialNavigationLine"
REFERENCE_PROBE_TEMPLATE_ATTRIBUTE = "SurgicalVision3D_Planner.ReferenceProbeTemplate"
REFERENCE_PROBE_TEMPLATE_SOURCE_PATH_ATTRIBUTE = "SurgicalVision3D_Planner.ReferenceProbeTemplateSourcePath"
REFERENCE_PROBE_TEMPLATE_AXIAL_PLACEMENT_OFFSET_MM_ATTRIBUTE = "SurgicalVision3D_Planner.ReferenceProbeTemplateAxialPlacementOffsetMm"
DEFAULT_TUMOR_SEGMENTATION_NAMES = ("Tumor",)
DEFAULT_TUMOR_SEGMENT_NAMES = ("Tumor", "tumor", "Target", "target", "Lesion", "lesion")
DEFAULT_ENDPOINTS_MARKUPS_NAMES = ("endpoints", "Endpoints")
DEFAULT_NATIVE_FIDUCIAL_NAMES = ("Native fiducials", "Native Fiducials", "NativeFiducials")
DEFAULT_REGISTERED_FIDUCIAL_NAMES = ("Registered fiducials", "Registered Fiducials", "RegisteredFiducials")
DEFAULT_CRITICAL_STRUCTURES_SEGMENTATION_NAMES = (
    "Critical Structures",
    "CriticalStructures",
    "Risk Structures",
    "RiskStructures",
    "Structures At Risk",
    "StructuresAtRisk",
)
TEMP_PROBE_MARGIN_INPUT_ATTRIBUTE = "SurgicalVision3D_Planner.TempProbeMarginInput"
TEMP_TUMOR_MARGIN_INPUT_ATTRIBUTE = "SurgicalVision3D_Planner.TempTumorMarginInput"
TEMP_PROBE_SAFETY_INPUT_ATTRIBUTE = "SurgicalVision3D_Planner.TempProbeSafetyInput"
TEMP_STRUCTURE_SAFETY_INPUT_ATTRIBUTE = "SurgicalVision3D_Planner.TempStructureSafetyInput"
TEMP_STRUCTURE_SAFETY_DISTANCE_OUTPUT_ATTRIBUTE = "SurgicalVision3D_Planner.TempStructureSafetyDistanceOutput"
SIGNED_DISTANCE_ARRAY_NAME = "Signed"
SIGNED_DISTANCE_BACKUP_ARRAY_NAME = "SignedOriginal"
DEFAULT_MARGIN_COLOR_NODE_ID = "vtkMRMLColorTableNode2"
COMBINED_PROBE_NODE_NAME = "SV3D Combined Ablation Zone"
MARGIN_MODEL_NODE_NAME = "SV3D Signed Margin Model"
MARGIN_TABLE_NODE_NAME = "SV3D Signed Margin Table"
TEMP_PROBE_MODEL_NODE_NAME = "SV3D Temp Probe Margin Input"
TEMP_TUMOR_MODEL_NODE_NAME = "SV3D Temp Tumor Margin Input"
TRAJECTORY_SUMMARY_TABLE_NODE_NAME = "SV3D Trajectory Summary"
DERIVED_TRAJECTORY_SUMMARY_TABLE_NODE_NAME = "SV3D Derived Trajectory Bundle Summary"
PLAN_SUMMARY_TABLE_NODE_NAME = "SV3D Plan Summary"
MARGIN_THRESHOLD_SUMMARY_TABLE_NODE_NAME = "SV3D Margin Threshold Summary"
STRUCTURE_SAFETY_SUMMARY_TABLE_NODE_NAME = "SV3D Structure Safety Summary"
STRUCTURE_SAFETY_THRESHOLD_SUMMARY_TABLE_NODE_NAME = "SV3D Structure Safety Threshold Summary"
PROBE_COORDINATION_SETTINGS_TABLE_NODE_NAME = "SV3D Probe Coordination Constraint Settings"
PROBE_PAIR_COORDINATION_TABLE_NODE_NAME = "SV3D Probe Pair Coordination Summary"
PROBE_COORDINATION_SUMMARY_TABLE_NODE_NAME = "SV3D Probe Coordination Summary"
NO_TOUCH_SUMMARY_TABLE_NODE_NAME = "SV3D NoTouch Summary"
EXPORT_SUMMARY_TABLE_NODE_NAME = "SV3D Export Summary"
EXPORT_MANIFEST_PREVIEW_TABLE_NODE_NAME = "SV3D Export Manifest Preview"
COHORT_EXECUTION_SUMMARY_TABLE_NODE_NAME = "SV3D Cohort Execution Summary"
COHORT_CASE_SUMMARY_TABLE_NODE_NAME = "SV3D Cohort Case Summary"
COHORT_AGGREGATE_METRICS_TABLE_NODE_NAME = "SV3D Cohort Aggregate Metrics"
COHORT_COMPARISON_SUMMARY_TABLE_NODE_NAME = "SV3D Cohort Comparison Summary"
REPRODUCIBILITY_PACKAGE_SUMMARY_TABLE_NODE_NAME = "SV3D Reproducibility Package Summary"
REPRODUCIBILITY_MANIFEST_PREVIEW_TABLE_NODE_NAME = "SV3D Reproducibility Manifest Preview"
REPRODUCIBILITY_ARTIFACT_INDEX_TABLE_NODE_NAME = "SV3D Reproducibility Artifact Index"
MASTER_TRAJECTORY_VALIDATION_TABLE_NODE_NAME = "SV3D Master Trajectory Validation"
MASTER_PLAN_SNAPSHOT_TABLE_NODE_NAME = "SV3D Locked Master Plan"
COAXIAL_PLAN_TABLE_NODE_NAME = "SV3D Coaxial Plan Summary"
COAXIAL_NAVIGATION_TARGET_NODE_NAME = "SV3D Coaxial Navigation Target"
COAXIAL_NAVIGATION_LINE_NODE_NAME = "SV3D Coaxial Navigation Line"
TEMP_PROBE_SAFETY_MODEL_NODE_NAME = "SV3D Temp Probe Safety Input"
TEMP_STRUCTURE_SAFETY_MODEL_NODE_NAME = "SV3D Temp Structure Safety Input"
TEMP_STRUCTURE_SAFETY_DISTANCE_MODEL_NODE_NAME = "SV3D Temp Structure Safety Distance"
SAMPLE_DATA_CATEGORY = "SurgicalVision3D Planner"
SAMPLE_DATA_CRLM1001_NAME = "CRLM-1001 Demo Scene"
SAMPLE_DATA_CRLM1001_RELATIVE_SCENE_PATH = "Resources/Cohorts/CRLM-1001/DemoScene.mrml"
SAMPLE_DATA_CRLM1001_RELATIVE_THUMBNAIL_PATH = "Resources/Cohorts/CRLM-1001/DemoScene.png"
SAMPLE_DATA_ALREADY_REGISTERED = False
BEGINNER_WORKFLOW_MODE = True
BEGINNER_GEOMETRY_CATALOG_RELATIVE_PATH = "Resources/geometry_catalog.json"
REFERENCE_PROBE_TEMPLATE_DIRECTORY_RELATIVE_PATH = "Resources/Geometries"
BEGINNER_WORKFLOW_BANNER_TEXT = (
    "Beginner mode: import a case, define one master trajectory, validate it, optionally run endpoint auto-adjust after a failed validation, assess MAM, lock and derive coaxial guidance, then export the full planning package."
)
BEGINNER_MAM_COLOR_NODE_NAME = "SV3D Beginner MAM Colors"
BEGINNER_TRAJECTORY_DISTANCE_SAMPLE_COUNT = 121
LINE_THICKNESS_SCALE = 0.20
BEGINNER_AUTO_ADJUST_ENDPOINT_BUTTON_TEXT = "Auto-adjust Endpoint"
AUTO_ADJUST_MAX_ENDPOINT_SHIFT_MM = 15.0
AUTO_ADJUST_ENDPOINT_SHIFT_STEP_MM = 1.5
AUTO_ADJUST_AZIMUTH_SAMPLE_COUNT = 36
DEFAULT_CT_ABDOMEN_WINDOW = 350.0
DEFAULT_CT_ABDOMEN_LEVEL = 40.0
EVALUATE_DERIVED_BUNDLE_INLINE_ON_PLACEMENT = False
USE_SEGMENT_EDITOR_UNION_FOR_PROBE_MERGE = True
# Guard rail: VTK distance fast path can hard-crash on malformed surfaces in some scenes.
# Keep disabled by default and enable only after validating scene geometry stability.
ENABLE_VTK_DISTANCE_FAST_PATH = False
ENABLE_MAM_DEBUG_LOGGING = True


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    normalized = np.array(vector, dtype=float)
    if normalized.size != 3:
        raise ValueError("Vector must have exactly 3 components.")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Cannot normalize a vector with non-finite values.")
    length = float(np.linalg.norm(normalized))
    if not math.isfinite(length) or length <= 1e-8:
        raise ValueError("Cannot normalize a zero-length vector.")
    return normalized / length


def rotation_matrix_from_vectors(source_vector: Sequence[float], target_vector: Sequence[float]) -> np.ndarray:
    """Compute a robust 3x3 rotation matrix that aligns source_vector with target_vector."""

    source = _normalize_vector(source_vector)
    target = _normalize_vector(target_vector)
    dot_product = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if math.isclose(dot_product, 1.0, abs_tol=1e-8):
        return np.eye(3)

    if math.isclose(dot_product, -1.0, abs_tol=1e-8):
        orthogonal_axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=float))
        if np.linalg.norm(orthogonal_axis) <= 1e-8:
            orthogonal_axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=float))
        axis = _normalize_vector(orthogonal_axis)
        return (2.0 * np.outer(axis, axis)) - np.eye(3)

    cross_product = np.cross(source, target)
    cross_length = float(np.linalg.norm(cross_product))
    skew = np.array(
        [
            [0.0, -cross_product[2], cross_product[1]],
            [cross_product[2], 0.0, -cross_product[0]],
            [-cross_product[1], cross_product[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + skew + skew.dot(skew) * ((1.0 - dot_product) / (cross_length * cross_length))


def _build_rigid_transform(rotation_matrix: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[0:3, 0:3] = rotation_matrix
    transform[0:3, 3] = np.array(translation, dtype=float)
    return transform


def _data_array_value_count(data_array: vtk.vtkDataArray) -> int:
    return int(data_array.GetNumberOfValues())


@dataclass
class ProbeTrajectory:
    entryPointRAS: tuple[float, float, float]
    targetPointRAS: tuple[float, float, float]
    directionVector: tuple[float, float, float]
    lengthMm: float
    trajectoryIndex: int
    generatedProbeNodeID: str | None = None
    label: str = ""
    status: str = "pending"
    sourceControlPointIndices: tuple[int, int] | None = None
    role: str = "Master"
    angleDeg: float | None = None
    radialOffsetMm: float = 0.0
    derivedFromMaster: bool = False


@dataclass
class GeometryCatalogEntry:
    geometryId: str
    displayName: str
    templateRelativePath: str
    activeElementLengthMm: float
    axialPlacementOffsetMm: float = 0.0


@dataclass
class ProbeCoordinationConstraintSettings:
    minInterProbeDistanceMm: float = 5.0
    maxInterProbeDistanceMm: float = 120.0
    minEntryPointSpacingMm: float = 5.0
    minTargetPointSpacingMm: float = 3.0
    maxParallelAngleDeg: float = 10.0
    maxAllowedOverlapPercentBetweenPerProbeVolumes: float = 80.0
    enableNoTouchCheck: bool = False
    requireAllProbePairsFeasible: bool = True
    enableInterProbeDistanceRule: bool = True
    enableEntrySpacingRule: bool = True
    enableTargetSpacingRule: bool = True
    enableAngleRule: bool = False
    enableOverlapRule: bool = False


@dataclass
class PlanExportConfig:
    exportMode: str = "CurrentWorkingPlan"
    selectedExportScenarioID: str = ""
    exportBaseName: str = "SV3D_Export"
    exportDirectory: str = ""
    lastExportSequence: int = 0
    includeWorkingPlan: bool = True
    includeSelectedScenario: bool = False
    includeScenarioComparison: bool = True
    includeRecommendationOutputs: bool = True
    includeTrajectoryTables: bool = True
    includeSafetyTables: bool = True
    includeCoverageTables: bool = True
    includeFeasibilityTables: bool = True
    includeCoordinationTables: bool = True


@dataclass
class PlanExportManifest:
    exportId: str
    exportTimestampISO: str
    exportSequence: int
    exportMode: str
    exportBaseName: str
    selectedScenarioID: str
    selectedScenarioName: str
    profileSourceMode: str
    presetID: str
    presetName: str
    targetSegmentID: str
    targetSegmentName: str
    filesExported: list[str]
    includeFlags: dict[str, bool]
    notes: str = ""


@dataclass
class CohortCaseMember:
    caseId: str
    displayName: str
    inputReference: str = "ScenarioID"
    scenarioId: str = ""
    presetId: str = ""
    targetSegmentId: str = ""
    notes: str = ""


@dataclass
class CohortStudyDefinition:
    studyId: str
    displayName: str = ""
    description: str = ""
    cases: list[CohortCaseMember] = field(default_factory=list)


@dataclass
class CohortExecutionConfig:
    studyDefinitionPath: str = ""
    executionMode: str = "ScenarioRegistry"
    includeMarginMetrics: bool = True
    includeSafetyMetrics: bool = True
    includeCoverageMetrics: bool = True
    includeFeasibilityMetrics: bool = True
    includeCoordinationMetrics: bool = True
    includeVerificationMetrics: bool = True
    includeRecommendationMetrics: bool = True
    maxCases: int = 0


@dataclass
class CohortCaseResult:
    caseId: str
    displayName: str
    inputReference: str
    scenarioId: str
    executionStatus: str
    statusMessage: str
    presetId: str = ""
    targetSegmentId: str = ""
    metricValues: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReproducibilityPackageConfig:
    packageMode: str = "ReviewerSupplement"
    includeBenchmarkArtifacts: bool = True
    includeScenarioRegistry: bool = True
    includeCohortStudyArtifacts: bool = True
    includeStudyAnalytics: bool = True
    includeReports: bool = True
    includeCanonicalJson: bool = True
    includeValidationResults: bool = True
    packageBaseName: str = "SV3D_ReproducibilityPackage"
    outputDirectory: str = ""
    lastPackageSequence: int = 0


@dataclass
class ReproducibilityArtifactEntry:
    artifactKey: str
    category: str
    relativePath: str
    status: str
    sourcePath: str = ""
    sizeBytes: int = 0
    sha256: str = ""
    warning: str = ""


@dataclass
class ReproducibilityManifest:
    packageId: str
    packageTimestampISO: str
    packageSequence: int
    packageMode: str
    packageBaseName: str
    createdByModule: str
    schemaVersions: dict[str, str]
    includedArtifacts: list[dict[str, Any]]
    benchmarkCaseIds: list[str]
    studyIds: list[str]
    scenarioIds: list[str]
    reportIds: list[str]
    warnings: list[str]
    notes: str = ""


@dataclass
class LockedMasterPlanSnapshot:
    geometryId: str
    geometryDisplayName: str
    mamMm: float
    entryPointRAS: tuple[float, float, float]
    targetPointRAS: tuple[float, float, float]
    directionVector: tuple[float, float, float]
    trajectoryLengthMm: float
    activeElementLengthMm: float
    trajectoryValidationPass: bool
    marginValidationPass: bool
    lockedAtISO: str
    tumorSegmentationID: str = ""
    tumorSegmentID: str = ""
    tumorSegmentName: str = ""
    endpointsMarkupsID: str = ""
    firstDroppedPointRAS: tuple[float, float, float] | None = None
    secondDroppedPointRAS: tuple[float, float, float] | None = None


@dataclass
class CoaxialPlanSummary:
    technique: str
    activeElementLengthMm: float
    navigationTargetRAS: tuple[float, float, float]
    masterEntryPointRAS: tuple[float, float, float]
    masterTargetPointRAS: tuple[float, float, float]
    spareMm: float = 0.0
    pushThroughOffsetMm: float = 0.0
    notes: str = ""


@dataclass
class DerivedTrajectoryArrayConfig:
    planningMode: str = "Single"
    derivedTrajectoryCount: int = 4
    radiusMm: float = 10.0
    angleOffsetDeg: float = 0.0
    includeMasterTrajectory: bool = True


@dataclass
class DerivedTrajectoryDescriptor:
    trajectoryIndex: int
    role: str
    angleDeg: float
    radialOffsetMm: float
    source: str = "derived"


@dataclass
class DerivedTrajectoryArraySummary:
    planningMode: str
    masterIncluded: bool
    derivedTrajectoryCount: int
    totalTrajectoryCount: int
    radiusMm: float
    angleOffsetDeg: float
    notes: str = ""


@dataclass
class EndpointAutoAdjustResult:
    applied: bool
    reason: str
    statusText: str
    maxEndpointShiftMm: float
    shiftStepMm: float
    azimuthSampleCount: int
    checkedCandidateCount: int
    insideTumorCandidateCount: int
    zeroIntersectionCandidateCount: int
    selectedEndpointShiftMm: float = 0.0
    selectedMinDistanceMm: float = float("nan")
    selectedTargetPointRAS: tuple[float, float, float] | None = None
    selectedShellIndex: int = -1
    selectedAzimuthIndex: int = -1


#
# SurgicalVision3D_Planner
#


class SurgicalVision3D_Planner(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SurgicalVision3D Planner")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Quantification")]
        self.parent.dependencies = ["SegmentEditor", "FiducialRegistration"]
        self.parent.contributors = ["Juan Verde (Surgeon Scientist)"]
        self.parent.helpText = _("""
Research-oriented ablation planning module for 3D Slicer.
1. Import cases, choose applicator geometries, and define master or paired trajectories.
2. Generate single or deterministic multi-trajectory probe plans and merge them into one ablation zone.
3. Evaluate signed margins, MAM coverage, structure safety, and probe-coordination constraints.
4. Lock validated plans, compute coaxial guidance, and export deterministic cohort/reproducibility artifacts.
""")
        self.parent.acknowledgementText = _("""
Refactored from the legacy AblationPlanner workflow and extended with beginner-guided planning,
derived trajectory arrays, cohort summaries, export bundles, and reproducibility packaging.
""")
        try:
            slicer.app.connect("startupCompleted()", registerSampleData)
        except Exception:
            logging.exception("Failed to connect SurgicalVision3D Planner sample-data registration callback.")
        registerSampleData()


def registerSampleData() -> None:
    global SAMPLE_DATA_ALREADY_REGISTERED
    if SAMPLE_DATA_ALREADY_REGISTERED:
        return

    try:
        import SampleData
    except Exception:
        logging.debug("SampleData module is not available yet; deferring bundled sample registration.")
        return

    moduleDirectory = Path(__file__).resolve().parent
    demoScenePath = (moduleDirectory / SAMPLE_DATA_CRLM1001_RELATIVE_SCENE_PATH).resolve()
    demoThumbnailPath = (moduleDirectory / SAMPLE_DATA_CRLM1001_RELATIVE_THUMBNAIL_PATH).resolve()
    if not demoScenePath.exists():
        logging.warning("Bundled sample scene was not found: %s", demoScenePath)
        return

    def loadBundledCrlm1001Scene(source=None, *args, **kwargs):
        SurgicalVision3D_PlannerLogic.loadBundledSampleScene(demoScenePath)
        return True

    registrationArguments: dict[str, Any] = {
        "category": SAMPLE_DATA_CATEGORY,
        "sampleName": SAMPLE_DATA_CRLM1001_NAME,
        "fileNames": demoScenePath.name,
        "uris": demoScenePath.as_uri(),
        "nodeNames": SAMPLE_DATA_CRLM1001_NAME,
        "loadFileType": "SceneFile",
        "loadFiles": True,
        "customDownloader": loadBundledCrlm1001Scene,
    }
    if demoThumbnailPath.exists():
        registrationArguments["thumbnailFileName"] = str(demoThumbnailPath)

    try:
        SampleData.SampleDataLogic.registerCustomSampleDataSource(**registrationArguments)
        SAMPLE_DATA_ALREADY_REGISTERED = True
    except TypeError:
        # Fallback for older SampleData APIs that do not accept customDownloader.
        fallbackArguments: dict[str, Any] = {
            "category": SAMPLE_DATA_CATEGORY,
            "sampleName": SAMPLE_DATA_CRLM1001_NAME,
            "fileNames": demoScenePath.name,
            "uris": demoScenePath.as_uri(),
            "nodeNames": SAMPLE_DATA_CRLM1001_NAME,
            "loadFileType": "SceneFile",
            "loadFiles": True,
        }
        if demoThumbnailPath.exists():
            fallbackArguments["thumbnailFileName"] = str(demoThumbnailPath)
        SampleData.SampleDataLogic.registerCustomSampleDataSource(**fallbackArguments)
        SAMPLE_DATA_ALREADY_REGISTERED = True


#
# SurgicalVision3D_PlannerParameterNode
#


@parameterNodeWrapper
class SurgicalVision3D_PlannerParameterNode:
    caseFolderPath: str = ""
    selectedGeometryId: str = ""
    referenceProbeSegmentation: vtkMRMLSegmentationNode | None = None
    endpointsMarkups: vtkMRMLMarkupsFiducialNode | None = None
    tumorSegmentation: vtkMRMLSegmentationNode | None = None
    criticalStructuresSegmentation: vtkMRMLSegmentationNode | None = None
    riskStructuresSegmentation: vtkMRMLSegmentationNode | None = None
    nativeFiducials: vtkMRMLMarkupsFiducialNode | None = None
    registeredFiducials: vtkMRMLMarkupsFiducialNode | None = None
    combinedProbeSegmentation: vtkMRMLSegmentationNode | None = None
    outputMarginModel: vtkMRMLModelNode | None = None
    resultTable: vtkMRMLTableNode | None = None
    tumorTransform: vtkMRMLTransformNode | None = None
    coaxialNavigationTarget: vtkMRMLMarkupsFiducialNode | None = None
    trajectorySummaryTable: vtkMRMLTableNode | None = None
    derivedTrajectorySummaryTable: vtkMRMLTableNode | None = None
    planSummaryTable: vtkMRMLTableNode | None = None
    marginThresholdSummaryTable: vtkMRMLTableNode | None = None
    structureSafetySummaryTable: vtkMRMLTableNode | None = None
    structureSafetyThresholdSummaryTable: vtkMRMLTableNode | None = None
    probeCoordinationConstraintSettingsTable: vtkMRMLTableNode | None = None
    probePairCoordinationSummaryTable: vtkMRMLTableNode | None = None
    probeCoordinationSummaryTable: vtkMRMLTableNode | None = None
    noTouchSummaryTable: vtkMRMLTableNode | None = None
    exportSummaryTable: vtkMRMLTableNode | None = None
    exportManifestPreviewTable: vtkMRMLTableNode | None = None
    cohortExecutionSummaryTable: vtkMRMLTableNode | None = None
    cohortCaseSummaryTable: vtkMRMLTableNode | None = None
    cohortAggregateMetricsTable: vtkMRMLTableNode | None = None
    cohortComparisonSummaryTable: vtkMRMLTableNode | None = None
    reproducibilityPackageSummaryTable: vtkMRMLTableNode | None = None
    reproducibilityManifestPreviewTable: vtkMRMLTableNode | None = None
    reproducibilityArtifactIndexTable: vtkMRMLTableNode | None = None
    masterTrajectoryValidationTable: vtkMRMLTableNode | None = None
    masterPlanSnapshotTable: vtkMRMLTableNode | None = None
    coaxialPlanTable: vtkMRMLTableNode | None = None
    derivedTrajectoryPreviewNode: vtkMRMLMarkupsLineNode | None = None

    createTrajectoryLinesOnPlacement: bool = True
    clearPreviousGeneratedProbes: bool = True
    placeMultipleControlPoints: bool = True
    trajectoryPlanningMode: Annotated[str, Choice(["Single", "MultiTrajectoryArray"])] = "Single"
    derivedTrajectoryCount: int = 4
    derivedTrajectoryRadiusMm: float = 10.0
    derivedTrajectoryAngleOffsetDeg: float = 0.0
    includeMasterTrajectoryInArray: bool = True
    mamMm: float = 10.0
    recolorThresholdLow: float = -10.0
    recolorThresholdMid: float = -5.0
    recolorThresholdHigh: float = -2.0
    minInterProbeDistanceMm: float = 5.0
    maxInterProbeDistanceMm: float = 120.0
    minEntryPointSpacingMm: float = 5.0
    minTargetPointSpacingMm: float = 3.0
    maxParallelAngleDeg: float = 10.0
    maxAllowedOverlapPercentBetweenPerProbeVolumes: float = 80.0
    enableNoTouchCheck: bool = False
    requireAllProbePairsFeasible: bool = True
    enableInterProbeDistanceRule: bool = True
    enableEntrySpacingRule: bool = True
    enableTargetSpacingRule: bool = True
    enableAngleRule: bool = False
    enableOverlapRule: bool = False
    exportMode: Annotated[str, Choice(["CurrentWorkingPlan", "SelectedScenario", "CurrentRecommendationContext"])] = "CurrentWorkingPlan"
    selectedExportScenarioID: str = ""
    exportBaseName: str = "SV3D_Export"
    lastExportDirectory: str = ""
    lastExportSequence: int = 0
    includeWorkingPlan: bool = True
    includeSelectedScenario: bool = False
    includeScenarioComparison: bool = True
    includeRecommendationOutputs: bool = True
    includeTrajectoryTables: bool = True
    includeSafetyTables: bool = True
    includeCoverageTables: bool = True
    includeFeasibilityTables: bool = True
    includeCoordinationTables: bool = True
    cohortStudyDefinitionPath: str = "Resources/Cohorts/studies/example_cohort_v1.json"
    cohortExecutionMode: Annotated[str, Choice(["ScenarioRegistry", "CurrentWorkingPlan"])] = "ScenarioRegistry"
    cohortIncludeMarginMetrics: bool = True
    cohortIncludeSafetyMetrics: bool = True
    cohortIncludeCoverageMetrics: bool = True
    cohortIncludeFeasibilityMetrics: bool = True
    cohortIncludeCoordinationMetrics: bool = True
    cohortIncludeVerificationMetrics: bool = True
    cohortIncludeRecommendationMetrics: bool = True
    cohortMaxCases: int = 0
    packageMode: Annotated[str, Choice(["ReviewerSupplement", "ValidationArchive", "InternalHandoff"])] = "ReviewerSupplement"
    includeBenchmarkArtifacts: bool = True
    includeScenarioRegistry: bool = True
    includeCohortStudyArtifacts: bool = True
    includeStudyAnalytics: bool = True
    includeReports: bool = True
    includeCanonicalJson: bool = True
    includeValidationResults: bool = True
    packageBaseName: str = "SV3D_ReproducibilityPackage"
    packageOutputDirectory: str = ""
    lastReproducibilityPackageSequence: int = 0
    masterTrajectoryLocked: bool = False
    coaxialTechnique: Annotated[str, Choice(["PullBack", "PushThrough"])] = "PullBack"
    coaxialSpareMm: float = 5.0

    generatedProbeNodeIDs: str = "[]"
    generatedTrajectoryLineIDs: str = "[]"
    derivedTrajectoryBundleSummaryJson: str = "{}"
    trajectoryValidationSummaryJson: str = "{}"
    endpointAutoAdjustSummaryJson: str = "{}"
    mamAssessmentSummaryJson: str = "{}"
    masterPlanSnapshotJson: str = "{}"
    coaxialPlanSummaryJson: str = "{}"


#
# SurgicalVision3D_PlannerWidget
#


class SurgicalVision3D_PlannerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic: SurgicalVision3D_PlannerLogic | None = None
        self._parameterNode: SurgicalVision3D_PlannerParameterNode | None = None
        self._parameterNodeGuiTag = None
        self._gitRepositoryRoot: Path | None = None
        self._gitAgentLogPath: Path | None = None
        self._gitDashboardStatusLabel = None
        self._gitDashboardTextEdit = None
        self._gitCommitMessageLineEdit = None
        self._gitEntryLineEdit = None
        self._gitRefreshButton = None
        self._gitStageAllButton = None
        self._gitCommitButton = None
        self._gitPushButton = None
        self._gitAddEntryButton = None
        self._gitDashboardCollapsibleButton = None
        self._beginnerModeBannerLabel = None
        self._beginnerWorkflowSections: list[ctk.ctkCollapsibleButton] = []
        self._caseFolderLineEdit = None
        self._browseCaseFolderButton = None
        self._browseExportDirectoryButton = None
        self._importCaseFolderButton = None
        self._openSegmentEditorButton = None
        self._geometryComboBox = None
        self._planningModeComboBox = None
        self._derivedTrajectoryCountSpinBox = None
        self._derivedTrajectoryRadiusSpinBox = None
        self._derivedTrajectoryAngleOffsetSpinBox = None
        self._includeMasterTrajectoryCheckBox = None
        self._previewDerivedArrayButton = None
        self._clearDerivedArrayButton = None
        self._derivedArrayStatusLabel = None
        self._mamSpinBox = None
        self._validateTrajectoryButton = None
        self._autoAdjustEndpointButton = None
        self._trajectoryRescueGuideLabel = None
        self._trajectoryValidationStatusLabel = None
        self._segmentEditorStatusLabel = None
        self._marginAssessmentStatusLabel = None
        self._lockMasterPlanButton = None
        self._resetMasterPlanButton = None
        self._coaxialTechniqueComboBox = None
        self._coaxialSpareSpinBox = None
        self._computeCoaxialPlanButton = None
        self._coaxialStatusLabel = None
        self._exportWorkflowStatusLabel = None
        self._placeMultipleControlPointsCheckBox = None
        self._endpointsPlaceWidget = None
        self._observedEndpointsMarkupsNode = None
        self._observedEndpointsGeometrySignature = ""
        self._sceneImportInProgress = False
        self._sceneCloseInProgress = False
        self._step123TraceSignature = ""

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SurgicalVision3D_Planner.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self._configureTooltips()
        self._createGitAgentDashboard(uiWidget)
        self._gitRepositoryRoot = self._resolveGitRepositoryRoot()
        if self._gitRepositoryRoot:
            self._gitAgentLogPath = self._gitRepositoryRoot / "git_agent_log.md"
        self._refreshGitDashboard()
        self._createBeginnerWorkflowUi(uiWidget)
        self._applyBeginnerWorkflowMode(uiWidget)
        self._configureBeginnerTooltips()

        for selectorName in (
            "probeSegmentationSelector",
            "endpointsMarkupsSelector",
            "tumorSegmentationSelector",
            "riskStructuresSegmentationSelector",
            "nativeFiducialsSelector",
            "registeredFiducialsSelector",
            "combinedProbeSegmentationSelector",
            "outputMarginModelSelector",
            "resultTableSelector",
            "tumorTransformSelector",
        ):
            selector = getattr(self.ui, selectorName, None)
            if selector:
                selector.setMRMLScene(slicer.mrmlScene)

        self._configureNodeSelectorShowHidden(getattr(self.ui, "probeSegmentationSelector", None), True)
        self._configureNodeSelectorShowHidden(getattr(self.ui, "tumorSegmentationSelector", None), False)
        self._configureNodeSelectorShowHidden(getattr(self.ui, "riskStructuresSegmentationSelector", None), False)
        self._configureNodeSelectorNodeTypes(
            getattr(self.ui, "probeSegmentationSelector", None),
            ("vtkMRMLSegmentationNode",),
        )
        self._configureNodeSelectorNodeTypes(
            getattr(self.ui, "tumorSegmentationSelector", None),
            ("vtkMRMLSegmentationNode",),
        )
        self._configureNodeSelectorNodeTypes(
            getattr(self.ui, "riskStructuresSegmentationSelector", None),
            ("vtkMRMLSegmentationNode",),
        )
        self._configureNodeSelectorNodeTypes(
            getattr(self.ui, "combinedProbeSegmentationSelector", None),
            ("vtkMRMLSegmentationNode",),
        )

        self.logic = SurgicalVision3D_PlannerLogic()
        self.logic.ensureReferenceProbeTemplatesLoaded()
        self._populateSampleCaseComboBox()
        self._populateBeginnerGeometryComboBox()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        if hasattr(slicer.mrmlScene, "StartImportEvent"):
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartImportEvent, self.onSceneStartImport)
        if hasattr(slicer.mrmlScene, "EndImportEvent"):
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)

        if hasattr(self.ui, "loadSampleCaseButton"):
            self.ui.loadSampleCaseButton.connect("clicked(bool)", self.onLoadSampleCaseButton)
        if hasattr(self.ui, "sampleCaseComboBox"):
            self.ui.sampleCaseComboBox.connect("currentIndexChanged(int)", self._updateButtonStates)
        if hasattr(self.ui, "endpointsMarkupsSelector"):
            self.ui.endpointsMarkupsSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onEndpointsMarkupsChanged)
        if hasattr(self.ui, "nativeFiducialsSelector"):
            self.ui.nativeFiducialsSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onNativeFiducialsChanged)
        if hasattr(self.ui, "registeredFiducialsSelector"):
            self.ui.registeredFiducialsSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onRegisteredFiducialsChanged)
        if hasattr(self.ui, "probeSegmentationSelector"):
            self.ui.probeSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onReferenceProbeSegmentationChanged)
        if hasattr(self.ui, "tumorSegmentationSelector"):
            self.ui.tumorSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onTumorSegmentationChanged)
        if hasattr(self.ui, "combinedProbeSegmentationSelector"):
            self.ui.combinedProbeSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onCombinedProbeSegmentationChanged)
        self.ui.placeProbesButton.connect("clicked(bool)", self.onPlaceProbesButton)
        self.ui.createTrajectoryLinesButton.connect("clicked(bool)", self.onCreateTrajectoryLinesButton)
        self.ui.mergeTranslatedProbesButton.connect("clicked(bool)", self.onMergeTranslatedProbesButton)
        self.ui.registerTumorButton.connect("clicked(bool)", self.onRegisterTumorButton)
        self.ui.hardenTumorTransformButton.connect("clicked(bool)", self.onHardenTumorTransformButton)
        self.ui.evaluateMarginsButton.connect("clicked(bool)", self.onEvaluateMarginsButton)
        self.ui.recolorMarginsButton.connect("clicked(bool)", self.onRecolorMarginsButton)
        self.ui.resetMarginColorsButton.connect("clicked(bool)", self.onResetMarginColorsButton)
        self.ui.evaluateProbeCoordinationButton.connect("clicked(bool)", self.onEvaluateProbeCoordinationButton)
        self.ui.runCohortEvaluationButton.connect("clicked(bool)", self.onRunCohortEvaluationButton)
        self.ui.generateReproducibilityPackageButton.connect("clicked(bool)", self.onGenerateReproducibilityPackageButton)
        self.ui.exportBundleButton.connect("clicked(bool)", self.onExportBundleButton)
        if self._browseExportDirectoryButton:
            self._browseExportDirectoryButton.connect("clicked(bool)", self.onBrowseExportDirectoryButton)
        self.ui.riskStructuresSegmentationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.onRiskStructuresSegmentationChanged,
        )
        if hasattr(self.ui, "cohortStudyDefinitionPathLineEdit"):
            self.ui.cohortStudyDefinitionPathLineEdit.connect("textChanged(QString)", self._updateButtonStates)
        if hasattr(self.ui, "cohortExecutionModeComboBox"):
            self.ui.cohortExecutionModeComboBox.connect("currentIndexChanged(int)", self._updateButtonStates)
        if hasattr(self.ui, "exportModeComboBox"):
            self.ui.exportModeComboBox.connect("currentIndexChanged(int)", self._updateButtonStates)
        if hasattr(self.ui, "selectedExportScenarioIDLineEdit"):
            self.ui.selectedExportScenarioIDLineEdit.connect("textChanged(QString)", self._updateButtonStates)
        if hasattr(self.ui, "exportBaseNameLineEdit"):
            self.ui.exportBaseNameLineEdit.connect("textChanged(QString)", self._updateButtonStates)
        if hasattr(self.ui, "packageModeComboBox"):
            self.ui.packageModeComboBox.connect("currentIndexChanged(int)", self._updateButtonStates)
        if hasattr(self.ui, "packageBaseNameLineEdit"):
            self.ui.packageBaseNameLineEdit.connect("textChanged(QString)", self._updateButtonStates)
        if hasattr(self.ui, "packageOutputDirectoryLineEdit"):
            self.ui.packageOutputDirectoryLineEdit.connect("textChanged(QString)", self._updateButtonStates)
        if self._gitRefreshButton:
            self._gitRefreshButton.connect("clicked(bool)", self.onGitRefreshDashboardButton)
        if self._gitStageAllButton:
            self._gitStageAllButton.connect("clicked(bool)", self.onGitStageAllButton)
        if self._gitCommitButton:
            self._gitCommitButton.connect("clicked(bool)", self.onGitCommitButton)
        if self._gitPushButton:
            self._gitPushButton.connect("clicked(bool)", self.onGitPushButton)
        if self._gitAddEntryButton:
            self._gitAddEntryButton.connect("clicked(bool)", self.onGitAddEntryButton)
        if self._gitCommitMessageLineEdit:
            self._gitCommitMessageLineEdit.connect("textChanged(QString)", self._updateButtonStates)
        if self._gitEntryLineEdit:
            self._gitEntryLineEdit.connect("textChanged(QString)", self._updateButtonStates)
        if self._browseCaseFolderButton:
            self._browseCaseFolderButton.connect("clicked(bool)", self.onBrowseCaseFolderButton)
        if self._importCaseFolderButton:
            self._importCaseFolderButton.connect("clicked(bool)", self.onImportCaseFolderButton)
        if self._caseFolderLineEdit:
            self._caseFolderLineEdit.connect("textChanged(QString)", self.onCaseFolderPathChanged)
        if self._openSegmentEditorButton:
            self._openSegmentEditorButton.connect("clicked(bool)", self.onOpenSegmentEditorButton)
        if self._geometryComboBox:
            self._geometryComboBox.connect("currentIndexChanged(int)", self.onSelectedGeometryChanged)
        if self._planningModeComboBox:
            self._planningModeComboBox.connect("currentIndexChanged(int)", self.onTrajectoryPlanningModeChanged)
        if self._derivedTrajectoryCountSpinBox:
            self._derivedTrajectoryCountSpinBox.connect("valueChanged(int)", self.onDerivedTrajectoryCountChanged)
        if self._derivedTrajectoryRadiusSpinBox:
            self._derivedTrajectoryRadiusSpinBox.connect("valueChanged(double)", self.onDerivedTrajectoryRadiusChanged)
        if self._derivedTrajectoryAngleOffsetSpinBox:
            self._derivedTrajectoryAngleOffsetSpinBox.connect("valueChanged(double)", self.onDerivedTrajectoryAngleOffsetChanged)
        if self._includeMasterTrajectoryCheckBox:
            self._includeMasterTrajectoryCheckBox.connect("toggled(bool)", self.onIncludeMasterTrajectoryInArrayToggled)
        if self._previewDerivedArrayButton:
            self._previewDerivedArrayButton.connect("clicked(bool)", self.onPreviewDerivedArrayButton)
        if self._clearDerivedArrayButton:
            self._clearDerivedArrayButton.connect("clicked(bool)", self.onClearDerivedArrayButton)
        if self._mamSpinBox:
            self._mamSpinBox.connect("valueChanged(double)", self.onMamValueChanged)
        if self._placeMultipleControlPointsCheckBox:
            self._placeMultipleControlPointsCheckBox.connect("toggled(bool)", self.onPlaceMultipleControlPointsToggled)
        if self._validateTrajectoryButton:
            self._validateTrajectoryButton.connect("clicked(bool)", self.onValidateTrajectoryButton)
        if self._autoAdjustEndpointButton:
            self._autoAdjustEndpointButton.connect("clicked(bool)", self.onAutoAdjustEndpointButton)
        if self._lockMasterPlanButton:
            self._lockMasterPlanButton.connect("clicked(bool)", self.onLockMasterPlanButton)
        if self._resetMasterPlanButton:
            self._resetMasterPlanButton.connect("clicked(bool)", self.onResetMasterPlanButton)
        if self._coaxialTechniqueComboBox:
            self._coaxialTechniqueComboBox.connect("currentIndexChanged(int)", self.onCoaxialTechniqueChanged)
        if self._coaxialSpareSpinBox:
            self._coaxialSpareSpinBox.connect("valueChanged(double)", self.onCoaxialSpareMmChanged)
        if self._computeCoaxialPlanButton:
            self._computeCoaxialPlanButton.connect("clicked(bool)", self.onComputeCoaxialPlanButton)

        if self.parent.isEntered:
            self.initializeParameterNode()
        else:
            self._updateButtonStates()

    def _createGitAgentDashboard(self, uiWidget) -> None:
        if not uiWidget or not uiWidget.layout():
            return

        gitCollapsibleButton = ctk.ctkCollapsibleButton()
        self._gitDashboardCollapsibleButton = gitCollapsibleButton
        gitCollapsibleButton.text = "Git Agent Dashboard"
        gitFormLayout = qt.QFormLayout(gitCollapsibleButton)

        self._gitDashboardStatusLabel = qt.QLabel("Initializing git dashboard...")
        self._gitDashboardStatusLabel.wordWrap = True
        gitFormLayout.addRow("Status:", self._gitDashboardStatusLabel)

        self._gitDashboardTextEdit = qt.QPlainTextEdit()
        self._gitDashboardTextEdit.readOnly = True
        self._gitDashboardTextEdit.minimumHeight = 220
        self._gitDashboardTextEdit.setLineWrapMode(qt.QPlainTextEdit.NoWrap)
        gitFormLayout.addRow("Changes:", self._gitDashboardTextEdit)

        self._gitCommitMessageLineEdit = qt.QLineEdit()
        self._gitCommitMessageLineEdit.placeholderText = "Commit message..."
        gitFormLayout.addRow("Commit message:", self._gitCommitMessageLineEdit)

        self._gitEntryLineEdit = qt.QLineEdit()
        self._gitEntryLineEdit.placeholderText = "Log entry (optional note for git_agent_log.md)..."
        gitFormLayout.addRow("Agent log entry:", self._gitEntryLineEdit)

        buttonRowWidget = qt.QWidget()
        buttonRowLayout = qt.QHBoxLayout(buttonRowWidget)
        buttonRowLayout.setContentsMargins(0, 0, 0, 0)
        self._gitRefreshButton = qt.QPushButton("Refresh")
        self._gitStageAllButton = qt.QPushButton("Stage All")
        self._gitCommitButton = qt.QPushButton("Commit")
        self._gitPushButton = qt.QPushButton("Push")
        self._gitAddEntryButton = qt.QPushButton("Add Log Entry")
        buttonRowLayout.addWidget(self._gitRefreshButton)
        buttonRowLayout.addWidget(self._gitStageAllButton)
        buttonRowLayout.addWidget(self._gitCommitButton)
        buttonRowLayout.addWidget(self._gitPushButton)
        buttonRowLayout.addWidget(self._gitAddEntryButton)
        buttonRowLayout.addStretch(1)
        gitFormLayout.addRow(buttonRowWidget)

        self._gitRefreshButton.setToolTip("Refresh branch, pending changes, and recent commits.")
        self._gitStageAllButton.setToolTip("Stage all tracked/untracked changes using 'git add -A'.")
        self._gitCommitButton.setToolTip("Commit staged changes with the entered message.")
        self._gitPushButton.setToolTip("Push current branch to its configured remote/upstream.")
        self._gitAddEntryButton.setToolTip("Append an operator note to git_agent_log.md in the repository root.")

        uiWidget.layout().addWidget(gitCollapsibleButton)

    @staticmethod
    def _buildButtonRowWidget(*widgets) -> qt.QWidget:
        rowWidget = qt.QWidget()
        rowLayout = qt.QHBoxLayout(rowWidget)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            if widget:
                rowLayout.addWidget(widget)
        rowLayout.addStretch(1)
        return rowWidget

    @staticmethod
    def _createStatusLabel(initialText: str) -> qt.QLabel:
        label = qt.QLabel(initialText)
        label.wordWrap = True
        return label

    @staticmethod
    def _statusLabelStyleForText(text: str) -> str:
        normalizedText = str(text or "").strip().lower()
        if normalizedText.startswith("blocked:"):
            return "QLabel { color: #b71c1c; font-weight: 600; }"
        if normalizedText.startswith("ready:"):
            return "QLabel { color: #1b5e20; font-weight: 600; }"
        return ""

    def _setStatusLabelText(self, label, text: str) -> None:
        if not label:
            return
        label.text = str(text or "")
        label.setStyleSheet(self._statusLabelStyleForText(label.text))

    @staticmethod
    def _logStepTrace(stepName: str, message: str) -> None:
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        traceLine = f"[SV3D_TRACE] {timestamp} [{stepName}] {message}"
        try:
            logging.info(traceLine)
        except Exception:
            print(traceLine)

    @staticmethod
    def _nodeDisplayName(node) -> str:
        if not node:
            return "(none)"
        if not hasattr(node, "GetName"):
            return "(unnamed)"
        try:
            nodeName = str(node.GetName() or "")
        except Exception:
            nodeName = ""
        return nodeName if nodeName else "(unnamed)"

    def _traceStep123State(self, source: str, force: bool = False) -> None:
        if not BEGINNER_WORKFLOW_MODE:
            return

        sceneBusy = bool(self._sceneImportInProgress or self._sceneIsBusy())
        if not self._parameterNode:
            signature = f"{source}|busy:{int(sceneBusy)}|parameter:none"
            if (not force) and signature == self._step123TraceSignature:
                return
            self._step123TraceSignature = signature
            self._logStepTrace("Step1-3", f"{source}: parameter node unavailable (sceneBusy={sceneBusy}).")
            return

        nativeCount = self._markupsPointCount(self._parameterNode.nativeFiducials)
        registeredCount = self._markupsPointCount(self._parameterNode.registeredFiducials)
        registrationReady = bool(
            self._parameterNode.tumorSegmentation
            and self._parameterNode.nativeFiducials
            and self._parameterNode.registeredFiducials
            and nativeCount >= 3
            and registeredCount >= 3
            and nativeCount == registeredCount
        )
        hasTumorTransform = bool(
            self._parameterNode.tumorSegmentation
            and self._parameterNode.tumorSegmentation.GetTransformNodeID()
        )
        endpointCount = self._markupsPointCount(self._parameterNode.endpointsMarkups)
        hasSingleMasterTrajectory = endpointCount == 2
        planningMode = str(self._parameterNode.trajectoryPlanningMode or "Single")
        isLocked = bool(self._parameterNode.masterTrajectoryLocked)
        previewLineCount = 0
        if self.logic:
            previewLineCount = len(
                self.logic.resolveExistingNodeIDs(
                    self.logic.deserializeNodeIDs(self._parameterNode.generatedTrajectoryLineIDs)
                )
            )

        signature = "|".join(
            (
                source,
                f"busy:{int(sceneBusy)}",
                f"tumor:{self._nodeDisplayName(self._parameterNode.tumorSegmentation)}",
                f"critical:{self._nodeDisplayName(self._parameterNode.riskStructuresSegmentation)}",
                f"native:{nativeCount}",
                f"registered:{registeredCount}",
                f"regready:{int(registrationReady)}",
                f"hastransform:{int(hasTumorTransform)}",
                f"endpoints:{self._nodeDisplayName(self._parameterNode.endpointsMarkups)}",
                f"endpointcount:{endpointCount}",
                f"singlemaster:{int(hasSingleMasterTrajectory)}",
                f"mode:{planningMode}",
                f"locked:{int(isLocked)}",
                f"previewlines:{previewLineCount}",
            )
        )
        if (not force) and signature == self._step123TraceSignature:
            return
        self._step123TraceSignature = signature

        step1Ready = bool(self._parameterNode.tumorSegmentation and self._parameterNode.riskStructuresSegmentation)
        self._logStepTrace(
            "Step1-3",
            (
                f"{source}: Step1(imported={step1Ready}, "
                f"tumor='{self._nodeDisplayName(self._parameterNode.tumorSegmentation)}', "
                f"critical='{self._nodeDisplayName(self._parameterNode.riskStructuresSegmentation)}') | "
                f"Step2(registrationReady={registrationReady}, nativePts={nativeCount}, "
                f"registeredPts={registeredCount}, hasTransform={hasTumorTransform}) | "
                f"Step3(endpoints='{self._nodeDisplayName(self._parameterNode.endpointsMarkups)}', "
                f"points={endpointCount}, singleMaster={hasSingleMasterTrajectory}, "
                f"planningMode='{planningMode}', locked={isLocked}, previewLines={previewLineCount})"
            ),
        )

    def _createBeginnerWorkflowUi(self, uiWidget) -> None:
        if not BEGINNER_WORKFLOW_MODE or not uiWidget or not uiWidget.layout() or len(self._beginnerWorkflowSections) > 0:
            return

        importSection = ctk.ctkCollapsibleButton()
        importSection.text = "Step 1 - Import Case"
        importLayout = qt.QFormLayout(importSection)

        caseFolderRow = qt.QWidget()
        caseFolderLayout = qt.QHBoxLayout(caseFolderRow)
        caseFolderLayout.setContentsMargins(0, 0, 0, 0)
        self._caseFolderLineEdit = qt.QLineEdit()
        self._caseFolderLineEdit.placeholderText = "Select a case folder containing .nrrd / .seg.nrrd files"
        self._browseCaseFolderButton = qt.QPushButton("Browse")
        self._importCaseFolderButton = qt.QPushButton("Import")
        caseFolderLayout.addWidget(self._caseFolderLineEdit)
        caseFolderLayout.addWidget(self._browseCaseFolderButton)
        caseFolderLayout.addWidget(self._importCaseFolderButton)
        importLayout.addRow("Case folder:", caseFolderRow)
        importLayout.addRow("Bundled sample:", self._buildButtonRowWidget(self.ui.sampleCaseComboBox, self.ui.loadSampleCaseButton))

        segmentationSection = ctk.ctkCollapsibleButton()
        segmentationSection.text = "Step 2 - Segment Structures"
        segmentationLayout = qt.QFormLayout(segmentationSection)
        self._openSegmentEditorButton = qt.QPushButton("Open Segment Editor")
        self._segmentEditorStatusLabel = self._createStatusLabel(
            "Segment the tumor and critical structures in Segment Editor, then return to this module."
        )
        segmentationLayout.addRow("Tumor segmentation:", self.ui.tumorSegmentationSelector)
        segmentationLayout.addRow("Critical structures:", self.ui.riskStructuresSegmentationSelector)
        segmentationLayout.addRow("Native fiducials:", self.ui.nativeFiducialsSelector)
        segmentationLayout.addRow("Registered fiducials:", self.ui.registeredFiducialsSelector)
        segmentationLayout.addRow("Segment Editor:", self._openSegmentEditorButton)
        segmentationLayout.addRow("Registration:", self._buildButtonRowWidget(self.ui.registerTumorButton, self.ui.hardenTumorTransformButton))
        segmentationLayout.addRow("Status:", self._segmentEditorStatusLabel)

        trajectorySection = ctk.ctkCollapsibleButton()
        trajectorySection.text = "Step 3 - Define Master Trajectory"
        trajectoryLayout = qt.QFormLayout(trajectorySection)
        trajectoryInstructionLabel = self._createStatusLabel(
            "Use exactly two points in the markups node: entry point first, applicator endpoint second."
        )
        self._endpointsPlaceWidget = None
        if hasattr(slicer, "qSlicerMarkupsPlaceWidget"):
            try:
                self._endpointsPlaceWidget = slicer.qSlicerMarkupsPlaceWidget()
                self._endpointsPlaceWidget.setMRMLScene(slicer.mrmlScene)
                if hasattr(self._endpointsPlaceWidget, "setButtonsVisible"):
                    self._endpointsPlaceWidget.setButtonsVisible(False)
                elif hasattr(self._endpointsPlaceWidget, "buttonsVisible"):
                    self._endpointsPlaceWidget.buttonsVisible = False
                if hasattr(self._endpointsPlaceWidget, "placeButton"):
                    placeButton = self._endpointsPlaceWidget.placeButton()
                    if placeButton and hasattr(placeButton, "show"):
                        placeButton.show()
            except Exception:
                self._endpointsPlaceWidget = None
        self._placeMultipleControlPointsCheckBox = qt.QCheckBox("Place multiple control points")
        self._placeMultipleControlPointsCheckBox.checked = True
        endpointsRowWidget = self._buildButtonRowWidget(self.ui.endpointsMarkupsSelector, self._endpointsPlaceWidget)
        trajectoryLayout.addRow("Endpoint markups:", endpointsRowWidget)
        trajectoryLayout.addRow("", self._placeMultipleControlPointsCheckBox)
        trajectoryLayout.addRow("Instruction:", trajectoryInstructionLabel)
        trajectoryLayout.addRow("Preview:", self._buildButtonRowWidget(self.ui.createTrajectoryLinesButton))
        self._setEndpointsPlaceWidgetNode(self.ui.endpointsMarkupsSelector.currentNode() if hasattr(self.ui, "endpointsMarkupsSelector") else None)
        self._setEndpointsPlaceWidgetMultiplePlacement(True)

        validationSection = ctk.ctkCollapsibleButton()
        validationSection.text = "Step 4 - Validate + Endpoint Rescue"
        validationLayout = qt.QFormLayout(validationSection)
        self._validateTrajectoryButton = qt.QPushButton("Validate Against Critical Structures")
        self._trajectoryValidationStatusLabel = self._createStatusLabel("Trajectory not validated.")
        self._autoAdjustEndpointButton = qt.QPushButton(BEGINNER_AUTO_ADJUST_ENDPOINT_BUTTON_TEXT)
        self._trajectoryRescueGuideLabel = self._createStatusLabel(
            "Available only after a failed validation. It moves endpoint only (entry fixed), keeps trajectory length fixed, "
            f"and applies only when a zero-intersection endpoint is found inside tumor within {AUTO_ADJUST_MAX_ENDPOINT_SHIFT_MM:.1f} mm."
        )
        validationLayout.addRow("Validation:", self._validateTrajectoryButton)
        validationLayout.addRow("Rescue:", self._autoAdjustEndpointButton)
        validationLayout.addRow("Status:", self._trajectoryValidationStatusLabel)
        validationLayout.addRow("Guide:", self._trajectoryRescueGuideLabel)

        applicatorSection = ctk.ctkCollapsibleButton()
        applicatorSection.text = "Step 5 - Applicator Plan"
        applicatorLayout = qt.QFormLayout(applicatorSection)
        self._geometryComboBox = qt.QComboBox()
        self._planningModeComboBox = qt.QComboBox()
        self._planningModeComboBox.addItem("Single trajectory", "Single")
        self._planningModeComboBox.addItem("Multiple trajectories", "MultiTrajectoryArray")
        self._derivedTrajectoryCountSpinBox = qt.QSpinBox()
        self._derivedTrajectoryCountSpinBox.minimum = 1
        self._derivedTrajectoryCountSpinBox.maximum = 24
        self._derivedTrajectoryCountSpinBox.value = 4
        self._derivedTrajectoryRadiusSpinBox = ctk.ctkDoubleSpinBox()
        self._derivedTrajectoryRadiusSpinBox.minimum = 0.0
        self._derivedTrajectoryRadiusSpinBox.maximum = 100.0
        self._derivedTrajectoryRadiusSpinBox.decimals = 1
        self._derivedTrajectoryRadiusSpinBox.singleStep = 0.5
        self._derivedTrajectoryRadiusSpinBox.value = 10.0
        self._derivedTrajectoryAngleOffsetSpinBox = ctk.ctkDoubleSpinBox()
        self._derivedTrajectoryAngleOffsetSpinBox.minimum = -180.0
        self._derivedTrajectoryAngleOffsetSpinBox.maximum = 180.0
        self._derivedTrajectoryAngleOffsetSpinBox.decimals = 1
        self._derivedTrajectoryAngleOffsetSpinBox.singleStep = 1.0
        self._derivedTrajectoryAngleOffsetSpinBox.value = 0.0
        self._includeMasterTrajectoryCheckBox = qt.QCheckBox("Include master trajectory in placement bundle")
        self._includeMasterTrajectoryCheckBox.checked = True
        self._previewDerivedArrayButton = qt.QPushButton("Preview Derived Array")
        self._clearDerivedArrayButton = qt.QPushButton("Clear Derived Array")
        self._derivedArrayStatusLabel = self._createStatusLabel("Array not generated.")
        self._mamSpinBox = ctk.ctkDoubleSpinBox()
        self._mamSpinBox.minimum = 0.0
        self._mamSpinBox.maximum = 50.0
        self._mamSpinBox.decimals = 1
        self._mamSpinBox.singleStep = 0.5
        self._mamSpinBox.value = 10.0
        self._marginAssessmentStatusLabel = self._createStatusLabel("MAM assessment not run.")
        applicatorLayout.addRow("Ablation geometry:", self._geometryComboBox)
        applicatorLayout.addRow("Planning mode:", self._planningModeComboBox)
        applicatorLayout.addRow("Derived trajectories:", self._derivedTrajectoryCountSpinBox)
        applicatorLayout.addRow("Radius (mm):", self._derivedTrajectoryRadiusSpinBox)
        applicatorLayout.addRow("Angle offset (deg):", self._derivedTrajectoryAngleOffsetSpinBox)
        applicatorLayout.addRow("", self._includeMasterTrajectoryCheckBox)
        applicatorLayout.addRow(
            "Trajectory array:",
            self._buildButtonRowWidget(self._previewDerivedArrayButton, self._clearDerivedArrayButton),
        )
        applicatorLayout.addRow("Array status:", self._derivedArrayStatusLabel)
        applicatorLayout.addRow("MAM (mm):", self._mamSpinBox)
        applicatorLayout.addRow(
            "Planning:",
            self._buildButtonRowWidget(self.ui.placeProbesButton, self.ui.mergeTranslatedProbesButton, self.ui.evaluateMarginsButton),
        )
        applicatorLayout.addRow("Status:", self._marginAssessmentStatusLabel)

        lockSection = ctk.ctkCollapsibleButton()
        lockSection.text = "Step 6 - Lock + Coaxial Plan"
        lockLayout = qt.QFormLayout(lockSection)
        self._lockMasterPlanButton = qt.QPushButton("Lock Master Plan")
        self._resetMasterPlanButton = qt.QPushButton("Reset Lock")
        self._coaxialTechniqueComboBox = qt.QComboBox()
        self._coaxialTechniqueComboBox.addItem("PullBack")
        self._coaxialTechniqueComboBox.addItem("PushThrough")
        self._coaxialSpareSpinBox = ctk.ctkDoubleSpinBox()
        self._coaxialSpareSpinBox.minimum = 0.0
        self._coaxialSpareSpinBox.maximum = 20.0
        self._coaxialSpareSpinBox.decimals = 1
        self._coaxialSpareSpinBox.singleStep = 0.5
        self._coaxialSpareSpinBox.value = 5.0
        self._computeCoaxialPlanButton = qt.QPushButton("Compute Coaxial Plan")
        self._coaxialStatusLabel = self._createStatusLabel("Coaxial plan not computed.")
        lockLayout.addRow("Locking:", self._buildButtonRowWidget(self._lockMasterPlanButton, self._resetMasterPlanButton))
        lockLayout.addRow("Technique:", self._coaxialTechniqueComboBox)
        lockLayout.addRow("Spare (mm):", self._coaxialSpareSpinBox)
        lockLayout.addRow("Coaxial plan:", self._computeCoaxialPlanButton)
        lockLayout.addRow("Status:", self._coaxialStatusLabel)

        exportSection = ctk.ctkCollapsibleButton()
        exportSection.text = "Step 7 - Export Plan Package"
        exportLayout = qt.QFormLayout(exportSection)
        if hasattr(self.ui, "exportBaseNameLineEdit"):
            exportLayout.addRow("Export base name:", self.ui.exportBaseNameLineEdit)
        self._browseExportDirectoryButton = qt.QPushButton("Browse")
        if hasattr(self.ui, "exportDirectoryLineEdit"):
            exportLayout.addRow(
                "Export directory:",
                self._buildButtonRowWidget(self.ui.exportDirectoryLineEdit, self._browseExportDirectoryButton),
            )
        else:
            self._browseExportDirectoryButton = None
        exportLayout.addRow("Package:", self.ui.exportBundleButton)
        if hasattr(self.ui, "exportStatusLabel"):
            exportLayout.addRow("Last export:", self.ui.exportStatusLabel)
        self._exportWorkflowStatusLabel = self._createStatusLabel("Blocked: complete Step 6 first.")
        exportLayout.addRow("Status:", self._exportWorkflowStatusLabel)

        self._beginnerWorkflowSections = [
            importSection,
            segmentationSection,
            trajectorySection,
            validationSection,
            applicatorSection,
            lockSection,
            exportSection,
        ]
        for section in reversed(self._beginnerWorkflowSections):
            uiWidget.layout().insertWidget(0, section)

    def _populateBeginnerGeometryComboBox(self) -> None:
        if not self.logic or not self._geometryComboBox:
            return
        currentGeometryId = str(self._geometryComboBox.itemData(self._geometryComboBox.currentIndex) or "")
        self._geometryComboBox.blockSignals(True)
        self._geometryComboBox.clear()
        for geometryEntry in self.logic.loadGeometryCatalog():
            self._geometryComboBox.addItem(geometryEntry.displayName, geometryEntry.geometryId)
        if self._geometryComboBox.count > 0:
            targetGeometryId = currentGeometryId or str(self._geometryComboBox.itemData(0) or "")
            targetIndex = self._geometryComboBox.findData(targetGeometryId)
            self._geometryComboBox.currentIndex = targetIndex if targetIndex >= 0 else 0
        self._geometryComboBox.blockSignals(False)

    def _resolveGitRepositoryRoot(self) -> Path | None:
        startPath = Path(__file__).resolve().parent
        for candidate in (startPath, *startPath.parents):
            if (candidate / ".git").exists():
                return candidate
        return None

    def _runGitCommand(self, *gitArgs: str) -> tuple[int, str, str]:
        if not self._gitRepositoryRoot:
            raise RuntimeError("Git repository root was not found.")
        process = subprocess.run(
            ["git", *gitArgs],
            cwd=str(self._gitRepositoryRoot),
            text=True,
            capture_output=True,
            check=False,
        )
        return int(process.returncode), str(process.stdout or "").strip(), str(process.stderr or "").strip()

    def _appendGitAgentLogEntry(self, action: str, details: str) -> None:
        if not self._gitAgentLogPath:
            return
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        branch = "(unknown)"
        try:
            _, branchOut, _ = self._runGitCommand("rev-parse", "--abbrev-ref", "HEAD")
            if branchOut:
                branch = branchOut
        except Exception:
            pass
        with self._gitAgentLogPath.open("a", encoding="utf-8") as logFile:
            logFile.write(f"- {timestamp} [{branch}] {action}: {details}\n")

    def _refreshGitDashboard(self) -> None:
        if not self._gitDashboardTextEdit or not self._gitDashboardStatusLabel:
            return
        if not self._gitRepositoryRoot:
            self._gitDashboardStatusLabel.text = "Git repository not found for this module path."
            self._gitDashboardTextEdit.setPlainText("No repository context is available.")
            self._updateButtonStates()
            return

        statusLines: list[str] = []
        statusLines.append(f"Repository: {self._gitRepositoryRoot}")
        statusLines.append(f"Timestamp: {datetime.utcnow().replace(microsecond=0).isoformat()}Z")
        statusLines.append("")

        _, branchOut, branchErr = self._runGitCommand("rev-parse", "--abbrev-ref", "HEAD")
        branchName = branchOut or "(unknown)"
        if branchErr:
            statusLines.append(f"Branch lookup warning: {branchErr}")
        statusLines.append(f"Branch: {branchName}")
        statusLines.append("")

        _, statusOut, statusErr = self._runGitCommand("status", "--short", "--branch")
        statusLines.append("Pending changes:")
        statusLines.append(statusOut if statusOut else "(clean)")
        if statusErr:
            statusLines.append(f"Status warning: {statusErr}")
        statusLines.append("")

        _, stagedOut, _ = self._runGitCommand("diff", "--cached", "--name-status")
        statusLines.append("Staged files:")
        statusLines.append(stagedOut if stagedOut else "(none)")
        statusLines.append("")

        _, recentLogOut, recentLogErr = self._runGitCommand("log", "--oneline", "-n", "5")
        statusLines.append("Recent commits:")
        statusLines.append(recentLogOut if recentLogOut else "(no commits)")
        if recentLogErr:
            statusLines.append(f"Log warning: {recentLogErr}")

        if self._gitAgentLogPath and self._gitAgentLogPath.exists():
            recentEntries = self._gitAgentLogPath.read_text(encoding="utf-8").splitlines()[-10:]
            statusLines.append("")
            statusLines.append("Agent log (last 10 entries):")
            statusLines.extend(recentEntries if recentEntries else ["(empty)"])

        self._gitDashboardTextEdit.setPlainText("\n".join(statusLines))
        self._gitDashboardStatusLabel.text = f"Git dashboard ready ({branchName})."
        self._updateButtonStates()

    def onGitRefreshDashboardButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to refresh git dashboard."), waitCursor=False):
            self._refreshGitDashboard()

    def onGitStageAllButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to stage changes."), waitCursor=False):
            code, _stdoutText, stderrText = self._runGitCommand("add", "-A")
            if code != 0:
                raise RuntimeError(stderrText or "git add -A failed.")
            self._appendGitAgentLogEntry("STAGE", "Staged all changes with git add -A.")
            self._refreshGitDashboard()

    def onGitCommitButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to commit changes."), waitCursor=False):
            commitMessage = str(self._gitCommitMessageLineEdit.text).strip() if self._gitCommitMessageLineEdit else ""
            if not commitMessage:
                raise ValueError("Enter a commit message before committing.")
            code, stdoutText, stderrText = self._runGitCommand("commit", "-m", commitMessage)
            if code != 0:
                combinedOutput = "\n".join([line for line in (stdoutText, stderrText) if line.strip()])
                raise RuntimeError(combinedOutput or "git commit failed.")
            self._appendGitAgentLogEntry("COMMIT", commitMessage)
            if self._gitCommitMessageLineEdit:
                self._gitCommitMessageLineEdit.text = ""
            self._refreshGitDashboard()

    def onGitPushButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to push changes."), waitCursor=False):
            code, stdoutText, stderrText = self._runGitCommand("push")
            if code != 0:
                combinedOutput = "\n".join([line for line in (stdoutText, stderrText) if line.strip()])
                raise RuntimeError(combinedOutput or "git push failed.")
            self._appendGitAgentLogEntry("PUSH", "Pushed current branch to remote.")
            self._refreshGitDashboard()

    def onGitAddEntryButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to add git agent log entry."), waitCursor=False):
            entryText = str(self._gitEntryLineEdit.text).strip() if self._gitEntryLineEdit else ""
            if not entryText:
                raise ValueError("Enter a log entry before adding it.")
            self._appendGitAgentLogEntry("NOTE", entryText)
            if self._gitEntryLineEdit:
                self._gitEntryLineEdit.text = ""
            self._refreshGitDashboard()

    @staticmethod
    def _configureNodeSelectorBaseName(selector, baseName: str) -> None:
        if not selector:
            return
        if hasattr(selector, "setBaseName"):
            selector.setBaseName(baseName)
            return
        if hasattr(selector, "baseName"):
            selector.baseName = baseName

    def _renameGenericFiducialsNode(self, node, preferredBaseName: str) -> None:
        if not self.logic or not node or not node.IsA("vtkMRMLMarkupsFiducialNode"):
            return
        if self.logic.isGenericDefaultFiducialsNodeName(node.GetName()):
            node.SetName(slicer.mrmlScene.GenerateUniqueName(preferredBaseName))

    def onEndpointsMarkupsChanged(self, node) -> None:
        if self._sceneImportInProgress or self._sceneIsBusy():
            self._setEndpointsPlaceWidgetNode(node)
            self._setObservedEndpointsMarkupsNode(node)
            return
        self._renameGenericFiducialsNode(node, "endpoints")
        if self._parameterNode and node and self._parameterNode.endpointsMarkups is not node:
            self._parameterNode.endpointsMarkups = node
            self._invalidateGeneratedPlanningOutputs(
                "master trajectory source node changed. Re-preview and re-place probes.",
                invalidateMasterValidation=True,
            )
        if BEGINNER_WORKFLOW_MODE:
            pointCount = int(node.GetNumberOfControlPoints()) if node and hasattr(node, "GetNumberOfControlPoints") else 0
            self._logStepTrace("Step3", f"Endpoint markup node selected: '{node.GetName() if node else '(none)'}' ({pointCount} points)")
        self._setEndpointsPlaceWidgetNode(node)
        self._setObservedEndpointsMarkupsNode(node)
        self._autoSeedRegistrationFiducialsFromEndpoints()
        self._updateButtonStates()

    def onNativeFiducialsChanged(self, node) -> None:
        self._renameGenericFiducialsNode(node, "NativeFiducials")

    def onRegisteredFiducialsChanged(self, node) -> None:
        self._renameGenericFiducialsNode(node, "RegisteredFiducials")

    def onReferenceProbeSegmentationChanged(self, node) -> None:
        selector = getattr(self.ui, "probeSegmentationSelector", None) if hasattr(self, "ui") else None
        sanitizedNode = self._sanitizeSelectorNodeClass(
            selector,
            node,
            "vtkMRMLSegmentationNode",
            "probeSegmentationSelector",
        )
        if not self._parameterNode:
            return
        if self._sceneImportInProgress or self._sceneIsBusy():
            return
        if self._parameterNode.referenceProbeSegmentation is sanitizedNode:
            return
        self._parameterNode.referenceProbeSegmentation = sanitizedNode
        self._invalidateGeneratedPlanningOutputs("probe template changed. Re-preview and re-place probes.")
        self._updateButtonStates()

    def onTumorSegmentationChanged(self, node) -> None:
        selector = getattr(self.ui, "tumorSegmentationSelector", None) if hasattr(self, "ui") else None
        sanitizedNode = self._sanitizeSelectorNodeClass(
            selector,
            node,
            "vtkMRMLSegmentationNode",
            "tumorSegmentationSelector",
        )
        if not self._parameterNode:
            return
        if self._sceneImportInProgress or self._sceneIsBusy():
            return
        if self._parameterNode.tumorSegmentation is sanitizedNode:
            return
        self._parameterNode.tumorSegmentation = sanitizedNode
        self._clearOwnedSafetyOutputs(clearReferences=True)
        self._clearOwnedBeginnerOutputs(clearReferences=True)
        self._updateButtonStates()

    def onCombinedProbeSegmentationChanged(self, node) -> None:
        selector = getattr(self.ui, "combinedProbeSegmentationSelector", None) if hasattr(self, "ui") else None
        sanitizedNode = self._sanitizeSelectorNodeClass(
            selector,
            node,
            "vtkMRMLSegmentationNode",
            "combinedProbeSegmentationSelector",
        )
        if not self._parameterNode:
            return
        if self._sceneImportInProgress or self._sceneIsBusy():
            return
        if self._parameterNode.combinedProbeSegmentation is sanitizedNode:
            return
        self._parameterNode.combinedProbeSegmentation = sanitizedNode
        self._updateButtonStates()

    @staticmethod
    def _setPlaceModePersistence(enabled: bool) -> None:
        appLogic = slicer.app.applicationLogic() if hasattr(slicer, "app") else None
        interactionNode = appLogic.GetInteractionNode() if appLogic and hasattr(appLogic, "GetInteractionNode") else None
        if interactionNode and hasattr(interactionNode, "SetPlaceModePersistence"):
            interactionNode.SetPlaceModePersistence(1 if enabled else 0)

    def _setEndpointsPlaceWidgetNode(self, markupsNode) -> None:
        if not self._endpointsPlaceWidget or not hasattr(self._endpointsPlaceWidget, "setCurrentNode"):
            return
        try:
            self._endpointsPlaceWidget.setCurrentNode(markupsNode)
        except Exception:
            pass

    def _setEndpointsPlaceWidgetMultiplePlacement(self, enabled: bool) -> None:
        if not self._endpointsPlaceWidget:
            return
        placeWidgetClass = getattr(slicer, "qSlicerMarkupsPlaceWidget", None)
        if (
            placeWidgetClass
            and hasattr(placeWidgetClass, "ForcePlaceMultipleMarkups")
            and hasattr(placeWidgetClass, "ForcePlaceSingleMarkup")
            and hasattr(self._endpointsPlaceWidget, "placeMultipleMarkups")
        ):
            try:
                self._endpointsPlaceWidget.placeMultipleMarkups = (
                    placeWidgetClass.ForcePlaceMultipleMarkups
                    if enabled
                    else placeWidgetClass.ForcePlaceSingleMarkup
                )
                return
            except Exception:
                pass
        if hasattr(self._endpointsPlaceWidget, "setPlaceModePersistency"):
            try:
                self._endpointsPlaceWidget.setPlaceModePersistency(bool(enabled))
            except Exception:
                pass

    @staticmethod
    def _markupsPointCount(markupsNode) -> int:
        if not markupsNode:
            return 0
        if hasattr(markupsNode, "GetNumberOfDefinedControlPoints"):
            return int(markupsNode.GetNumberOfDefinedControlPoints())
        if hasattr(markupsNode, "GetNumberOfControlPoints"):
            return int(markupsNode.GetNumberOfControlPoints())
        return 0

    @staticmethod
    def _endpointsGeometrySignature(markupsNode) -> str:
        if not markupsNode:
            return ""
        if not hasattr(markupsNode, "GetNumberOfControlPoints"):
            return ""
        signatureParts: list[str] = [f"count:{int(markupsNode.GetNumberOfControlPoints())}"]
        pointPosition = [0.0, 0.0, 0.0]
        for pointIndex in range(int(markupsNode.GetNumberOfControlPoints())):
            pointStatus = 1
            if hasattr(markupsNode, "GetNthControlPointPositionStatus"):
                try:
                    pointStatus = int(markupsNode.GetNthControlPointPositionStatus(pointIndex))
                except Exception:
                    pointStatus = 1
            markupsNode.GetNthControlPointPosition(pointIndex, pointPosition)
            signatureParts.append(
                f"{pointStatus}:{pointPosition[0]:.4f},{pointPosition[1]:.4f},{pointPosition[2]:.4f}"
            )
        return "|".join(signatureParts)

    def _setObservedEndpointsMarkupsNode(self, markupsNode) -> None:
        if self._observedEndpointsMarkupsNode is markupsNode:
            return
        if self._observedEndpointsMarkupsNode:
            self.removeObserver(
                self._observedEndpointsMarkupsNode,
                vtk.vtkCommand.ModifiedEvent,
                self._onEndpointsMarkupsModified,
            )
        self._observedEndpointsMarkupsNode = markupsNode
        self._observedEndpointsGeometrySignature = self._endpointsGeometrySignature(markupsNode)
        if self._observedEndpointsMarkupsNode:
            self.addObserver(
                self._observedEndpointsMarkupsNode,
                vtk.vtkCommand.ModifiedEvent,
                self._onEndpointsMarkupsModified,
            )

    def _onEndpointsMarkupsModified(self, caller=None, event=None) -> None:
        if self._sceneImportInProgress or self._sceneIsBusy():
            return
        if not self._parameterNode:
            return
        if caller and self._parameterNode.endpointsMarkups is not caller:
            self._parameterNode.endpointsMarkups = caller
        geometrySignature = self._endpointsGeometrySignature(caller if caller else self._parameterNode.endpointsMarkups)
        geometryChanged = geometrySignature != self._observedEndpointsGeometrySignature
        self._observedEndpointsGeometrySignature = geometrySignature
        if geometryChanged:
            if BEGINNER_WORKFLOW_MODE and caller and hasattr(caller, "GetNumberOfControlPoints"):
                pointCount = int(caller.GetNumberOfControlPoints())
                self._logStepTrace("Step3", f"Master trajectory control points updated: {pointCount}")
            self._invalidateGeneratedPlanningOutputs(
                "master trajectory edited. Re-preview and re-place probes.",
                invalidateMasterValidation=True,
            )
        self._autoSeedRegistrationFiducialsFromEndpoints()
        self._updateButtonStates()

    @staticmethod
    def _autoRegistrationFiducialPointsFromEndpoints(endpointsMarkups) -> np.ndarray | None:
        if not endpointsMarkups or int(endpointsMarkups.GetNumberOfControlPoints()) < 2:
            return None

        entryPoint = [0.0, 0.0, 0.0]
        endpointPoint = [0.0, 0.0, 0.0]
        endpointsMarkups.GetNthControlPointPosition(0, entryPoint)
        endpointsMarkups.GetNthControlPointPosition(1, endpointPoint)
        endpointVector = np.array(endpointPoint, dtype=float)
        entryVector = np.array(entryPoint, dtype=float)
        axisVector = endpointVector - entryVector
        axisNorm = float(np.linalg.norm(axisVector))
        if axisNorm <= 1e-6:
            return None
        axisDirection = axisVector / axisNorm

        orthogonalSeed = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(axisDirection, orthogonalSeed))) > 0.95:
            orthogonalSeed = np.array([0.0, 1.0, 0.0], dtype=float)
        orthogonalDirection = np.cross(axisDirection, orthogonalSeed)
        orthogonalNorm = float(np.linalg.norm(orthogonalDirection))
        if orthogonalNorm <= 1e-6:
            orthogonalSeed = np.array([1.0, 0.0, 0.0], dtype=float)
            orthogonalDirection = np.cross(axisDirection, orthogonalSeed)
            orthogonalNorm = float(np.linalg.norm(orthogonalDirection))
            if orthogonalNorm <= 1e-6:
                return None
        orthogonalDirection /= orthogonalNorm

        midpoint = 0.5 * (endpointVector + entryVector)
        anchorOffsetMm = min(max(axisNorm * 0.25, 3.0), 12.0)
        anchorPoint = midpoint + (orthogonalDirection * anchorOffsetMm)
        return np.array([entryVector, endpointVector, anchorPoint], dtype=float)

    def _autoSeedRegistrationFiducialsFromEndpoints(self) -> bool:
        if not self._parameterNode:
            return False
        endpointsMarkups = self._parameterNode.endpointsMarkups
        nativeFiducials = self._parameterNode.nativeFiducials
        registeredFiducials = self._parameterNode.registeredFiducials
        if not endpointsMarkups or not nativeFiducials or not registeredFiducials:
            return False
        if self._markupsPointCount(nativeFiducials) > 0 or self._markupsPointCount(registeredFiducials) > 0:
            return False

        seededPoints = self._autoRegistrationFiducialPointsFromEndpoints(endpointsMarkups)
        if seededPoints is None:
            return False
        slicer.util.updateMarkupsControlPointsFromArray(nativeFiducials, seededPoints)
        slicer.util.updateMarkupsControlPointsFromArray(registeredFiducials, seededPoints)
        fiducialLabels = ("Auto Entry", "Auto Endpoint", "Auto Anchor")
        for markupsNode in (nativeFiducials, registeredFiducials):
            if hasattr(markupsNode, "SetAttribute"):
                markupsNode.SetAttribute("SV3D.AutoSeededFromEndpoints", "1")
            if hasattr(markupsNode, "SetNthControlPointLabel"):
                for controlPointIndex, controlPointLabel in enumerate(fiducialLabels):
                    markupsNode.SetNthControlPointLabel(controlPointIndex, controlPointLabel)
        return True

    @staticmethod
    def _configureNodeSelectorShowHidden(selector, showHidden: bool) -> None:
        if not selector:
            return
        if hasattr(selector, "setShowHidden"):
            selector.setShowHidden(showHidden)
            return
        if hasattr(selector, "showHidden"):
            selector.showHidden = bool(showHidden)

    @staticmethod
    def _configureNodeSelectorNodeTypes(selector, nodeTypes: Sequence[str]) -> None:
        if not selector:
            return
        resolvedNodeTypes = [str(nodeType) for nodeType in nodeTypes if str(nodeType)]
        if hasattr(selector, "setNodeTypes"):
            selector.setNodeTypes(resolvedNodeTypes)
            return
        if hasattr(selector, "nodeTypes"):
            selector.nodeTypes = resolvedNodeTypes

    @staticmethod
    def _selectorNodeMatchesClass(node, expectedClassName: str) -> bool:
        return bool(node and hasattr(node, "IsA") and node.IsA(expectedClassName))

    def _sanitizeSelectorNodeClass(
        self,
        selector,
        node,
        expectedClassName: str,
        selectorLabel: str,
    ):
        if not node:
            return None
        if self._selectorNodeMatchesClass(node, expectedClassName):
            return node
        logging.warning(
            "Ignoring invalid node type on selector '%s': expected %s, got %s (%s).",
            selectorLabel,
            expectedClassName,
            str(node.GetClassName() if hasattr(node, "GetClassName") else type(node).__name__),
            str(node.GetName() if hasattr(node, "GetName") else "unnamed"),
        )
        if selector and hasattr(selector, "setCurrentNode"):
            wasBlocked = bool(selector.blockSignals(True)) if hasattr(selector, "blockSignals") else False
            try:
                selector.setCurrentNode(None)
            except Exception:
                pass
            finally:
                if hasattr(selector, "blockSignals"):
                    selector.blockSignals(wasBlocked)
        return None

    def _populateSampleCaseComboBox(self) -> None:
        if not self.logic or not hasattr(self.ui, "sampleCaseComboBox"):
            return

        sampleCaseComboBox = self.ui.sampleCaseComboBox
        sampleCaseComboBox.clear()
        sampleScenes = self.logic.discoverBundledSampleScenes()
        for sampleSceneName, sampleScenePath in sampleScenes:
            sampleCaseComboBox.addItem(sampleSceneName, str(sampleScenePath))

        if sampleCaseComboBox.count == 0:
            sampleCaseComboBox.addItem("No sample scenes found", "")
            sampleCaseComboBox.enabled = False
        else:
            sampleCaseComboBox.enabled = True
            preferredSampleIndex = sampleCaseComboBox.findText("CRLM-1001")
            if preferredSampleIndex >= 0:
                sampleCaseComboBox.currentIndex = preferredSampleIndex

    def _selectedSampleCaseScenePath(self) -> str:
        if not hasattr(self.ui, "sampleCaseComboBox"):
            return ""
        sampleCaseComboBox = self.ui.sampleCaseComboBox
        if sampleCaseComboBox.count <= 0 or sampleCaseComboBox.currentIndex < 0:
            return ""
        return str(sampleCaseComboBox.itemData(sampleCaseComboBox.currentIndex) or "")

    def _syncBeginnerWidgetsFromParameterNode(self) -> None:
        if not self._parameterNode:
            return

        if self._caseFolderLineEdit and str(self._caseFolderLineEdit.text) != str(self._parameterNode.caseFolderPath or ""):
            self._caseFolderLineEdit.blockSignals(True)
            self._caseFolderLineEdit.text = str(self._parameterNode.caseFolderPath or "")
            self._caseFolderLineEdit.blockSignals(False)

        if self._mamSpinBox and not math.isclose(float(self._mamSpinBox.value), float(self._parameterNode.mamMm), abs_tol=1e-6):
            self._mamSpinBox.blockSignals(True)
            self._mamSpinBox.value = float(self._parameterNode.mamMm)
            self._mamSpinBox.blockSignals(False)

        if self._placeMultipleControlPointsCheckBox:
            targetChecked = bool(self._parameterNode.placeMultipleControlPoints)
            if bool(self._placeMultipleControlPointsCheckBox.checked) != targetChecked:
                self._placeMultipleControlPointsCheckBox.blockSignals(True)
                self._placeMultipleControlPointsCheckBox.checked = targetChecked
                self._placeMultipleControlPointsCheckBox.blockSignals(False)
            self._setPlaceModePersistence(targetChecked)
            self._setEndpointsPlaceWidgetMultiplePlacement(targetChecked)
        self._setEndpointsPlaceWidgetNode(self._parameterNode.endpointsMarkups)
        self._setObservedEndpointsMarkupsNode(self._parameterNode.endpointsMarkups)

        if self._coaxialTechniqueComboBox:
            targetTechnique = str(self._parameterNode.coaxialTechnique or "PullBack")
            targetIndex = self._coaxialTechniqueComboBox.findText(targetTechnique)
            if targetIndex >= 0 and targetIndex != self._coaxialTechniqueComboBox.currentIndex:
                self._coaxialTechniqueComboBox.blockSignals(True)
                self._coaxialTechniqueComboBox.currentIndex = targetIndex
                self._coaxialTechniqueComboBox.blockSignals(False)
        if self._coaxialSpareSpinBox:
            targetSpareMm = float(self._parameterNode.coaxialSpareMm)
            if not math.isclose(float(self._coaxialSpareSpinBox.value), targetSpareMm, abs_tol=1e-6):
                self._coaxialSpareSpinBox.blockSignals(True)
                self._coaxialSpareSpinBox.value = targetSpareMm
                self._coaxialSpareSpinBox.blockSignals(False)

        if self._geometryComboBox:
            targetGeometryId = str(self._parameterNode.selectedGeometryId or "")
            if not targetGeometryId and self._geometryComboBox.count > 0:
                targetGeometryId = str(self._geometryComboBox.itemData(0) or "")
            if targetGeometryId:
                targetIndex = self._geometryComboBox.findData(targetGeometryId)
                if targetIndex >= 0 and targetIndex != self._geometryComboBox.currentIndex:
                    self._geometryComboBox.blockSignals(True)
                    self._geometryComboBox.currentIndex = targetIndex
                    self._geometryComboBox.blockSignals(False)

        if self._planningModeComboBox:
            targetPlanningMode = str(self._parameterNode.trajectoryPlanningMode or "Single")
            targetIndex = self._planningModeComboBox.findData(targetPlanningMode)
            if targetIndex >= 0 and targetIndex != self._planningModeComboBox.currentIndex:
                self._planningModeComboBox.blockSignals(True)
                self._planningModeComboBox.currentIndex = targetIndex
                self._planningModeComboBox.blockSignals(False)
        if self._derivedTrajectoryCountSpinBox:
            targetCount = int(max(1, int(self._parameterNode.derivedTrajectoryCount)))
            if int(self._derivedTrajectoryCountSpinBox.value) != targetCount:
                self._derivedTrajectoryCountSpinBox.blockSignals(True)
                self._derivedTrajectoryCountSpinBox.value = targetCount
                self._derivedTrajectoryCountSpinBox.blockSignals(False)
        if self._derivedTrajectoryRadiusSpinBox:
            targetRadiusMm = float(max(0.0, float(self._parameterNode.derivedTrajectoryRadiusMm)))
            if not math.isclose(float(self._derivedTrajectoryRadiusSpinBox.value), targetRadiusMm, abs_tol=1e-6):
                self._derivedTrajectoryRadiusSpinBox.blockSignals(True)
                self._derivedTrajectoryRadiusSpinBox.value = targetRadiusMm
                self._derivedTrajectoryRadiusSpinBox.blockSignals(False)
        if self._derivedTrajectoryAngleOffsetSpinBox:
            targetAngleOffsetDeg = float(self._parameterNode.derivedTrajectoryAngleOffsetDeg)
            if not math.isclose(float(self._derivedTrajectoryAngleOffsetSpinBox.value), targetAngleOffsetDeg, abs_tol=1e-6):
                self._derivedTrajectoryAngleOffsetSpinBox.blockSignals(True)
                self._derivedTrajectoryAngleOffsetSpinBox.value = targetAngleOffsetDeg
                self._derivedTrajectoryAngleOffsetSpinBox.blockSignals(False)
        if self._includeMasterTrajectoryCheckBox:
            includeMaster = bool(self._parameterNode.includeMasterTrajectoryInArray)
            if bool(self._includeMasterTrajectoryCheckBox.checked) != includeMaster:
                self._includeMasterTrajectoryCheckBox.blockSignals(True)
                self._includeMasterTrajectoryCheckBox.checked = includeMaster
                self._includeMasterTrajectoryCheckBox.blockSignals(False)

        trajectorySummary = self._jsonSummary(self._parameterNode.trajectoryValidationSummaryJson)
        autoAdjustSummary = self._jsonSummary(self._parameterNode.endpointAutoAdjustSummaryJson)
        if self._trajectoryValidationStatusLabel:
            autoAdjustStatusText = str(autoAdjustSummary.get("statusText", autoAdjustSummary.get("StatusText", ""))).strip()
            self._setStatusLabelText(
                self._trajectoryValidationStatusLabel,
                autoAdjustStatusText or str(trajectorySummary.get("StatusText", "Trajectory not validated.")),
            )
        mamSummary = self._jsonSummary(self._parameterNode.mamAssessmentSummaryJson)
        if self._marginAssessmentStatusLabel:
            self._setStatusLabelText(
                self._marginAssessmentStatusLabel,
                str(mamSummary.get("StatusText", "MAM assessment not run.")),
            )
        coaxialSummary = self._jsonSummary(self._parameterNode.coaxialPlanSummaryJson)
        if self._coaxialStatusLabel:
            self._setStatusLabelText(
                self._coaxialStatusLabel,
                str(coaxialSummary.get("notes", "Coaxial plan not computed.")),
            )

    @staticmethod
    def _setWidgetVisible(widget, visible: bool) -> None:
        if widget and hasattr(widget, "setVisible"):
            widget.setVisible(bool(visible))

    def _setUiWidgetsVisible(self, widgetNames: Sequence[str], visible: bool, uiWidget=None) -> None:
        for widgetName in widgetNames:
            widget = getattr(self.ui, widgetName, None)
            if widget is None and uiWidget and hasattr(uiWidget, "findChild"):
                widget = uiWidget.findChild(qt.QWidget, str(widgetName))
            self._setWidgetVisible(widget, visible)

    def _setCollapsibleButtonsVisibleByText(self, uiWidget, buttonTexts: Sequence[str], visible: bool) -> None:
        if not uiWidget or not hasattr(uiWidget, "findChildren"):
            return
        targetTexts = {
            str(buttonText or "").strip()
            for buttonText in buttonTexts
            if str(buttonText or "").strip()
        }
        if not targetTexts:
            return
        for collapsibleButton in uiWidget.findChildren(ctk.ctkCollapsibleButton):
            collapsibleText = str(getattr(collapsibleButton, "text", "") or "").strip()
            if collapsibleText in targetTexts:
                self._setWidgetVisible(collapsibleButton, visible)

    def _applyBeginnerWorkflowMode(self, uiWidget) -> None:
        if not BEGINNER_WORKFLOW_MODE:
            return

        if uiWidget and uiWidget.layout() and not self._beginnerModeBannerLabel:
            self._beginnerModeBannerLabel = qt.QLabel(BEGINNER_WORKFLOW_BANNER_TEXT)
            self._beginnerModeBannerLabel.wordWrap = True
            self._beginnerModeBannerLabel.setStyleSheet(
                "QLabel { background: #f4f8ff; border: 1px solid #bccce6; border-radius: 4px; padding: 6px; }"
            )
            uiWidget.layout().insertWidget(0, self._beginnerModeBannerLabel)

        # Hide the original module sections; the beginner workflow reuses the widgets inside new step panels.
        self._setUiWidgetsVisible(
            (
                "inputsCollapsibleButton",
                "registrationCollapsibleButton",
                "evaluationCollapsibleButton",
                "cohortCollapsibleButton",
                "reproducibilityCollapsibleButton",
                "exportCollapsibleButton",
            ),
            False,
            uiWidget=uiWidget,
        )
        self._setCollapsibleButtonsVisibleByText(
            uiWidget,
            (
                "Probe Planning Inputs",
                "Tumor Registration",
                "Ablation Margin Evaluation",
                "Cohort / Study Evaluation",
                "Reproducibility Package",
                "Export",
            ),
            False,
        )
        self._setWidgetVisible(self._gitDashboardCollapsibleButton, False)
        self._setUiWidgetsVisible(
            (
                "recolorThresholdLowLabel",
                "recolorThresholdLowSpinBox",
                "recolorThresholdMidLabel",
                "recolorThresholdMidSpinBox",
                "recolorThresholdHighLabel",
                "recolorThresholdHighSpinBox",
                "recolorMarginsButton",
                "resetMarginColorsButton",
                "coordinationSectionLabel",
                "minInterProbeDistanceLabel",
                "minInterProbeDistanceSpinBox",
                "maxInterProbeDistanceLabel",
                "maxInterProbeDistanceSpinBox",
                "minEntryPointSpacingLabel",
                "minEntryPointSpacingSpinBox",
                "minTargetPointSpacingLabel",
                "minTargetPointSpacingSpinBox",
                "maxParallelAngleLabel",
                "maxParallelAngleSpinBox",
                "maxOverlapRedundancyLabel",
                "maxOverlapRedundancySpinBox",
                "requireAllPairsLabel",
                "requireAllProbePairsFeasibleCheckBox",
                "enableNoTouchCheckLabel",
                "enableNoTouchCheckBox",
                "enableInterProbeDistanceRuleLabel",
                "enableInterProbeDistanceRuleCheckBox",
                "enableEntrySpacingRuleLabel",
                "enableEntrySpacingRuleCheckBox",
                "enableTargetSpacingRuleLabel",
                "enableTargetSpacingRuleCheckBox",
                "enableAngleRuleLabel",
                "enableAngleRuleCheckBox",
                "enableOverlapRuleLabel",
                "enableOverlapRuleCheckBox",
                "evaluateProbeCoordinationButton",
                "probeCoordinationStatusLabel",
            ),
            False,
            uiWidget=uiWidget,
        )
        if hasattr(self.ui, "placeProbesButton"):
            self.ui.placeProbesButton.text = "Place Single Applicator"
        if hasattr(self.ui, "mergeTranslatedProbesButton"):
            self.ui.mergeTranslatedProbesButton.text = "Create Ablation Volume"
        if hasattr(self.ui, "evaluateMarginsButton"):
            self.ui.evaluateMarginsButton.text = "Evaluate MAM"
        if hasattr(self.ui, "createTrajectoryLinesButton"):
            self.ui.createTrajectoryLinesButton.text = "Preview Master Trajectory"
        if hasattr(self.ui, "registerTumorButton"):
            self.ui.registerTumorButton.text = "Apply Fiducial Registration"
        if hasattr(self.ui, "hardenTumorTransformButton"):
            self.ui.hardenTumorTransformButton.text = "Harden Registration"

    def _configureTooltips(self) -> None:
        tooltipsByWidgetName: dict[str, str] = {
            "sampleCaseComboBox": "Select a bundled sample scene and click Load to open it in the current Slicer session.",
            "loadSampleCaseButton": "Load the selected bundled sample scene without switching to the Sample Data module.",
            # Core inputs
            "probeSegmentationSelector": (
                "Select the reference probe/applicator template segmentation. This source geometry is duplicated and "
                "placed on each trajectory during 'Place Probes'. STL templates in Resources/Geometries are auto-loaded "
                "into this selector. The template is expected to be oriented along local -Z."
            ),
            "endpointsMarkupsSelector": (
                "Select trajectory endpoints as ordered entry/endpoint pairs with an even number of control points: "
                "entry1,endpoint1,entry2,endpoint2,..."
            ),
            "tumorSegmentationSelector": (
                "Select the tumor/target segmentation used for margin evaluation, coverage context, and no-touch checks."
            ),
            "riskStructuresSegmentationSelector": (
                "Optional: select structures-at-risk segmentation. If provided, distance-based safety summary tables are generated."
            ),
            "nativeFiducialsSelector": "Fiducials in native image space for rigid registration.",
            "registeredFiducialsSelector": "Fiducials in registered target space for rigid registration.",
            "tumorTransformSelector": (
                "Optional/advanced: select an explicit transform node for tumor geometry alignment. "
                "Used by registration and harden-transform steps."
            ),
            "createTrajectoryLinesOnPlacementCheckBox": "If enabled, trajectory lines are generated automatically after placing probes.",
            "clearPreviousGeneratedProbesCheckBox": "If enabled, previous generated probes/lines and owned derived outputs are cleared before placement.",
            # Workflow actions
            "placeProbesButton": "Place probe instances along each entry/endpoint-pair trajectory.",
            "createTrajectoryLinesButton": "Create or refresh line markups from entry/endpoint pairs.",
            "mergeTranslatedProbesButton": "Merge generated probe instances into one combined ablation segmentation.",
            "registerTumorButton": "Compute rigid transform from native to registered fiducials and apply it to the tumor segmentation.",
            "hardenTumorTransformButton": "Permanently harden the current tumor transform into the segmentation geometry.",
            "evaluateMarginsButton": "Run signed-margin analysis between tumor and ablation zone and refresh summary tables.",
            "recolorMarginsButton": "Recolor signed-distance values using the three configurable thresholds below.",
            "resetMarginColorsButton": "Restore original signed-distance values and default coloring on the margin model.",
            "combinedProbeSegmentationSelector": (
                "Output selector for the merged ablation segmentation (module-owned node). "
                "Usually auto-managed after 'Merge Translated Probes'."
            ),
            "outputMarginModelSelector": (
                "Output selector for the signed-margin model (module-owned node) generated by margin evaluation."
            ),
            "resultTableSelector": (
                "Output selector for raw signed-distance table values produced by margin analysis."
            ),
            "recolorThresholdLowSpinBox": (
                "Lower signed-distance recolor threshold (mm). Values below this threshold are assigned the low bucket."
            ),
            "recolorThresholdMidSpinBox": (
                "Middle signed-distance recolor threshold (mm). Used for intermediate margin-risk bucket coloring."
            ),
            "recolorThresholdHighSpinBox": (
                "Upper signed-distance recolor threshold (mm). Values above this are assigned the high/safe bucket."
            ),
            # Probe coordination
            "minInterProbeDistanceSpinBox": "Minimum allowed distance (mm) between probe trajectory line segments.",
            "maxInterProbeDistanceSpinBox": "Maximum allowed distance (mm) between probe trajectory line segments.",
            "minEntryPointSpacingSpinBox": "Minimum allowed spacing (mm) between probe entry points.",
            "minTargetPointSpacingSpinBox": "Minimum allowed spacing (mm) between probe target points.",
            "maxParallelAngleSpinBox": "Maximum angle (degrees) considered too-parallel when the angle rule is enabled.",
            "maxOverlapRedundancySpinBox": "Maximum allowed overlap redundancy percent for conservative overlap gating.",
            "requireAllProbePairsFeasibleCheckBox": "If enabled, any infeasible probe pair makes the coordination gate fail.",
            "enableNoTouchCheckBox": "If enabled, run conservative no-touch rule: all entry points must be outside tumor.",
            "enableInterProbeDistanceRuleCheckBox": "Enable/disable inter-probe distance constraints.",
            "enableEntrySpacingRuleCheckBox": "Enable/disable minimum entry-point spacing constraint.",
            "enableTargetSpacingRuleCheckBox": "Enable/disable minimum target-point spacing constraint.",
            "enableAngleRuleCheckBox": "Enable/disable near-parallel probe axis angle constraint.",
            "enableOverlapRuleCheckBox": "Enable/disable conservative overlap redundancy constraint.",
            "evaluateProbeCoordinationButton": "Evaluate pairwise probe coordination and plan-level gate status.",
            # Cohort/study
            "cohortStudyDefinitionPathLineEdit": "Path to cohort JSON definition (relative paths are resolved from the module folder).",
            "cohortExecutionModeComboBox": "Execution mode for cohort cases (scenario-driven or current working-plan context).",
            "cohortIncludeMarginMetricsCheckBox": "Include signed-margin metrics in case-level and aggregate cohort outputs.",
            "cohortIncludeSafetyMetricsCheckBox": "Include structures-at-risk safety metrics in cohort outputs.",
            "cohortIncludeCoverageMetricsCheckBox": "Include coverage metrics in cohort outputs when available.",
            "cohortIncludeFeasibilityMetricsCheckBox": "Include feasibility pass/fail metrics in cohort outputs.",
            "cohortIncludeCoordinationMetricsCheckBox": "Include probe-coordination gate metrics in cohort outputs.",
            "cohortIncludeVerificationMetricsCheckBox": "Include planned-vs-actual verification metrics in cohort outputs when present.",
            "cohortIncludeRecommendationMetricsCheckBox": "Include recommendation/composite-score context in cohort outputs.",
            "cohortMaxCasesSpinBox": "Maximum number of cohort cases to execute (0 runs all listed cases).",
            "runCohortEvaluationButton": "Run deterministic cohort batch evaluation and update cohort summary tables.",
            # Reproducibility package
            "packageModeComboBox": "Package scope preset. ReviewerSupplement is the default frozen bundle profile.",
            "includeBenchmarkArtifactsCheckBox": "Include benchmark definitions/catalog files and benchmark runtime outputs when available.",
            "includeScenarioRegistryCheckBox": "Include scenario registry and recommendation provenance exports when available.",
            "includeCohortStudyArtifactsCheckBox": "Include cohort definitions and cohort summary tables when present.",
            "includeStudyAnalyticsCheckBox": "Include study-analytics tables if generated in the current scene.",
            "includeReportsCheckBox": "Include report-oriented tables/JSON artifacts when available.",
            "includeCanonicalJsonCheckBox": "Include canonical JSON summaries and schema resources for interop/review.",
            "includeValidationResultsCheckBox": "Include validation and benchmark result tables when present.",
            "packageBaseNameLineEdit": "Base name for reproducibility package folder. Sequence suffix is added automatically.",
            "packageOutputDirectoryLineEdit": "Destination directory for reproducibility packages. Empty uses the Slicer temp folder.",
            "generateReproducibilityPackageButton": "Assemble deterministic reviewer/reproducibility package without mutating plan state.",
            "reproducibilityStatusLabel": "Status of the last reproducibility package run.",
            # Export
            "exportModeComboBox": "Choose export scope: current plan, selected scenario, or recommendation context.",
            "selectedExportScenarioIDLineEdit": "Scenario ID used when export mode is SelectedScenario.",
            "exportBaseNameLineEdit": "Base name for exported bundle folder. A deterministic sequence suffix is added.",
            "exportDirectoryLineEdit": "Destination directory for export bundles. Leave empty to use default temp/export location.",
            "includeWorkingPlanCheckBox": "Include current working-plan summary outputs in export.",
            "includeSelectedScenarioCheckBox": "Include selected scenario summary payload in export when available.",
            "includeScenarioComparisonCheckBox": "Include scenario comparison/delta/frontier tables when present.",
            "includeRecommendationOutputsCheckBox": "Include recommendation summary outputs when present.",
            "includeTrajectoryTablesCheckBox": "Include trajectory summary tables in export.",
            "includeSafetyTablesCheckBox": "Include safety distance summary tables in export.",
            "includeCoverageTablesCheckBox": "Include coverage summary tables in export when present.",
            "includeFeasibilityTablesCheckBox": "Include feasibility and gating tables in export when present.",
            "includeCoordinationTablesCheckBox": "Include probe coordination and no-touch tables in export.",
            "exportBundleButton": "Write a deterministic export bundle with JSON/CSV tables, metric snapshots, screenshots, scene package, and manifest.",
        }
        for widgetName, tooltipText in tooltipsByWidgetName.items():
            widget = getattr(self.ui, widgetName, None)
            if widget and hasattr(widget, "setToolTip"):
                widget.setToolTip(tooltipText)

    def _configureBeginnerTooltips(self) -> None:
        if not BEGINNER_WORKFLOW_MODE:
            return
        beginnerTooltipsByWidgetName: dict[str, str] = {
            "sampleCaseComboBox": "Pick a sample case first, then click Load.",
            "loadSampleCaseButton": "Load the selected sample scene.",
            "probeSegmentationSelector": "Choose the probe template geometry to place on each trajectory.",
            "endpointsMarkupsSelector": "Choose exactly two points: entry point first, applicator endpoint second.",
            "tumorSegmentationSelector": "Choose the target tumor segmentation for margin evaluation.",
            "riskStructuresSegmentationSelector": "Choose the segmentation that contains the critical structures to avoid.",
            "placeProbesButton": "Create probe instances from entry/endpoint pairs.",
            "mergeTranslatedProbesButton": "Union the placed probes into one combined ablation zone.",
            "evaluateMarginsButton": "Compute signed margins between the target tumor and the combined ablation zone.",
            "createTrajectoryLinesButton": "Preview the currently planned trajectory set (master only or master + derived array).",
            "registerTumorButton": "Apply fiducial-based rigid registration to the tumor segmentation.",
            "hardenTumorTransformButton": "Make the current tumor registration permanent.",
        }
        for widgetName, tooltipText in beginnerTooltipsByWidgetName.items():
            widget = getattr(self.ui, widgetName, None)
            if widget and hasattr(widget, "setToolTip"):
                widget.setToolTip(tooltipText)
        if self._browseCaseFolderButton:
            self._browseCaseFolderButton.setToolTip("Browse to a case folder containing one main .nrrd file and optional .seg.nrrd files.")
        if self._importCaseFolderButton:
            self._importCaseFolderButton.setToolTip("Import the selected case folder into the current Slicer scene.")
        if self._openSegmentEditorButton:
            self._openSegmentEditorButton.setToolTip("Open Slicer Segment Editor to create or refine tumor and critical-structure segments.")
        if self._geometryComboBox:
            self._geometryComboBox.setToolTip("Choose a predefined ablation geometry. Each entry also defines the active-element length.")
        if self._planningModeComboBox:
            self._planningModeComboBox.setToolTip("Single uses only the master trajectory. Multiple generates a deterministic parallel array.")
        if self._derivedTrajectoryCountSpinBox:
            self._derivedTrajectoryCountSpinBox.setToolTip(
                "Number of derived child trajectories distributed around the master (children only)."
            )
        if self._derivedTrajectoryRadiusSpinBox:
            self._derivedTrajectoryRadiusSpinBox.setToolTip("Radial offset in mm from the master trajectory to each child trajectory.")
        if self._derivedTrajectoryAngleOffsetSpinBox:
            self._derivedTrajectoryAngleOffsetSpinBox.setToolTip("Global angular offset in degrees applied to the derived trajectory ring.")
        if self._includeMasterTrajectoryCheckBox:
            self._includeMasterTrajectoryCheckBox.setToolTip("Include/exclude the master trajectory when placing the generated bundle.")
        if self._previewDerivedArrayButton:
            self._previewDerivedArrayButton.setToolTip(
                "Generate and preview deterministic derived trajectories after master trajectory validation."
            )
        if self._clearDerivedArrayButton:
            self._clearDerivedArrayButton.setToolTip("Clear derived trajectory preview lines and derived-bundle summary outputs.")
        if self._mamSpinBox:
            self._mamSpinBox.setToolTip("Minimal Ablative Margin in mm. Default is 10 mm.")
        if self._placeMultipleControlPointsCheckBox:
            self._placeMultipleControlPointsCheckBox.setToolTip(
                "Keep markup placement active after each click so you can place entry/endpoint points consecutively."
            )
        if self._endpointsPlaceWidget:
            self._endpointsPlaceWidget.setToolTip(
                "Place entry/endpoint control points directly from this module."
            )
        if self._validateTrajectoryButton:
            self._validateTrajectoryButton.setToolTip("Check whether the master trajectory intersects any critical-structure segment.")
        if self._autoAdjustEndpointButton:
            self._autoAdjustEndpointButton.setToolTip(
                "Available only after failed validation. Moves endpoint only (entry fixed), keeps trajectory length fixed, "
                f"and applies only if a zero-intersection endpoint is found inside tumor within {AUTO_ADJUST_MAX_ENDPOINT_SHIFT_MM:.1f} mm."
            )
        if self._lockMasterPlanButton:
            self._lockMasterPlanButton.setToolTip(
                "Lock the current master trajectory after successful MAM assessment and save its snapshot."
            )
        if self._resetMasterPlanButton:
            self._resetMasterPlanButton.setToolTip("Unlock the master plan and clear the saved lock/coaxial outputs.")
        if self._coaxialTechniqueComboBox:
            self._coaxialTechniqueComboBox.setToolTip("Choose how the coaxial sheath and applicator are coordinated.")
        if self._coaxialSpareSpinBox:
            self._coaxialSpareSpinBox.setToolTip(
                "Additional depth reserve in mm for push-through guidance (offset = active element + spare)."
            )
        if self._computeCoaxialPlanButton:
            self._computeCoaxialPlanButton.setToolTip("Derive the coaxial navigation target for the selected technique.")
        if hasattr(self.ui, "exportBaseNameLineEdit"):
            self.ui.exportBaseNameLineEdit.setToolTip(
                "Name prefix for the Step 7 package folder (screenshots, scene, tables, and planning summaries)."
            )
        if hasattr(self.ui, "exportDirectoryLineEdit"):
            self.ui.exportDirectoryLineEdit.setToolTip(
                "Destination directory for Step 7 packages. Empty uses the default Slicer temp export folder."
            )
        if self._browseExportDirectoryButton:
            self._browseExportDirectoryButton.setToolTip(
                "Browse and choose destination directory for the Step 7 export package."
            )
        if hasattr(self.ui, "exportBundleButton"):
            self.ui.exportBundleButton.setToolTip(
                "Export full planning package: JSON/CSV tables, metric snapshots, screenshots, and scene bundle."
            )

    def cleanup(self) -> None:
        self.removeObservers()

    def enter(self) -> None:
        if (
            self._parameterNode
            and self._parameterNodeGuiTag is not None
            and not self._sceneImportInProgress
            and not self._sceneIsBusy()
        ):
            self._updateButtonStates()
            return
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            self._disconnectParameterNodeGui()
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateButtonStates)
        self._setObservedEndpointsMarkupsNode(None)

    def _disconnectParameterNodeGui(self) -> None:
        if self._parameterNode and self._parameterNodeGuiTag is not None:
            try:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            except Exception:
                pass
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        self._sceneCloseInProgress = True
        self._step123TraceSignature = ""
        self._logStepTrace("Runtime", "Scene close started.")
        self.setParameterNode(None)

    def onSceneStartImport(self, caller, event) -> None:
        scene = slicer.mrmlScene
        isImporting = bool(scene.IsImporting()) if hasattr(scene, "IsImporting") else True
        # Ignore synthetic/import-like notifications that can occur during local batch processing.
        if not isImporting:
            self._logStepTrace("Runtime", "Scene import start event ignored (scene.IsImporting() is False).")
            return
        self._sceneImportInProgress = True
        self._step123TraceSignature = ""
        self._disconnectParameterNodeGui()
        self._logStepTrace("Runtime", "Scene import started.")

    def _sceneIsBusy(self) -> bool:
        if self._sceneCloseInProgress:
            return True
        scene = slicer.mrmlScene
        if hasattr(scene, "IsClosing") and bool(scene.IsClosing()):
            return True
        if hasattr(scene, "IsImporting") and bool(scene.IsImporting()):
            return True
        if hasattr(scene, "IsBatchProcessing") and bool(scene.IsBatchProcessing()):
            return True
        return False

    def _initializeParameterNodeWhenSceneReady(self) -> None:
        if not self.parent.isEntered:
            return
        if self._sceneImportInProgress or self._sceneIsBusy():
            self._traceStep123State("_initializeParameterNodeWhenSceneReady", force=False)
            return
        self.initializeParameterNode()

    def onSceneEndClose(self, caller, event) -> None:
        self._sceneCloseInProgress = False
        self._logStepTrace("Runtime", "Scene close finished.")
        qt.QTimer.singleShot(0, self._initializeParameterNodeWhenSceneReady)

    def onSceneEndImport(self, caller, event) -> None:
        scene = slicer.mrmlScene
        isImporting = bool(scene.IsImporting()) if hasattr(scene, "IsImporting") else False
        if isImporting:
            # End event may arrive while scene still reports importing; wait for the real completion.
            self._logStepTrace("Runtime", "Scene import end event deferred (scene.IsImporting() is still True).")
            return
        if not self._sceneImportInProgress:
            return
        self._sceneImportInProgress = False
        self._logStepTrace("Runtime", "Scene import finished.")
        qt.QTimer.singleShot(0, self._initializeParameterNodeWhenSceneReady)

    def initializeParameterNode(self) -> None:
        if not self.logic:
            return
        if self._sceneImportInProgress or self._sceneIsBusy():
            self._traceStep123State("initializeParameterNode(scene busy)", force=False)
            return
        self._logStepTrace("Runtime", "initializeParameterNode started.")
        templateNodes = self.logic.ensureReferenceProbeTemplatesLoaded()
        parameterNode = self.logic.getParameterNode()
        self._preloadNamedSceneInputs(parameterNode)
        resolvedReferenceProbeSegmentation = self.logic.resolveUsableReferenceProbeSegmentation(
            parameterNode.referenceProbeSegmentation
        )
        if resolvedReferenceProbeSegmentation is not parameterNode.referenceProbeSegmentation:
            parameterNode.referenceProbeSegmentation = resolvedReferenceProbeSegmentation
        if parameterNode.endpointsMarkups and not slicer.mrmlScene.IsNodePresent(parameterNode.endpointsMarkups):
            parameterNode.endpointsMarkups = None
        if parameterNode.referenceProbeSegmentation and not slicer.mrmlScene.IsNodePresent(parameterNode.referenceProbeSegmentation):
            parameterNode.referenceProbeSegmentation = None
        if not parameterNode.referenceProbeSegmentation and len(templateNodes) > 0:
            parameterNode.referenceProbeSegmentation = templateNodes[0]
        if not parameterNode.selectedGeometryId and self._geometryComboBox and self._geometryComboBox.count > 0:
            parameterNode.selectedGeometryId = str(self._geometryComboBox.itemData(0) or "")
        self.setParameterNode(parameterNode)
        self._traceStep123State("initializeParameterNode", force=True)

    def _preloadNamedSceneInputs(self, parameterNode: SurgicalVision3D_PlannerParameterNode) -> None:
        if not self.logic:
            return

        if parameterNode.criticalStructuresSegmentation and self.logic.isReferenceProbeTemplateSegmentation(
            parameterNode.criticalStructuresSegmentation
        ):
            parameterNode.criticalStructuresSegmentation = None
        if parameterNode.riskStructuresSegmentation and self.logic.isReferenceProbeTemplateSegmentation(
            parameterNode.riskStructuresSegmentation
        ):
            parameterNode.riskStructuresSegmentation = None

        if not parameterNode.tumorSegmentation:
            tumorSegmentation = self.logic.findFirstNodeByClassAndPreferredNames(
                "vtkMRMLSegmentationNode",
                DEFAULT_TUMOR_SEGMENTATION_NAMES,
            )
            if tumorSegmentation:
                parameterNode.tumorSegmentation = tumorSegmentation

        if not parameterNode.criticalStructuresSegmentation:
            criticalStructuresSegmentation = self.logic.findFirstNodeByClassAndPreferredNames(
                "vtkMRMLSegmentationNode",
                DEFAULT_CRITICAL_STRUCTURES_SEGMENTATION_NAMES,
            )
            if criticalStructuresSegmentation:
                parameterNode.criticalStructuresSegmentation = criticalStructuresSegmentation
            elif parameterNode.tumorSegmentation and self.logic.segmentationSegmentCount(parameterNode.tumorSegmentation) > 1:
                parameterNode.criticalStructuresSegmentation = parameterNode.tumorSegmentation
            else:
                for segmentationNode in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
                    if parameterNode.tumorSegmentation and segmentationNode.GetID() == parameterNode.tumorSegmentation.GetID():
                        continue
                    if self.logic.isReferenceProbeTemplateSegmentation(segmentationNode):
                        continue
                    parameterNode.criticalStructuresSegmentation = segmentationNode
                    break
        if not parameterNode.riskStructuresSegmentation and parameterNode.criticalStructuresSegmentation:
            parameterNode.riskStructuresSegmentation = parameterNode.criticalStructuresSegmentation

        endpointMarkups = self.logic.findFirstNodeByClassAndPreferredNames(
            "vtkMRMLMarkupsFiducialNode",
            DEFAULT_ENDPOINTS_MARKUPS_NAMES,
        )
        if endpointMarkups and (
            not parameterNode.endpointsMarkups
            or self.logic.isGenericDefaultFiducialsNodeName(parameterNode.endpointsMarkups.GetName())
        ):
            parameterNode.endpointsMarkups = endpointMarkups
        if not parameterNode.endpointsMarkups:
            parameterNode.endpointsMarkups = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode",
                slicer.mrmlScene.GenerateUniqueName("endpoints"),
            )
        elif self.logic.isGenericDefaultFiducialsNodeName(parameterNode.endpointsMarkups.GetName()):
            parameterNode.endpointsMarkups.SetName(slicer.mrmlScene.GenerateUniqueName("endpoints"))

        if parameterNode.nativeFiducials and not slicer.mrmlScene.IsNodePresent(parameterNode.nativeFiducials):
            parameterNode.nativeFiducials = None
        if parameterNode.registeredFiducials and not slicer.mrmlScene.IsNodePresent(parameterNode.registeredFiducials):
            parameterNode.registeredFiducials = None

        endpointsNodeID = parameterNode.endpointsMarkups.GetID() if parameterNode.endpointsMarkups else ""
        nativeFiducials = self.logic.findFirstNodeByClassAndPreferredNames(
            "vtkMRMLMarkupsFiducialNode",
            DEFAULT_NATIVE_FIDUCIAL_NAMES,
        )
        nativeNeedsReplacement = (
            not parameterNode.nativeFiducials
            or self.logic.isGenericDefaultFiducialsNodeName(parameterNode.nativeFiducials.GetName())
            or (endpointsNodeID and parameterNode.nativeFiducials.GetID() == endpointsNodeID)
        )
        if nativeFiducials and (
            not parameterNode.nativeFiducials
            or self.logic.isGenericDefaultFiducialsNodeName(parameterNode.nativeFiducials.GetName())
            or (endpointsNodeID and parameterNode.nativeFiducials.GetID() == endpointsNodeID)
        ) and nativeFiducials.GetID() != endpointsNodeID:
            parameterNode.nativeFiducials = nativeFiducials
        if nativeNeedsReplacement and (
            not parameterNode.nativeFiducials
            or (endpointsNodeID and parameterNode.nativeFiducials.GetID() == endpointsNodeID)
        ):
            parameterNode.nativeFiducials = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode",
                slicer.mrmlScene.GenerateUniqueName("NativeFiducials"),
            )
        elif parameterNode.nativeFiducials and self.logic.isGenericDefaultFiducialsNodeName(parameterNode.nativeFiducials.GetName()):
            parameterNode.nativeFiducials.SetName(slicer.mrmlScene.GenerateUniqueName("NativeFiducials"))

        nativeNodeID = parameterNode.nativeFiducials.GetID() if parameterNode.nativeFiducials else ""
        registeredFiducials = self.logic.findFirstNodeByClassAndPreferredNames(
            "vtkMRMLMarkupsFiducialNode",
            DEFAULT_REGISTERED_FIDUCIAL_NAMES,
        )
        registeredNeedsReplacement = (
            not parameterNode.registeredFiducials
            or self.logic.isGenericDefaultFiducialsNodeName(parameterNode.registeredFiducials.GetName())
            or (endpointsNodeID and parameterNode.registeredFiducials.GetID() == endpointsNodeID)
            or (nativeNodeID and parameterNode.registeredFiducials.GetID() == nativeNodeID)
        )
        if registeredFiducials and (
            not parameterNode.registeredFiducials
            or self.logic.isGenericDefaultFiducialsNodeName(parameterNode.registeredFiducials.GetName())
            or (endpointsNodeID and parameterNode.registeredFiducials.GetID() == endpointsNodeID)
            or (nativeNodeID and parameterNode.registeredFiducials.GetID() == nativeNodeID)
        ) and registeredFiducials.GetID() not in {endpointsNodeID, nativeNodeID}:
            parameterNode.registeredFiducials = registeredFiducials
        if registeredNeedsReplacement and (
            not parameterNode.registeredFiducials
            or parameterNode.registeredFiducials.GetID() in {endpointsNodeID, nativeNodeID}
        ):
            parameterNode.registeredFiducials = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode",
                slicer.mrmlScene.GenerateUniqueName("RegisteredFiducials"),
            )
        elif parameterNode.registeredFiducials and self.logic.isGenericDefaultFiducialsNodeName(parameterNode.registeredFiducials.GetName()):
            parameterNode.registeredFiducials.SetName(slicer.mrmlScene.GenerateUniqueName("RegisteredFiducials"))

    def setParameterNode(self, inputParameterNode: SurgicalVision3D_PlannerParameterNode | None) -> None:
        if self._parameterNode is inputParameterNode and self._parameterNodeGuiTag is not None:
            self._syncExportWidgetsFromParameterNode()
            self._syncCohortWidgetsFromParameterNode()
            self._syncReproducibilityWidgetsFromParameterNode()
            self._syncBeginnerWidgetsFromParameterNode()
            self._updateButtonStates()
            return

        if self._parameterNode:
            if self._parameterNodeGuiTag is not None:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateButtonStates)
        self._parameterNodeGuiTag = None
        self._setObservedEndpointsMarkupsNode(None)

        self._parameterNode = inputParameterNode

        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateButtonStates)
            self._setObservedEndpointsMarkupsNode(self._parameterNode.endpointsMarkups)
            self._autoSeedRegistrationFiducialsFromEndpoints()
            self._syncExportWidgetsFromParameterNode()
            self._syncCohortWidgetsFromParameterNode()
            self._syncReproducibilityWidgetsFromParameterNode()
            self._syncBeginnerWidgetsFromParameterNode()
            self._updateButtonStates()

    def _syncExportWidgetsFromParameterNode(self) -> None:
        if not self._parameterNode:
            return

        if hasattr(self.ui, "exportModeComboBox"):
            currentModeText = str(self._parameterNode.exportMode or "CurrentWorkingPlan")
            modeIndex = self.ui.exportModeComboBox.findText(currentModeText)
            if modeIndex >= 0:
                self.ui.exportModeComboBox.currentIndex = modeIndex
        if hasattr(self.ui, "selectedExportScenarioIDLineEdit"):
            self.ui.selectedExportScenarioIDLineEdit.text = str(self._parameterNode.selectedExportScenarioID or "")
        if hasattr(self.ui, "exportBaseNameLineEdit"):
            self.ui.exportBaseNameLineEdit.text = str(self._parameterNode.exportBaseName or "")
        if hasattr(self.ui, "exportDirectoryLineEdit"):
            self.ui.exportDirectoryLineEdit.text = str(self._parameterNode.lastExportDirectory or "")

    def _syncCohortWidgetsFromParameterNode(self) -> None:
        if not self._parameterNode:
            return

        if hasattr(self.ui, "cohortStudyDefinitionPathLineEdit"):
            self.ui.cohortStudyDefinitionPathLineEdit.text = str(self._parameterNode.cohortStudyDefinitionPath or "")
        if hasattr(self.ui, "cohortExecutionModeComboBox"):
            currentModeText = str(self._parameterNode.cohortExecutionMode or "ScenarioRegistry")
            modeIndex = self.ui.cohortExecutionModeComboBox.findText(currentModeText)
            if modeIndex >= 0:
                self.ui.cohortExecutionModeComboBox.currentIndex = modeIndex

    def _syncReproducibilityWidgetsFromParameterNode(self) -> None:
        if not self._parameterNode:
            return

        if hasattr(self.ui, "packageModeComboBox"):
            currentModeText = str(self._parameterNode.packageMode or "ReviewerSupplement")
            modeIndex = self.ui.packageModeComboBox.findText(currentModeText)
            if modeIndex >= 0:
                self.ui.packageModeComboBox.currentIndex = modeIndex
        if hasattr(self.ui, "packageBaseNameLineEdit"):
            self.ui.packageBaseNameLineEdit.text = str(self._parameterNode.packageBaseName or "")
        if hasattr(self.ui, "packageOutputDirectoryLineEdit"):
            self.ui.packageOutputDirectoryLineEdit.text = str(self._parameterNode.packageOutputDirectory or "")

    def _reconcileParameterNodeState(self) -> None:
        if not self.logic or not self._parameterNode:
            return

        existingProbeNodeIDs = self.logic.resolveExistingNodeIDs(self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs))
        existingLineNodeIDs = self.logic.resolveExistingNodeIDs(self.logic.deserializeNodeIDs(self._parameterNode.generatedTrajectoryLineIDs))
        serializedProbeNodeIDs = self.logic.serializeNodeIDs(existingProbeNodeIDs)
        serializedLineNodeIDs = self.logic.serializeNodeIDs(existingLineNodeIDs)
        if serializedProbeNodeIDs != self._parameterNode.generatedProbeNodeIDs:
            self._parameterNode.generatedProbeNodeIDs = serializedProbeNodeIDs
        if serializedLineNodeIDs != self._parameterNode.generatedTrajectoryLineIDs:
            self._parameterNode.generatedTrajectoryLineIDs = serializedLineNodeIDs
        if str(self._parameterNode.trajectoryPlanningMode or "") not in ("Single", "MultiTrajectoryArray"):
            self._parameterNode.trajectoryPlanningMode = "Single"
        self._parameterNode.derivedTrajectoryCount = max(1, int(self._parameterNode.derivedTrajectoryCount))
        self._parameterNode.derivedTrajectoryRadiusMm = max(0.0, float(self._parameterNode.derivedTrajectoryRadiusMm))
        self._parameterNode.derivedTrajectoryAngleOffsetDeg = float(self._parameterNode.derivedTrajectoryAngleOffsetDeg)
        if self._parameterNode.riskStructuresSegmentation and not self._parameterNode.criticalStructuresSegmentation:
            self._parameterNode.criticalStructuresSegmentation = self._parameterNode.riskStructuresSegmentation
        elif self._parameterNode.criticalStructuresSegmentation and not self._parameterNode.riskStructuresSegmentation:
            self._parameterNode.riskStructuresSegmentation = self._parameterNode.criticalStructuresSegmentation

        for nodeFieldName in (
            "referenceProbeSegmentation",
            "tumorSegmentation",
            "criticalStructuresSegmentation",
            "combinedProbeSegmentation",
            "outputMarginModel",
            "resultTable",
            "tumorTransform",
            "coaxialNavigationTarget",
            "riskStructuresSegmentation",
            "endpointsMarkups",
            "nativeFiducials",
            "registeredFiducials",
            "trajectorySummaryTable",
            "derivedTrajectorySummaryTable",
            "planSummaryTable",
            "marginThresholdSummaryTable",
            "structureSafetySummaryTable",
            "structureSafetyThresholdSummaryTable",
            "probeCoordinationConstraintSettingsTable",
            "probePairCoordinationSummaryTable",
            "probeCoordinationSummaryTable",
            "noTouchSummaryTable",
            "exportSummaryTable",
            "exportManifestPreviewTable",
            "cohortExecutionSummaryTable",
            "cohortCaseSummaryTable",
            "cohortAggregateMetricsTable",
            "cohortComparisonSummaryTable",
            "reproducibilityPackageSummaryTable",
            "reproducibilityManifestPreviewTable",
            "reproducibilityArtifactIndexTable",
            "masterTrajectoryValidationTable",
            "masterPlanSnapshotTable",
            "coaxialPlanTable",
            "derivedTrajectoryPreviewNode",
        ):
            node = getattr(self._parameterNode, nodeFieldName)
            if not node:
                continue
            try:
                nodeIsPresent = bool(slicer.mrmlScene.IsNodePresent(node))
            except Exception:
                nodeIsPresent = False
            if not nodeIsPresent:
                setattr(self._parameterNode, nodeFieldName, None)

        expectedNodeClassesByFieldName: dict[str, str] = {
            "referenceProbeSegmentation": "vtkMRMLSegmentationNode",
            "tumorSegmentation": "vtkMRMLSegmentationNode",
            "criticalStructuresSegmentation": "vtkMRMLSegmentationNode",
            "riskStructuresSegmentation": "vtkMRMLSegmentationNode",
            "endpointsMarkups": "vtkMRMLMarkupsFiducialNode",
            "nativeFiducials": "vtkMRMLMarkupsFiducialNode",
            "registeredFiducials": "vtkMRMLMarkupsFiducialNode",
            "tumorTransform": "vtkMRMLTransformNode",
            "derivedTrajectoryPreviewNode": "vtkMRMLMarkupsLineNode",
        }
        for nodeFieldName, expectedClassName in expectedNodeClassesByFieldName.items():
            node = getattr(self._parameterNode, nodeFieldName)
            if not node:
                continue
            try:
                nodeIsPresent = bool(slicer.mrmlScene.IsNodePresent(node))
            except Exception:
                nodeIsPresent = False
            if not nodeIsPresent:
                setattr(self._parameterNode, nodeFieldName, None)
                continue
            try:
                classMatches = bool(hasattr(node, "IsA") and node.IsA(expectedClassName))
            except Exception:
                classMatches = False
            if not classMatches:
                setattr(self._parameterNode, nodeFieldName, None)

    def _clearOwnedSafetyOutputs(self, clearReferences: bool = False) -> None:
        if not self.logic or not self._parameterNode:
            return

        if self.logic.removeNodeIfOwned(
            self._parameterNode.structureSafetySummaryTable,
            GENERATED_STRUCTURE_SAFETY_SUMMARY_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.structureSafetySummaryTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.structureSafetyThresholdSummaryTable,
            GENERATED_STRUCTURE_SAFETY_THRESHOLD_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.structureSafetyThresholdSummaryTable = None

    def _clearOwnedCoordinationOutputs(self, clearReferences: bool = False) -> None:
        if not self.logic or not self._parameterNode:
            return

        if self.logic.removeNodeIfOwned(
            self._parameterNode.probeCoordinationConstraintSettingsTable,
            GENERATED_PROBE_COORDINATION_SETTINGS_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.probeCoordinationConstraintSettingsTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.probePairCoordinationSummaryTable,
            GENERATED_PROBE_PAIR_COORDINATION_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.probePairCoordinationSummaryTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.probeCoordinationSummaryTable,
            GENERATED_PROBE_COORDINATION_SUMMARY_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.probeCoordinationSummaryTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.noTouchSummaryTable,
            GENERATED_NO_TOUCH_SUMMARY_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.noTouchSummaryTable = None
        if clearReferences and hasattr(self.ui, "probeCoordinationStatusLabel"):
            self.ui.probeCoordinationStatusLabel.text = "Probe coordination not evaluated."

    def _clearOwnedExportOutputs(self, clearReferences: bool = False) -> None:
        if not self.logic or not self._parameterNode:
            return

        if self.logic.removeNodeIfOwned(
            self._parameterNode.exportSummaryTable,
            GENERATED_EXPORT_SUMMARY_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.exportSummaryTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.exportManifestPreviewTable,
            GENERATED_EXPORT_MANIFEST_PREVIEW_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.exportManifestPreviewTable = None
        if clearReferences and hasattr(self.ui, "exportStatusLabel"):
            self.ui.exportStatusLabel.text = "No export run yet."
        if clearReferences and self._exportWorkflowStatusLabel:
            self._setStatusLabelText(self._exportWorkflowStatusLabel, "Blocked: complete Step 6 first.")

    def _clearOwnedCohortOutputs(self, clearReferences: bool = False) -> None:
        if not self.logic or not self._parameterNode:
            return

        if self.logic.removeNodeIfOwned(
            self._parameterNode.cohortExecutionSummaryTable,
            GENERATED_COHORT_EXECUTION_SUMMARY_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.cohortExecutionSummaryTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.cohortCaseSummaryTable,
            GENERATED_COHORT_CASE_SUMMARY_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.cohortCaseSummaryTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.cohortAggregateMetricsTable,
            GENERATED_COHORT_AGGREGATE_METRICS_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.cohortAggregateMetricsTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.cohortComparisonSummaryTable,
            GENERATED_COHORT_COMPARISON_SUMMARY_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.cohortComparisonSummaryTable = None
        if clearReferences and hasattr(self.ui, "cohortStatusLabel"):
            self.ui.cohortStatusLabel.text = "Cohort evaluation not run."

    def _clearOwnedBeginnerOutputs(
        self,
        clearReferences: bool = False,
        unlockPlan: bool = False,
        clearTrajectoryValidation: bool = True,
    ) -> None:
        if not self.logic or not self._parameterNode:
            return

        if clearTrajectoryValidation:
            if self.logic.removeNodeIfOwned(
                self._parameterNode.masterTrajectoryValidationTable,
                GENERATED_MASTER_TRAJECTORY_VALIDATION_TABLE_ATTRIBUTE,
            ) or clearReferences:
                self._parameterNode.masterTrajectoryValidationTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.masterPlanSnapshotTable,
            GENERATED_MASTER_PLAN_SNAPSHOT_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.masterPlanSnapshotTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.coaxialPlanTable,
            GENERATED_COAXIAL_PLAN_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.coaxialPlanTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.coaxialNavigationTarget,
            GENERATED_COAXIAL_TARGET_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.coaxialNavigationTarget = None
        if self.logic:
            self.logic.removeNodesByAttribute("vtkMRMLMarkupsLineNode", GENERATED_COAXIAL_LINE_ATTRIBUTE)

        if clearTrajectoryValidation:
            self._parameterNode.trajectoryValidationSummaryJson = "{}"
            self._parameterNode.endpointAutoAdjustSummaryJson = "{}"
        self._parameterNode.mamAssessmentSummaryJson = "{}"
        self._parameterNode.masterPlanSnapshotJson = "{}"
        self._parameterNode.coaxialPlanSummaryJson = "{}"
        if unlockPlan:
            self._parameterNode.masterTrajectoryLocked = False
            self._setMasterTrajectoryLocked(False)

        if clearReferences:
            if clearTrajectoryValidation and self._trajectoryValidationStatusLabel:
                self._setStatusLabelText(self._trajectoryValidationStatusLabel, "Trajectory not validated.")
            if self._marginAssessmentStatusLabel:
                self._setStatusLabelText(self._marginAssessmentStatusLabel, "MAM assessment not run.")
            if self._coaxialStatusLabel:
                self._setStatusLabelText(self._coaxialStatusLabel, "Coaxial plan not computed.")

    def _clearOwnedDerivedOutputs(self, clearReferences: bool = False, clearBeginnerOutputs: bool = True) -> None:
        if not self.logic or not self._parameterNode:
            return

        if self.logic.removeNodeIfOwned(
            self._parameterNode.trajectorySummaryTable,
            GENERATED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.trajectorySummaryTable = None
        if self.logic.removeNodeIfOwned(self._parameterNode.combinedProbeSegmentation, GENERATED_COMBINED_PROBE_ATTRIBUTE) or clearReferences:
            self._parameterNode.combinedProbeSegmentation = None
        if self.logic.removeNodeIfOwned(self._parameterNode.outputMarginModel, GENERATED_MARGIN_MODEL_ATTRIBUTE) or clearReferences:
            self._parameterNode.outputMarginModel = None
        if self.logic.removeNodeIfOwned(self._parameterNode.resultTable, GENERATED_RESULT_TABLE_ATTRIBUTE) or clearReferences:
            self._parameterNode.resultTable = None
        if self.logic.removeNodeIfOwned(self._parameterNode.planSummaryTable, GENERATED_PLAN_SUMMARY_TABLE_ATTRIBUTE) or clearReferences:
            self._parameterNode.planSummaryTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.marginThresholdSummaryTable,
            GENERATED_MARGIN_THRESHOLD_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.marginThresholdSummaryTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.derivedTrajectorySummaryTable,
            GENERATED_DERIVED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.derivedTrajectorySummaryTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.derivedTrajectoryPreviewNode,
            GENERATED_TRAJECTORY_LINE_ATTRIBUTE,
        ) or clearReferences:
            self._parameterNode.derivedTrajectoryPreviewNode = None
        self._parameterNode.derivedTrajectoryBundleSummaryJson = "{}"
        self._clearOwnedSafetyOutputs(clearReferences=clearReferences)
        self._clearOwnedCoordinationOutputs(clearReferences=clearReferences)
        self._clearOwnedCohortOutputs(clearReferences=clearReferences)
        if clearBeginnerOutputs:
            self._clearOwnedBeginnerOutputs(clearReferences=clearReferences)

    def _updateButtonStates(self, caller=None, event=None) -> None:
        if self._sceneImportInProgress or self._sceneIsBusy():
            self._traceStep123State("_updateButtonStates(scene busy)", force=False)
            self._updateGitDashboardButtonStates()
            return

        if not self._parameterNode:
            if hasattr(self.ui, "loadSampleCaseButton"):
                self.ui.loadSampleCaseButton.enabled = bool(self._selectedSampleCaseScenePath())
            for buttonName in (
                "placeProbesButton",
                "createTrajectoryLinesButton",
                "mergeTranslatedProbesButton",
                "registerTumorButton",
                "hardenTumorTransformButton",
                "evaluateMarginsButton",
                "recolorMarginsButton",
                "resetMarginColorsButton",
                "evaluateProbeCoordinationButton",
                "runCohortEvaluationButton",
                "generateReproducibilityPackageButton",
                "exportBundleButton",
            ):
                if hasattr(self.ui, buttonName):
                    getattr(self.ui, buttonName).enabled = False
            for widget in (
                self._browseCaseFolderButton,
                self._browseExportDirectoryButton,
                self._importCaseFolderButton,
                self._openSegmentEditorButton,
                self._planningModeComboBox,
                self._derivedTrajectoryCountSpinBox,
                self._derivedTrajectoryRadiusSpinBox,
                self._derivedTrajectoryAngleOffsetSpinBox,
                self._includeMasterTrajectoryCheckBox,
                self._previewDerivedArrayButton,
                self._clearDerivedArrayButton,
                self._validateTrajectoryButton,
                self._autoAdjustEndpointButton,
                self._lockMasterPlanButton,
                self._resetMasterPlanButton,
                self._coaxialSpareSpinBox,
                self._computeCoaxialPlanButton,
            ):
                if widget:
                    widget.enabled = False
            self._traceStep123State("_updateButtonStates(no parameter node)", force=False)
            self._updateGitDashboardButtonStates()
            return

        self._reconcileParameterNodeState()
        self._syncBeginnerWidgetsFromParameterNode()

        hasProbeAndEndpoints = bool(self._parameterNode.referenceProbeSegmentation and self._parameterNode.endpointsMarkups)
        generatedProbeNodeIDs = self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs) if self.logic else []
        generatedTrajectoryLineIDs = self.logic.deserializeNodeIDs(self._parameterNode.generatedTrajectoryLineIDs) if self.logic else []
        hasGeneratedProbes = len(generatedProbeNodeIDs) > 0
        hasSingleGeneratedProbe = len(generatedProbeNodeIDs) == 1
        hasGeneratedTrajectoryLines = len(generatedTrajectoryLineIDs) > 0
        hasCombinedProbe = self._parameterNode.combinedProbeSegmentation is not None
        hasTumor = self._parameterNode.tumorSegmentation is not None
        hasCriticalStructures = self._parameterNode.criticalStructuresSegmentation is not None
        nativeFiducialCount = self._markupsPointCount(self._parameterNode.nativeFiducials)
        registeredFiducialCount = self._markupsPointCount(self._parameterNode.registeredFiducials)
        hasRegistrationInputs = bool(
            self._parameterNode.tumorSegmentation
            and self._parameterNode.nativeFiducials
            and self._parameterNode.registeredFiducials
            and nativeFiducialCount >= 3
            and registeredFiducialCount >= 3
            and nativeFiducialCount == registeredFiducialCount
        )
        hasMarginModel = self._parameterNode.outputMarginModel is not None
        hasTumorTransform = bool(self._parameterNode.tumorSegmentation and self._parameterNode.tumorSegmentation.GetTransformNodeID())
        endpointControlPointCount = int(self._parameterNode.endpointsMarkups.GetNumberOfControlPoints()) if self._parameterNode.endpointsMarkups else 0
        hasEvenEndpointPairs = endpointControlPointCount >= 2 and endpointControlPointCount % 2 == 0
        hasSingleMasterTrajectory = endpointControlPointCount == 2
        isLocked = bool(self._parameterNode.masterTrajectoryLocked)
        trajectoryValidationSummary = self._jsonSummary(self._parameterNode.trajectoryValidationSummaryJson)
        mamAssessmentSummary = self._jsonSummary(self._parameterNode.mamAssessmentSummaryJson)
        masterPlanSnapshotSummary = self._jsonSummary(self._parameterNode.masterPlanSnapshotJson)
        coaxialPlanSummary = self._jsonSummary(self._parameterNode.coaxialPlanSummaryJson)
        derivedBundleSummary = self._jsonSummary(self._parameterNode.derivedTrajectoryBundleSummaryJson)
        trajectoryValidationPass = bool(trajectoryValidationSummary.get("TrajectoryPass", False))
        trajectoryValidationHasRun = bool(
            ("TrajectoryPass" in trajectoryValidationSummary)
            or self._parameterNode.masterTrajectoryValidationTable
        )
        trajectoryValidationFailed = bool(trajectoryValidationHasRun and not trajectoryValidationPass)
        mamValidationPass = bool(mamAssessmentSummary.get("MamPass", False))
        hasMasterPlanSnapshot = bool(masterPlanSnapshotSummary)
        isMultiTrajectoryArrayMode = self._isMultiTrajectoryArrayMode()

        if BEGINNER_WORKFLOW_MODE:
            singlePlacementReady = hasProbeAndEndpoints and hasSingleMasterTrajectory and not isLocked
            multiPlacementReady = (
                hasProbeAndEndpoints
                and hasSingleMasterTrajectory
                and trajectoryValidationPass
                and not isLocked
            )
            self.ui.placeProbesButton.enabled = multiPlacementReady if isMultiTrajectoryArrayMode else singlePlacementReady
            self.ui.createTrajectoryLinesButton.enabled = (
                hasSingleMasterTrajectory
                and not isLocked
                and (trajectoryValidationPass if isMultiTrajectoryArrayMode else True)
            )
            allowMultipleGeneratedProbes = bool(isMultiTrajectoryArrayMode)
        else:
            self.ui.placeProbesButton.enabled = hasProbeAndEndpoints and hasEvenEndpointPairs and not isLocked
            self.ui.createTrajectoryLinesButton.enabled = hasEvenEndpointPairs and not isLocked
            allowMultipleGeneratedProbes = True

        self.ui.mergeTranslatedProbesButton.enabled = (
            hasGeneratedProbes
            and (hasSingleGeneratedProbe or allowMultipleGeneratedProbes or not BEGINNER_WORKFLOW_MODE)
            and not isLocked
        )
        self.ui.registerTumorButton.enabled = hasRegistrationInputs
        self.ui.hardenTumorTransformButton.enabled = hasTumorTransform
        if self._segmentEditorStatusLabel:
            registrationBlockers: list[str] = []
            if not self._parameterNode.tumorSegmentation:
                registrationBlockers.append("select a tumor segmentation")
            if not self._parameterNode.nativeFiducials:
                registrationBlockers.append("select a native fiducials node")
            if not self._parameterNode.registeredFiducials:
                registrationBlockers.append("select a registered fiducials node")
            if self._parameterNode.nativeFiducials and self._parameterNode.registeredFiducials:
                if nativeFiducialCount != registeredFiducialCount:
                    registrationBlockers.append(
                        f"native/registered point counts must match (native={nativeFiducialCount}, registered={registeredFiducialCount})"
                    )
                if nativeFiducialCount < 3 or registeredFiducialCount < 3:
                    registrationBlockers.append(
                        f"at least 3 fiducials are required in each list (native={nativeFiducialCount}, registered={registeredFiducialCount})"
                    )
            if hasTumorTransform:
                self._setStatusLabelText(
                    self._segmentEditorStatusLabel,
                    "Registration applied. Click Harden Registration to bake the transform into the tumor segmentation.",
                )
            elif hasRegistrationInputs:
                self._setStatusLabelText(self._segmentEditorStatusLabel, "Ready: click Apply Fiducial Registration.")
            elif registrationBlockers:
                self._setStatusLabelText(
                    self._segmentEditorStatusLabel,
                    "Blocked: " + "; ".join(registrationBlockers) + ".",
                )
            else:
                self._setStatusLabelText(
                    self._segmentEditorStatusLabel,
                    "Segment the tumor and critical structures in Segment Editor, then return to this module.",
                )
        self.ui.evaluateMarginsButton.enabled = hasTumor and hasSingleMasterTrajectory and (hasCombinedProbe or hasGeneratedProbes) and not isLocked
        self.ui.recolorMarginsButton.enabled = hasMarginModel
        self.ui.resetMarginColorsButton.enabled = hasMarginModel
        if BEGINNER_WORKFLOW_MODE:
            self.ui.evaluateProbeCoordinationButton.enabled = (
                hasSingleMasterTrajectory
                and not isLocked
                and (trajectoryValidationPass if isMultiTrajectoryArrayMode else True)
            )
        else:
            self.ui.evaluateProbeCoordinationButton.enabled = hasEvenEndpointPairs and not isLocked
        if BEGINNER_WORKFLOW_MODE:
            scenarioRequired = False
            selectedScenarioID = ""
        else:
            exportModeText = str(self.ui.exportModeComboBox.currentText) if hasattr(self.ui, "exportModeComboBox") else str(self._parameterNode.exportMode)
            scenarioRequired = exportModeText == "SelectedScenario"
            selectedScenarioID = (
                str(self.ui.selectedExportScenarioIDLineEdit.text)
                if hasattr(self.ui, "selectedExportScenarioIDLineEdit")
                else str(self._parameterNode.selectedExportScenarioID)
            )
        exportBaseName = (
            str(self.ui.exportBaseNameLineEdit.text)
            if hasattr(self.ui, "exportBaseNameLineEdit")
            else str(self._parameterNode.exportBaseName)
        )
        cohortStudyDefinitionPath = (
            str(self.ui.cohortStudyDefinitionPathLineEdit.text)
            if hasattr(self.ui, "cohortStudyDefinitionPathLineEdit")
            else str(self._parameterNode.cohortStudyDefinitionPath)
        )
        packageBaseName = (
            str(self.ui.packageBaseNameLineEdit.text)
            if hasattr(self.ui, "packageBaseNameLineEdit")
            else str(self._parameterNode.packageBaseName)
        )
        exportInputReady = bool(exportBaseName.strip()) and (not scenarioRequired or bool(selectedScenarioID.strip()))
        beginnerExportReady = bool(isLocked and hasMasterPlanSnapshot)
        self.ui.exportBundleButton.enabled = exportInputReady and (beginnerExportReady if BEGINNER_WORKFLOW_MODE else True)
        self.ui.runCohortEvaluationButton.enabled = bool(cohortStudyDefinitionPath.strip())
        if hasattr(self.ui, "loadSampleCaseButton"):
            self.ui.loadSampleCaseButton.enabled = bool(self._selectedSampleCaseScenePath())
        if hasattr(self.ui, "generateReproducibilityPackageButton"):
            self.ui.generateReproducibilityPackageButton.enabled = bool(packageBaseName.strip())
        if self._browseCaseFolderButton:
            self._browseCaseFolderButton.enabled = True
        if self._browseExportDirectoryButton:
            self._browseExportDirectoryButton.enabled = True
        if self._importCaseFolderButton:
            self._importCaseFolderButton.enabled = bool(str(self._caseFolderLineEdit.text).strip()) if self._caseFolderLineEdit else False
        if self._openSegmentEditorButton:
            self._openSegmentEditorButton.enabled = True
        if self._geometryComboBox:
            self._geometryComboBox.enabled = not isLocked
        if self._planningModeComboBox:
            self._planningModeComboBox.enabled = not isLocked
        if self._derivedTrajectoryCountSpinBox:
            self._derivedTrajectoryCountSpinBox.enabled = (not isLocked) and isMultiTrajectoryArrayMode
        if self._derivedTrajectoryRadiusSpinBox:
            self._derivedTrajectoryRadiusSpinBox.enabled = (not isLocked) and isMultiTrajectoryArrayMode
        if self._derivedTrajectoryAngleOffsetSpinBox:
            self._derivedTrajectoryAngleOffsetSpinBox.enabled = (not isLocked) and isMultiTrajectoryArrayMode
        if self._includeMasterTrajectoryCheckBox:
            self._includeMasterTrajectoryCheckBox.enabled = (not isLocked) and isMultiTrajectoryArrayMode
        if self._previewDerivedArrayButton:
            self._previewDerivedArrayButton.enabled = (
                isMultiTrajectoryArrayMode
                and hasSingleMasterTrajectory
                and trajectoryValidationPass
                and not isLocked
            )
        if self._clearDerivedArrayButton:
            self._clearDerivedArrayButton.enabled = (
                not isLocked
                and (
                    hasGeneratedTrajectoryLines
                    or bool(derivedBundleSummary)
                    or bool(self._parameterNode.derivedTrajectorySummaryTable)
                )
            )
        if self._mamSpinBox:
            self._mamSpinBox.enabled = not isLocked
        if self._placeMultipleControlPointsCheckBox:
            self._placeMultipleControlPointsCheckBox.enabled = not isLocked
        if self._endpointsPlaceWidget:
            self._endpointsPlaceWidget.enabled = not isLocked
        if hasattr(self.ui, "endpointsMarkupsSelector"):
            self.ui.endpointsMarkupsSelector.enabled = not isLocked
        if self._validateTrajectoryButton:
            self._validateTrajectoryButton.enabled = hasSingleMasterTrajectory and hasCriticalStructures and not isLocked
        if self._autoAdjustEndpointButton:
            self._autoAdjustEndpointButton.enabled = (
                hasSingleMasterTrajectory
                and hasTumor
                and hasCriticalStructures
                and not isLocked
                and trajectoryValidationFailed
            )
        if hasattr(self.ui, "placeProbesButton") and BEGINNER_WORKFLOW_MODE:
            self.ui.placeProbesButton.text = "Place Trajectory Bundle" if isMultiTrajectoryArrayMode else "Place Single Applicator"
        if hasattr(self.ui, "createTrajectoryLinesButton") and BEGINNER_WORKFLOW_MODE:
            self.ui.createTrajectoryLinesButton.text = (
                "Preview Planned Trajectories"
                if isMultiTrajectoryArrayMode
                else "Preview Master Trajectory"
            )
        if self._lockMasterPlanButton:
            self._lockMasterPlanButton.enabled = (hasSingleMasterTrajectory and mamValidationPass and not isLocked)
        if self._resetMasterPlanButton:
            self._resetMasterPlanButton.enabled = isLocked
        if self._coaxialSpareSpinBox:
            self._coaxialSpareSpinBox.enabled = True
        if self._computeCoaxialPlanButton:
            self._computeCoaxialPlanButton.enabled = isLocked and hasMasterPlanSnapshot
        if self._coaxialStatusLabel:
            step6StatusText = "Coaxial plan not computed."
            if isLocked:
                coaxialNotes = str(coaxialPlanSummary.get("notes", "")).strip()
                if coaxialNotes:
                    step6StatusText = coaxialNotes
                elif hasMasterPlanSnapshot:
                    step6StatusText = "Ready: click Compute Coaxial Plan."
                else:
                    step6StatusText = "Blocked: lock is stale because the master plan snapshot is missing. Reset and lock again."
            else:
                blockedReasons: list[str] = []
                if not mamValidationPass:
                    mamStatusText = str(
                        mamAssessmentSummary.get("StatusText", "MAM assessment not run.")
                    ).strip()
                    blockedReasons.append(f"Step 5 MAM validation not passed ({mamStatusText})")
                if blockedReasons:
                    step6StatusText = "Blocked: " + "; ".join(blockedReasons) + "."
                else:
                    step6StatusText = "Ready: click Lock Master Plan."
            self._setStatusLabelText(self._coaxialStatusLabel, step6StatusText)
        if self._exportWorkflowStatusLabel:
            if not BEGINNER_WORKFLOW_MODE:
                step7StatusText = "Ready: click Export Bundle."
            elif not isLocked:
                step7StatusText = "Blocked: complete Step 6 lock first."
            elif not hasMasterPlanSnapshot:
                step7StatusText = "Blocked: lock snapshot is missing. Reset and lock again."
            elif not exportInputReady:
                step7StatusText = "Blocked: set export package name."
            else:
                step7StatusText = "Ready: export full planning package (data + metrics + screenshots + scene)."
            self._setStatusLabelText(self._exportWorkflowStatusLabel, step7StatusText)
        if self._derivedArrayStatusLabel:
            if not isMultiTrajectoryArrayMode:
                self._setStatusLabelText(self._derivedArrayStatusLabel, "Array disabled: planning mode is Single.")
            elif isLocked:
                self._setStatusLabelText(self._derivedArrayStatusLabel, "Blocked: reset lock before editing trajectory array.")
            elif not hasSingleMasterTrajectory:
                self._setStatusLabelText(self._derivedArrayStatusLabel, "Blocked: define exactly one master trajectory first.")
            elif not trajectoryValidationPass:
                self._setStatusLabelText(self._derivedArrayStatusLabel, "Blocked: validate master trajectory first.")
            elif bool(derivedBundleSummary):
                totalCount = int(derivedBundleSummary.get("totalTrajectoryCount", derivedBundleSummary.get("TotalTrajectoryCount", 0)))
                if totalCount <= 0:
                    totalCount = int(derivedBundleSummary.get("TrajectoryCount", 0))
                self._setStatusLabelText(
                    self._derivedArrayStatusLabel,
                    f"Preview generated: {max(totalCount, 0)} total trajectories.",
                )
            else:
                self._setStatusLabelText(self._derivedArrayStatusLabel, "Ready: preview derived array.")
        self._traceStep123State("_updateButtonStates", force=False)
        self._updateGitDashboardButtonStates()

    def _updateGitDashboardButtonStates(self) -> None:
        hasRepository = bool(self._gitRepositoryRoot)
        hasCommitMessage = bool(str(self._gitCommitMessageLineEdit.text).strip()) if self._gitCommitMessageLineEdit else False
        hasEntryMessage = bool(str(self._gitEntryLineEdit.text).strip()) if self._gitEntryLineEdit else False
        if self._gitRefreshButton:
            self._gitRefreshButton.enabled = hasRepository
        if self._gitStageAllButton:
            self._gitStageAllButton.enabled = hasRepository
        if self._gitCommitButton:
            self._gitCommitButton.enabled = hasRepository and hasCommitMessage
        if self._gitPushButton:
            self._gitPushButton.enabled = hasRepository
        if self._gitAddEntryButton:
            self._gitAddEntryButton.enabled = hasRepository and hasEntryMessage

    @staticmethod
    def _jsonSummary(serializedValue: str | None) -> dict[str, Any]:
        try:
            loadedValue = json.loads(str(serializedValue or "{}"))
        except Exception:
            return {}
        return loadedValue if isinstance(loadedValue, dict) else {}

    def _setMasterTrajectoryLocked(self, locked: bool) -> None:
        if not self._parameterNode or not self._parameterNode.endpointsMarkups:
            return
        endpointsNode = self._parameterNode.endpointsMarkups
        if hasattr(endpointsNode, "SetLocked"):
            endpointsNode.SetLocked(bool(locked))
        else:
            for pointIndex in range(endpointsNode.GetNumberOfControlPoints()):
                if hasattr(endpointsNode, "SetNthControlPointLocked"):
                    endpointsNode.SetNthControlPointLocked(pointIndex, bool(locked))

    def _selectedGeometryEntry(self) -> GeometryCatalogEntry | None:
        if not self.logic or not self._parameterNode:
            return None
        return self.logic.geometryCatalogEntryById(str(self._parameterNode.selectedGeometryId or ""))

    def _singleMasterTrajectory(self) -> ProbeTrajectory:
        if not self.logic or not self._parameterNode or not self._parameterNode.endpointsMarkups:
            raise RuntimeError("Master trajectory markups are not available.")
        trajectory = self.logic.extractSingleTrajectoryFromMarkups(self._parameterNode.endpointsMarkups)
        trajectory.role = "Master"
        trajectory.label = "Master"
        trajectory.angleDeg = None
        trajectory.radialOffsetMm = 0.0
        trajectory.derivedFromMaster = False
        return trajectory

    def _isMultiTrajectoryArrayMode(self) -> bool:
        if not self._parameterNode:
            return False
        return str(self._parameterNode.trajectoryPlanningMode or "Single") == "MultiTrajectoryArray"

    def _currentDerivedTrajectoryArrayConfig(self) -> DerivedTrajectoryArrayConfig:
        if not self._parameterNode:
            return DerivedTrajectoryArrayConfig()
        return DerivedTrajectoryArrayConfig(
            planningMode=str(self._parameterNode.trajectoryPlanningMode or "Single"),
            derivedTrajectoryCount=max(1, int(self._parameterNode.derivedTrajectoryCount)),
            radiusMm=max(0.0, float(self._parameterNode.derivedTrajectoryRadiusMm)),
            angleOffsetDeg=float(self._parameterNode.derivedTrajectoryAngleOffsetDeg),
            includeMasterTrajectory=bool(self._parameterNode.includeMasterTrajectoryInArray),
        )

    def _plannedTrajectoryBundle(self, requireValidatedMasterForArray: bool = True) -> list[ProbeTrajectory]:
        if not self.logic or not self._parameterNode or not self._parameterNode.endpointsMarkups:
            raise RuntimeError("Master trajectory markups are not available.")

        arrayConfig = self._currentDerivedTrajectoryArrayConfig()
        isMultiMode = arrayConfig.planningMode == "MultiTrajectoryArray"
        if not BEGINNER_WORKFLOW_MODE and not isMultiMode:
            return self.logic.extractTrajectoriesFromMarkups(self._parameterNode.endpointsMarkups, strictEven=True)

        masterTrajectory = self._singleMasterTrajectory()
        if not isMultiMode:
            return [masterTrajectory]

        if requireValidatedMasterForArray:
            validationSummary = self._jsonSummary(self._parameterNode.trajectoryValidationSummaryJson)
            if not bool(validationSummary.get("TrajectoryPass", False)):
                raise ValueError("Validate the master trajectory before generating a multi-trajectory array.")

        return self.logic.generateDerivedParallelTrajectories(
            masterTrajectory,
            derivedCount=int(arrayConfig.derivedTrajectoryCount),
            radiusMm=float(arrayConfig.radiusMm),
            angleOffsetDeg=float(arrayConfig.angleOffsetDeg),
            includeMaster=bool(arrayConfig.includeMasterTrajectory),
        )

    def _invalidateGeneratedPlanningOutputs(self, reasonText: str = "", invalidateMasterValidation: bool = False) -> None:
        if not self.logic or not self._parameterNode:
            return
        hasGeneratedNodes = bool(
            self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs)
            or self.logic.deserializeNodeIDs(self._parameterNode.generatedTrajectoryLineIDs)
        )
        hasDerivedSummary = bool(self._jsonSummary(self._parameterNode.derivedTrajectoryBundleSummaryJson))
        if (
            not hasGeneratedNodes
            and not hasDerivedSummary
            and not self._parameterNode.trajectorySummaryTable
            and not self._parameterNode.derivedTrajectorySummaryTable
            and not self._parameterNode.combinedProbeSegmentation
            and not self._parameterNode.planSummaryTable
            and not self._parameterNode.marginThresholdSummaryTable
        ):
            return

        self.logic.removeGeneratedProbeNodes()
        self.logic.removeGeneratedTrajectoryLines()
        self._parameterNode.generatedProbeNodeIDs = "[]"
        self._parameterNode.generatedTrajectoryLineIDs = "[]"
        self._clearOwnedDerivedOutputs(clearReferences=True, clearBeginnerOutputs=False)
        self._parameterNode.derivedTrajectoryPreviewNode = None
        self._parameterNode.derivedTrajectoryBundleSummaryJson = "{}"
        if invalidateMasterValidation:
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)
        if self._marginAssessmentStatusLabel and reasonText:
            self._setStatusLabelText(self._marginAssessmentStatusLabel, f"Ready: {reasonText}")

    @staticmethod
    def _formatRasPoint(pointRAS: Sequence[float]) -> str:
        return ",".join(f"{float(value):.3f}" for value in pointRAS)

    def _buildDeferredDerivedBundleSummary(
        self,
        trajectories: Sequence[ProbeTrajectory],
        statusText: str = "Bundle summary generated. Critical-structure evaluation is deferred.",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = [
            {
                "TrajectoryIndex": int(trajectory.trajectoryIndex + 1),
                "Role": str(trajectory.role or ""),
                "Source": "derived" if bool(trajectory.derivedFromMaster) else "master",
                "AngleDeg": float(trajectory.angleDeg) if trajectory.angleDeg is not None else float("nan"),
                "RadialOffsetMm": float(trajectory.radialOffsetMm),
                "EntryPointRAS": self._formatRasPoint(trajectory.entryPointRAS),
                "TargetPointRAS": self._formatRasPoint(trajectory.targetPointRAS),
                "IntersectsCriticalStructures": False,
                "MinDistanceToCriticalStructuresMm": float("nan"),
                "CheckedStructureCount": 0,
            }
            for trajectory in trajectories
        ]
        summary: dict[str, Any] = {
            "TrajectoryCount": int(len(trajectories)),
            "IntersectingTrajectoryCount": 0,
            "NonIntersectingTrajectoryCount": int(len(trajectories)),
            "MinDistanceToCriticalStructuresMm": float("nan"),
            "StatusText": str(statusText),
            "CriticalStructuresAvailable": bool(self._parameterNode and self._parameterNode.criticalStructuresSegmentation),
        }
        return rows, summary

    def _publishDerivedTrajectoryBundleSummary(
        self,
        trajectories: Sequence[ProbeTrajectory],
        evaluateAgainstCriticalStructures: bool = True,
    ) -> None:
        if not self.logic or not self._parameterNode:
            return

        if not self._isMultiTrajectoryArrayMode():
            if self.logic.removeNodeIfOwned(
                self._parameterNode.derivedTrajectorySummaryTable,
                GENERATED_DERIVED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
            ):
                self._parameterNode.derivedTrajectorySummaryTable = None
            else:
                self._parameterNode.derivedTrajectorySummaryTable = None
            self._parameterNode.derivedTrajectoryBundleSummaryJson = "{}"
            return

        if evaluateAgainstCriticalStructures:
            try:
                rows, validationSummary = self.logic.evaluateDerivedTrajectoryBundleAgainstCriticalStructures(
                    trajectories,
                    self._parameterNode.criticalStructuresSegmentation,
                    self._parameterNode.tumorSegmentation,
                )
            except Exception as exc:
                logging.exception("Derived bundle critical-structure evaluation failed; using deferred summary.")
                rows, validationSummary = self._buildDeferredDerivedBundleSummary(
                    trajectories,
                    statusText=f"Bundle summary generated. Critical-structure evaluation failed: {str(exc)}",
                )
        else:
            rows, validationSummary = self._buildDeferredDerivedBundleSummary(trajectories)
        arrayConfig = self._currentDerivedTrajectoryArrayConfig()
        summary = DerivedTrajectoryArraySummary(
            planningMode=arrayConfig.planningMode,
            masterIncluded=bool(arrayConfig.includeMasterTrajectory),
            derivedTrajectoryCount=int(arrayConfig.derivedTrajectoryCount),
            totalTrajectoryCount=int(len(trajectories)),
            radiusMm=float(arrayConfig.radiusMm),
            angleOffsetDeg=float(arrayConfig.angleOffsetDeg),
            notes=str(validationSummary.get("StatusText", "")),
        )
        summaryValues = asdict(summary)
        summaryValues.update(validationSummary)
        self._parameterNode.derivedTrajectoryBundleSummaryJson = json.dumps(summaryValues, sort_keys=True)

        summaryTable = self.logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            DERIVED_TRAJECTORY_SUMMARY_TABLE_NODE_NAME,
            GENERATED_DERIVED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
            self._parameterNode.derivedTrajectorySummaryTable,
        )
        self.logic.populateDerivedTrajectoryBundleSummaryTable(summaryTable, rows)
        self._parameterNode.derivedTrajectorySummaryTable = summaryTable

    def _applyDefaultCtAbdomenDisplay(self) -> None:
        if not self.logic:
            return
        primaryVolumeNode = self.logic.primaryScalarVolumeNode()
        if not primaryVolumeNode:
            return
        if not self.logic.applyCtAbdomenDisplayToVolume(primaryVolumeNode):
            return
        if hasattr(slicer.util, "setSliceViewerLayers"):
            try:
                slicer.util.setSliceViewerLayers(background=primaryVolumeNode, fit=True)
            except Exception:
                pass

    def _ensurePrimaryVolumeVisibleInSlices(self, fit: bool = False) -> None:
        if not self.logic or not hasattr(slicer.util, "setSliceViewerLayers"):
            return
        primaryVolumeNode = self.logic.primaryScalarVolumeNode()
        if not primaryVolumeNode:
            return
        self.logic.applyCtAbdomenDisplayToVolume(primaryVolumeNode)
        try:
            slicer.util.setSliceViewerLayers(background=primaryVolumeNode, fit=bool(fit))
        except Exception:
            pass

    def onCaseFolderPathChanged(self, _text=None) -> None:
        if self._parameterNode and self._caseFolderLineEdit:
            self._parameterNode.caseFolderPath = str(self._caseFolderLineEdit.text or "").strip()
        self._updateButtonStates()

    def onBrowseCaseFolderButton(self) -> None:
        selectedDirectory = qt.QFileDialog.getExistingDirectory(
            self.parent,
            "Select Case Folder",
            str(self._parameterNode.caseFolderPath or "") if self._parameterNode else "",
        )
        if not selectedDirectory:
            return
        if self._caseFolderLineEdit:
            self._caseFolderLineEdit.text = str(selectedDirectory)
        if self._parameterNode:
            self._parameterNode.caseFolderPath = str(selectedDirectory)
        self._updateButtonStates()

    def onBrowseExportDirectoryButton(self) -> None:
        currentDirectory = ""
        if hasattr(self.ui, "exportDirectoryLineEdit"):
            currentDirectory = str(self.ui.exportDirectoryLineEdit.text or "").strip()
        if not currentDirectory and self._parameterNode:
            currentDirectory = str(self._parameterNode.lastExportDirectory or "").strip()
        selectedDirectory = qt.QFileDialog.getExistingDirectory(
            self.parent,
            "Select Export Directory",
            currentDirectory,
        )
        if not selectedDirectory:
            return
        if hasattr(self.ui, "exportDirectoryLineEdit"):
            self.ui.exportDirectoryLineEdit.text = str(selectedDirectory)
        if self._parameterNode:
            self._parameterNode.lastExportDirectory = str(selectedDirectory)
        self._updateButtonStates()

    def onImportCaseFolderButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to import case folder."), waitCursor=True):
            if not self.logic:
                raise RuntimeError("Module logic is not initialized.")
            caseFolderPath = str(self._caseFolderLineEdit.text).strip() if self._caseFolderLineEdit else ""
            if not caseFolderPath:
                raise ValueError("Select a case folder before importing.")
            self._logStepTrace("Step1", f"Import case requested: {caseFolderPath}")
            self.logic.loadCaseFolder(caseFolderPath)
            self.initializeParameterNode()
            self._applyDefaultCtAbdomenDisplay()
            if self._parameterNode:
                self._parameterNode.caseFolderPath = caseFolderPath
                self._parameterNode.generatedProbeNodeIDs = "[]"
                self._parameterNode.generatedTrajectoryLineIDs = "[]"
            self.logic.removeGeneratedProbeNodes()
            self.logic.removeGeneratedTrajectoryLines()
            self._clearOwnedDerivedOutputs(clearReferences=True, clearBeginnerOutputs=False)
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)
            if self._segmentEditorStatusLabel:
                self._setStatusLabelText(
                    self._segmentEditorStatusLabel,
                    "Case imported. Review or refine segments in Segment Editor if needed.",
                )
            if self._parameterNode:
                tumorName = self._parameterNode.tumorSegmentation.GetName() if self._parameterNode.tumorSegmentation else "(none)"
                criticalName = (
                    self._parameterNode.riskStructuresSegmentation.GetName()
                    if self._parameterNode.riskStructuresSegmentation
                    else "(none)"
                )
                self._logStepTrace("Step1", f"Import completed. Tumor='{tumorName}', Critical structures='{criticalName}'")
            self._traceStep123State("onImportCaseFolderButton", force=True)
            self._updateButtonStates()

    def onOpenSegmentEditorButton(self) -> None:
        if self.logic and self._parameterNode:
            preparedNodeIDs: set[str] = set()
            for segmentationNode in (
                self._parameterNode.tumorSegmentation,
                self._parameterNode.criticalStructuresSegmentation,
                self._parameterNode.riskStructuresSegmentation,
            ):
                if not segmentationNode:
                    continue
                nodeID = segmentationNode.GetID()
                if nodeID in preparedNodeIDs:
                    continue
                preparedNodeIDs.add(nodeID)
                self.logic.prepareSegmentationForEditing(segmentationNode)
            tumorName = self._parameterNode.tumorSegmentation.GetName() if self._parameterNode.tumorSegmentation else "(none)"
            criticalName = (
                self._parameterNode.riskStructuresSegmentation.GetName()
                if self._parameterNode.riskStructuresSegmentation
                else "(none)"
            )
            self._logStepTrace("Step2", f"Open Segment Editor. Tumor='{tumorName}', Critical structures='{criticalName}'")
        slicer.util.selectModule("SegmentEditor")
        qt.QTimer.singleShot(0, self._configureSegmentEditorContextFromParameterNode)
        if self._segmentEditorStatusLabel:
            self._setStatusLabelText(
                self._segmentEditorStatusLabel,
                "Segment Editor opened. Registration stays blocked until tumor + native/registered fiducials "
                "are selected with matching counts and at least 3 points each.",
            )
        self._traceStep123State("onOpenSegmentEditorButton", force=True)

    def _configureSegmentEditorContextFromParameterNode(self) -> None:
        if not self.logic or not self._parameterNode:
            return
        try:
            segmentEditorWidget = slicer.util.getModuleWidget("SegmentEditor")
        except Exception:
            segmentEditorWidget = None
        if not segmentEditorWidget:
            return

        editor = getattr(segmentEditorWidget, "editor", None)
        if not editor:
            return

        preferredSegmentation = (
            self._parameterNode.tumorSegmentation
            or self._parameterNode.criticalStructuresSegmentation
            or self._parameterNode.riskStructuresSegmentation
        )
        if preferredSegmentation and hasattr(editor, "setSegmentationNode"):
            try:
                editor.setSegmentationNode(preferredSegmentation)
            except Exception:
                pass

        sourceVolume = self.logic.primaryScalarVolumeNode()
        if sourceVolume and hasattr(editor, "setSourceVolumeNode"):
            try:
                editor.setSourceVolumeNode(sourceVolume)
            except Exception:
                pass

    def onSelectedGeometryChanged(self, _index=None) -> None:
        if not self.logic or not self._parameterNode or not self._geometryComboBox:
            return
        if self._parameterNode.masterTrajectoryLocked:
            return
        selectedGeometryId = str(self._geometryComboBox.itemData(self._geometryComboBox.currentIndex) or "")
        self._parameterNode.selectedGeometryId = selectedGeometryId
        geometryEntry = self.logic.geometryCatalogEntryById(selectedGeometryId)
        if geometryEntry is not None:
            templateNode = self.logic.referenceProbeTemplateNodeForCatalogEntry(geometryEntry)
            if templateNode is not None:
                self._parameterNode.referenceProbeSegmentation = templateNode
        self.logic.removeGeneratedProbeNodes()
        self.logic.removeGeneratedTrajectoryLines()
        self._parameterNode.generatedProbeNodeIDs = "[]"
        self._parameterNode.generatedTrajectoryLineIDs = "[]"
        self._clearOwnedDerivedOutputs(clearReferences=True, clearBeginnerOutputs=False)
        self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True, clearTrajectoryValidation=False)
        self._updateButtonStates()

    def onTrajectoryPlanningModeChanged(self, _index=None) -> None:
        if not self._parameterNode or not self._planningModeComboBox:
            return
        selectedMode = str(self._planningModeComboBox.itemData(self._planningModeComboBox.currentIndex) or "Single")
        if selectedMode not in ("Single", "MultiTrajectoryArray"):
            selectedMode = "Single"
        if selectedMode != str(self._parameterNode.trajectoryPlanningMode or "Single"):
            self._parameterNode.trajectoryPlanningMode = selectedMode
            self._invalidateGeneratedPlanningOutputs("trajectory planning mode changed. Re-preview and re-place probes.")
        else:
            self._parameterNode.trajectoryPlanningMode = selectedMode
        self._logStepTrace("Step3", f"Trajectory planning mode set to '{selectedMode}'.")
        self._updateButtonStates()

    def onDerivedTrajectoryCountChanged(self, newValue: int) -> None:
        if not self._parameterNode:
            return
        updatedCount = int(max(1, newValue))
        if int(self._parameterNode.derivedTrajectoryCount) != updatedCount:
            self._parameterNode.derivedTrajectoryCount = updatedCount
            self._invalidateGeneratedPlanningOutputs("derived trajectory count changed. Re-preview and re-place probes.")
        else:
            self._parameterNode.derivedTrajectoryCount = updatedCount
        self._logStepTrace("Step3", f"Derived trajectory count set to {updatedCount}.")
        self._updateButtonStates()

    def onDerivedTrajectoryRadiusChanged(self, newValue: float) -> None:
        if not self._parameterNode:
            return
        updatedRadius = float(max(0.0, newValue))
        if not math.isclose(float(self._parameterNode.derivedTrajectoryRadiusMm), updatedRadius, abs_tol=1e-6):
            self._parameterNode.derivedTrajectoryRadiusMm = updatedRadius
            self._invalidateGeneratedPlanningOutputs("derived trajectory radius changed. Re-preview and re-place probes.")
        else:
            self._parameterNode.derivedTrajectoryRadiusMm = updatedRadius
        self._logStepTrace("Step3", f"Derived trajectory radius set to {updatedRadius:.1f} mm.")
        self._updateButtonStates()

    def onDerivedTrajectoryAngleOffsetChanged(self, newValue: float) -> None:
        if not self._parameterNode:
            return
        updatedOffset = float(newValue)
        if not math.isclose(float(self._parameterNode.derivedTrajectoryAngleOffsetDeg), updatedOffset, abs_tol=1e-6):
            self._parameterNode.derivedTrajectoryAngleOffsetDeg = updatedOffset
            self._invalidateGeneratedPlanningOutputs("derived angle offset changed. Re-preview and re-place probes.")
        else:
            self._parameterNode.derivedTrajectoryAngleOffsetDeg = updatedOffset
        self._logStepTrace("Step3", f"Derived trajectory angle offset set to {updatedOffset:.1f} deg.")
        self._updateButtonStates()

    def onIncludeMasterTrajectoryInArrayToggled(self, checked: bool) -> None:
        if not self._parameterNode:
            return
        includeMaster = bool(checked)
        if bool(self._parameterNode.includeMasterTrajectoryInArray) != includeMaster:
            self._parameterNode.includeMasterTrajectoryInArray = includeMaster
            self._invalidateGeneratedPlanningOutputs("bundle composition changed. Re-preview and re-place probes.")
        else:
            self._parameterNode.includeMasterTrajectoryInArray = includeMaster
        self._logStepTrace("Step3", f"Include master trajectory in array set to {includeMaster}.")
        self._updateButtonStates()

    def onMamValueChanged(self, newValue: float) -> None:
        if not self._parameterNode:
            return
        self._parameterNode.mamMm = float(newValue)
        if self._parameterNode.outputMarginModel:
            self._parameterNode.mamAssessmentSummaryJson = "{}"
            if self._marginAssessmentStatusLabel:
                self._setStatusLabelText(
                    self._marginAssessmentStatusLabel,
                    "MAM changed. Re-run MAM assessment to refresh colors and pass/fail.",
                )
        self._updateButtonStates()

    def onPlaceMultipleControlPointsToggled(self, checked: bool) -> None:
        if self._parameterNode:
            self._parameterNode.placeMultipleControlPoints = bool(checked)
        self._setPlaceModePersistence(bool(checked))
        self._setEndpointsPlaceWidgetMultiplePlacement(bool(checked))
        self._updateButtonStates()

    def _clearComputedCoaxialOutputs(self) -> None:
        if not self.logic or not self._parameterNode:
            return
        if self.logic.removeNodeIfOwned(
            self._parameterNode.coaxialPlanTable,
            GENERATED_COAXIAL_PLAN_TABLE_ATTRIBUTE,
        ):
            self._parameterNode.coaxialPlanTable = None
        if self.logic.removeNodeIfOwned(
            self._parameterNode.coaxialNavigationTarget,
            GENERATED_COAXIAL_TARGET_ATTRIBUTE,
        ):
            self._parameterNode.coaxialNavigationTarget = None
        self.logic.removeNodesByAttribute("vtkMRMLMarkupsLineNode", GENERATED_COAXIAL_LINE_ATTRIBUTE)
        self._parameterNode.coaxialPlanSummaryJson = "{}"

    def _invalidateCoaxialPlanWithReadyStatus(self, reasonText: str) -> None:
        if not self._parameterNode:
            return
        hadComputedPlan = bool(self._jsonSummary(self._parameterNode.coaxialPlanSummaryJson))
        if not hadComputedPlan and not self._parameterNode.coaxialPlanTable and not self._parameterNode.coaxialNavigationTarget:
            return
        self._clearComputedCoaxialOutputs()
        if self._coaxialStatusLabel:
            self._setStatusLabelText(self._coaxialStatusLabel, f"Ready: {reasonText}")

    def onCoaxialTechniqueChanged(self, _index=None) -> None:
        if not self._parameterNode or not self._coaxialTechniqueComboBox:
            return
        updatedTechnique = str(self._coaxialTechniqueComboBox.currentText or "PullBack")
        if updatedTechnique != str(self._parameterNode.coaxialTechnique or "PullBack"):
            self._parameterNode.coaxialTechnique = updatedTechnique
            self._invalidateCoaxialPlanWithReadyStatus("technique changed. Click Compute Coaxial Plan.")
        else:
            self._parameterNode.coaxialTechnique = updatedTechnique
        self._updateButtonStates()

    def onCoaxialSpareMmChanged(self, newValue: float) -> None:
        if not self._parameterNode:
            return
        updatedSpareMm = float(max(0.0, newValue))
        if not math.isclose(float(self._parameterNode.coaxialSpareMm), updatedSpareMm, abs_tol=1e-6):
            self._parameterNode.coaxialSpareMm = updatedSpareMm
            self._invalidateCoaxialPlanWithReadyStatus("spare changed. Click Compute Coaxial Plan.")
        else:
            self._parameterNode.coaxialSpareMm = updatedSpareMm
        self._updateButtonStates()

    def _runMasterTrajectoryValidation(
        self,
        trajectory: ProbeTrajectory | None = None,
    ) -> tuple[list[dict[str, float | int | bool | str]], dict[str, float | int | bool | str]]:
        if not self.logic or not self._parameterNode:
            raise RuntimeError("Module logic is not initialized.")
        evaluationTrajectory = trajectory if trajectory is not None else self._singleMasterTrajectory()
        validationRows, validationSummary = self.logic.evaluateMasterTrajectoryAgainstCriticalStructures(
            evaluationTrajectory,
            self._parameterNode.criticalStructuresSegmentation,
            self._parameterNode.tumorSegmentation,
        )
        validationTable = self.logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            MASTER_TRAJECTORY_VALIDATION_TABLE_NODE_NAME,
            GENERATED_MASTER_TRAJECTORY_VALIDATION_TABLE_ATTRIBUTE,
            self._parameterNode.masterTrajectoryValidationTable,
        )
        self.logic.populateMasterTrajectoryValidationTable(validationTable, validationRows)
        self._parameterNode.masterTrajectoryValidationTable = validationTable
        self._parameterNode.trajectoryValidationSummaryJson = json.dumps(validationSummary, sort_keys=True)
        if self._trajectoryValidationStatusLabel:
            self._setStatusLabelText(
                self._trajectoryValidationStatusLabel,
                str(validationSummary.get("StatusText", "Trajectory validation completed.")),
            )
        return validationRows, validationSummary

    def onValidateTrajectoryButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to validate the master trajectory."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            self._parameterNode.endpointAutoAdjustSummaryJson = "{}"
            self._logStepTrace("Step4", "Validating master trajectory against critical structures.")
            _validationRows, validationSummary = self._runMasterTrajectoryValidation()
            self._logStepTrace(
                "Step4",
                (
                    f"Validation completed. Pass={bool(validationSummary.get('TrajectoryPass', False))}, "
                    f"Intersected={int(validationSummary.get('IntersectedStructureCount', 0))}, "
                    f"Checked={int(validationSummary.get('CheckedStructureCount', 0))}"
                ),
            )
            self._updateButtonStates()

    def onAutoAdjustEndpointButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to auto-adjust the endpoint."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            if not self._parameterNode.endpointsMarkups:
                raise ValueError("Master trajectory markups are not available.")

            self._logStepTrace(
                "Step4",
                (
                    "Auto-adjust endpoint requested "
                    f"(maxShift={AUTO_ADJUST_MAX_ENDPOINT_SHIFT_MM:.1f} mm, step={AUTO_ADJUST_ENDPOINT_SHIFT_STEP_MM:.1f} mm, "
                    f"azimuth={AUTO_ADJUST_AZIMUTH_SAMPLE_COUNT})."
                ),
            )
            trajectory = self._singleMasterTrajectory()
            adjustResult = self.logic.autoAdjustMasterTrajectoryEndpoint(
                trajectory,
                self._parameterNode.criticalStructuresSegmentation,
                self._parameterNode.tumorSegmentation,
                maxEndpointShiftMm=AUTO_ADJUST_MAX_ENDPOINT_SHIFT_MM,
                shiftStepMm=AUTO_ADJUST_ENDPOINT_SHIFT_STEP_MM,
                azimuthSampleCount=AUTO_ADJUST_AZIMUTH_SAMPLE_COUNT,
            )
            autoAdjustSummary: dict[str, Any] = asdict(adjustResult)

            if adjustResult.applied and adjustResult.selectedTargetPointRAS is not None:
                adjustedTarget = np.array(adjustResult.selectedTargetPointRAS, dtype=float)
                self._parameterNode.endpointsMarkups.SetNthControlPointPosition(
                    0,
                    float(adjustedTarget[0]),
                    float(adjustedTarget[1]),
                    float(adjustedTarget[2]),
                )
                _validationRows, validationSummary = self._runMasterTrajectoryValidation()
                revalidationPass = bool(validationSummary.get("TrajectoryPass", False))
                minDistanceText = (
                    f"{float(adjustResult.selectedMinDistanceMm):.2f} mm"
                    if math.isfinite(float(adjustResult.selectedMinDistanceMm))
                    else "n/a"
                )
                statusText = (
                    f"Auto-adjust applied: endpoint shifted {float(adjustResult.selectedEndpointShiftMm):.1f} mm, "
                    f"min clearance {minDistanceText}. "
                    f"{str(validationSummary.get('StatusText', ''))}"
                ).strip()
                autoAdjustSummary["revalidatedTrajectoryPass"] = bool(revalidationPass)
                autoAdjustSummary["revalidationStatusText"] = str(validationSummary.get("StatusText", ""))
                autoAdjustSummary["statusText"] = statusText
                if self._trajectoryValidationStatusLabel:
                    self._setStatusLabelText(self._trajectoryValidationStatusLabel, statusText)
                self._logStepTrace(
                    "Step4",
                    (
                        f"Auto-adjust applied. ShiftMm={float(adjustResult.selectedEndpointShiftMm):.3f}, "
                        f"MinDistanceMm={float(adjustResult.selectedMinDistanceMm):.3f}, "
                        f"RevalidationPass={bool(revalidationPass)}, "
                        f"Checked={int(adjustResult.checkedCandidateCount)}, "
                        f"InsideTumor={int(adjustResult.insideTumorCandidateCount)}, "
                        f"ZeroIntersection={int(adjustResult.zeroIntersectionCandidateCount)}"
                    ),
                )
            else:
                statusText = str(adjustResult.statusText).strip() or (
                    f"No safe adjustment found within {AUTO_ADJUST_MAX_ENDPOINT_SHIFT_MM:.1f} mm; endpoint unchanged."
                )
                autoAdjustSummary["statusText"] = statusText
                if self._trajectoryValidationStatusLabel:
                    self._setStatusLabelText(self._trajectoryValidationStatusLabel, statusText)
                self._logStepTrace(
                    "Step4",
                    (
                        f"Auto-adjust not applied. Reason='{str(adjustResult.reason)}', "
                        f"Checked={int(adjustResult.checkedCandidateCount)}, "
                        f"InsideTumor={int(adjustResult.insideTumorCandidateCount)}, "
                        f"ZeroIntersection={int(adjustResult.zeroIntersectionCandidateCount)}"
                    ),
                )

            self._parameterNode.endpointAutoAdjustSummaryJson = json.dumps(autoAdjustSummary, sort_keys=True)
            self._updateButtonStates()

    def onLockMasterPlanButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to lock the master plan."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            trajectoryValidationSummary = self._jsonSummary(self._parameterNode.trajectoryValidationSummaryJson)
            mamAssessmentSummary = self._jsonSummary(self._parameterNode.mamAssessmentSummaryJson)
            if not bool(mamAssessmentSummary.get("MamPass", False)):
                raise ValueError("Run MAM assessment successfully before locking the plan.")

            trajectory = self._singleMasterTrajectory()
            geometryEntry = self._selectedGeometryEntry()
            if geometryEntry is None:
                raise ValueError("Select an ablation geometry before locking the plan.")
            tumorSegmentID, tumorSegmentName = self.logic.getPreferredSegmentInfo(
                self._parameterNode.tumorSegmentation,
                DEFAULT_TUMOR_SEGMENT_NAMES,
                "locked master plan snapshot",
            )
            firstDroppedPointRAS = None
            secondDroppedPointRAS = None
            if self._parameterNode.endpointsMarkups and int(self._parameterNode.endpointsMarkups.GetNumberOfControlPoints()) >= 2:
                firstPoint = [0.0, 0.0, 0.0]
                secondPoint = [0.0, 0.0, 0.0]
                self._parameterNode.endpointsMarkups.GetNthControlPointPosition(0, firstPoint)
                self._parameterNode.endpointsMarkups.GetNthControlPointPosition(1, secondPoint)
                firstDroppedPointRAS = tuple(float(value) for value in firstPoint)
                secondDroppedPointRAS = tuple(float(value) for value in secondPoint)
            snapshot = LockedMasterPlanSnapshot(
                geometryId=geometryEntry.geometryId,
                geometryDisplayName=geometryEntry.displayName,
                mamMm=float(self._parameterNode.mamMm),
                entryPointRAS=tuple(float(value) for value in trajectory.entryPointRAS),
                targetPointRAS=tuple(float(value) for value in trajectory.targetPointRAS),
                directionVector=tuple(float(value) for value in trajectory.directionVector),
                trajectoryLengthMm=float(trajectory.lengthMm),
                activeElementLengthMm=float(geometryEntry.activeElementLengthMm),
                trajectoryValidationPass=bool(trajectoryValidationSummary.get("TrajectoryPass", False)),
                marginValidationPass=True,
                lockedAtISO=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                tumorSegmentationID=self._parameterNode.tumorSegmentation.GetID() if self._parameterNode.tumorSegmentation else "",
                tumorSegmentID=tumorSegmentID,
                tumorSegmentName=tumorSegmentName,
                endpointsMarkupsID=self._parameterNode.endpointsMarkups.GetID() if self._parameterNode.endpointsMarkups else "",
                firstDroppedPointRAS=firstDroppedPointRAS,
                secondDroppedPointRAS=secondDroppedPointRAS,
            )
            snapshotTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                MASTER_PLAN_SNAPSHOT_TABLE_NODE_NAME,
                GENERATED_MASTER_PLAN_SNAPSHOT_TABLE_ATTRIBUTE,
                self._parameterNode.masterPlanSnapshotTable,
            )
            self.logic.populateKeyValueTable(snapshotTable, asdict(snapshot))
            self._parameterNode.masterPlanSnapshotTable = snapshotTable
            self._parameterNode.masterPlanSnapshotJson = json.dumps(asdict(snapshot), sort_keys=True)
            self._parameterNode.masterTrajectoryLocked = True
            self._setMasterTrajectoryLocked(True)
            if self._coaxialStatusLabel:
                self._setStatusLabelText(
                    self._coaxialStatusLabel,
                    "Master plan locked. Choose a technique and compute the coaxial plan.",
                )
            self._updateButtonStates()

    def onResetMasterPlanButton(self) -> None:
        if not self._parameterNode:
            return
        self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)
        self._updateButtonStates()

    def onComputeCoaxialPlanButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to compute the coaxial plan."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            snapshotValues = self._jsonSummary(self._parameterNode.masterPlanSnapshotJson)
            if not snapshotValues:
                raise ValueError("Lock the master plan before computing the coaxial plan.")
            selectedGeometryEntry = self._selectedGeometryEntry()
            snapshotGeometryId = str(snapshotValues.get("geometryId", "")).strip()
            snapshotGeometryEntry = (
                self.logic.geometryCatalogEntryById(snapshotGeometryId)
                if snapshotGeometryId
                else None
            )
            geometryEntry = snapshotGeometryEntry or selectedGeometryEntry
            if geometryEntry is None:
                geometryCatalogEntries = self.logic.loadGeometryCatalog()
                if len(geometryCatalogEntries) > 0:
                    geometryEntry = geometryCatalogEntries[0]

            snapshotActiveElementLengthMm = float(snapshotValues.get("activeElementLengthMm", 0.0) or 0.0)
            geometryActiveElementLengthMm = float(geometryEntry.activeElementLengthMm) if geometryEntry else 0.0
            # Prefer current catalog geometry values to avoid stale locked snapshots carrying old lengths.
            activeElementLengthMm = geometryActiveElementLengthMm if geometryActiveElementLengthMm > 0.0 else snapshotActiveElementLengthMm
            if activeElementLengthMm <= 0.0:
                raise ValueError("Active-element length is not available. Select a valid ablation geometry and relock the plan.")
            if (
                snapshotActiveElementLengthMm > 0.0
                and not math.isclose(snapshotActiveElementLengthMm, activeElementLengthMm, abs_tol=1e-3)
            ):
                logging.warning(
                    "Coaxial active-element mismatch (snapshot=%.1f mm, catalog=%.1f mm). Using %.1f mm.",
                    snapshotActiveElementLengthMm,
                    geometryActiveElementLengthMm,
                    activeElementLengthMm,
                )
            snapshotEntryPointRAS = tuple(float(value) for value in snapshotValues.get("entryPointRAS", (0.0, 0.0, 0.0)))
            snapshotTargetPointRAS = tuple(float(value) for value in snapshotValues.get("targetPointRAS", (0.0, 0.0, 0.0)))

            def _snapshotRasPoint(values: Any) -> tuple[float, float, float] | None:
                if not isinstance(values, (list, tuple)) or len(values) < 3:
                    return None
                try:
                    point = tuple(float(values[index]) for index in range(3))
                except Exception:
                    return None
                return point if np.all(np.isfinite(np.asarray(point, dtype=float))) else None

            firstDroppedPointRAS = _snapshotRasPoint(snapshotValues.get("firstDroppedPointRAS"))
            secondDroppedPointRAS = _snapshotRasPoint(snapshotValues.get("secondDroppedPointRAS"))
            if not firstDroppedPointRAS or not secondDroppedPointRAS:
                endpointsMarkupsID = str(snapshotValues.get("endpointsMarkupsID", "")).strip()
                endpointsMarkupsNode = slicer.mrmlScene.GetNodeByID(endpointsMarkupsID) if endpointsMarkupsID else None
                if (
                    endpointsMarkupsNode
                    and endpointsMarkupsNode.IsA("vtkMRMLMarkupsFiducialNode")
                    and int(endpointsMarkupsNode.GetNumberOfControlPoints()) >= 2
                ):
                    firstPoint = [0.0, 0.0, 0.0]
                    secondPoint = [0.0, 0.0, 0.0]
                    endpointsMarkupsNode.GetNthControlPointPosition(0, firstPoint)
                    endpointsMarkupsNode.GetNthControlPointPosition(1, secondPoint)
                    firstDroppedPointRAS = tuple(float(value) for value in firstPoint)
                    secondDroppedPointRAS = tuple(float(value) for value in secondPoint)

            # Coaxial distances are anchored to markups drop order: first point is applicator endpoint, second is entry.
            coaxialEntryPointRAS = snapshotEntryPointRAS
            coaxialTargetPointRAS = snapshotTargetPointRAS
            if firstDroppedPointRAS and secondDroppedPointRAS:
                coaxialTargetPointRAS = firstDroppedPointRAS
                coaxialEntryPointRAS = secondDroppedPointRAS

            coaxialDirectionVector = np.asarray(coaxialTargetPointRAS, dtype=float) - np.asarray(coaxialEntryPointRAS, dtype=float)
            coaxialTrajectoryLengthMm = float(np.linalg.norm(coaxialDirectionVector))
            if not math.isfinite(coaxialTrajectoryLengthMm) or coaxialTrajectoryLengthMm <= 1e-8:
                coaxialDirectionValues = tuple(float(value) for value in snapshotValues.get("directionVector", (0.0, 0.0, -1.0)))
                coaxialTrajectoryLengthMm = float(snapshotValues.get("trajectoryLengthMm", 0.0))
            else:
                coaxialDirectionValues = tuple((coaxialDirectionVector / coaxialTrajectoryLengthMm).tolist())
            snapshotTrajectory = ProbeTrajectory(
                entryPointRAS=coaxialEntryPointRAS,
                targetPointRAS=coaxialTargetPointRAS,
                directionVector=coaxialDirectionValues,
                lengthMm=float(coaxialTrajectoryLengthMm),
                trajectoryIndex=0,
                label="Master",
                role="Master",
                angleDeg=None,
                radialOffsetMm=0.0,
                derivedFromMaster=False,
            )
            trajectories: list[ProbeTrajectory]
            if self._isMultiTrajectoryArrayMode():
                arrayConfig = self._currentDerivedTrajectoryArrayConfig()
                trajectories = self.logic.generateDerivedParallelTrajectories(
                    snapshotTrajectory,
                    derivedCount=int(arrayConfig.derivedTrajectoryCount),
                    radiusMm=float(arrayConfig.radiusMm),
                    angleOffsetDeg=float(arrayConfig.angleOffsetDeg),
                    includeMaster=bool(arrayConfig.includeMasterTrajectory),
                )
            else:
                trajectories = [snapshotTrajectory]
            if len(trajectories) <= 0:
                raise ValueError("No trajectories are available for coaxial planning.")

            coaxialSpareMm = float(max(0.0, self._parameterNode.coaxialSpareMm))
            selectedTechnique = str(self._parameterNode.coaxialTechnique or "PullBack")
            coaxialRows: list[dict[str, Any]] = []
            lineSpecs: list[tuple[Sequence[float], Sequence[float], str]] = []
            navigationTargets: list[tuple[float, float, float]] = []
            firstCoaxialSummary: CoaxialPlanSummary | None = None
            for trajectory in trajectories:
                coaxialSummary = self.logic.computeCoaxialPlanFromTrajectory(
                    trajectory,
                    selectedTechnique,
                    activeElementLengthMm,
                    coaxialSpareMm,
                )
                if firstCoaxialSummary is None:
                    firstCoaxialSummary = coaxialSummary
                lineStartPointRAS = (
                    trajectory.targetPointRAS
                    if str(coaxialSummary.technique or "") == "PushThrough"
                    else trajectory.entryPointRAS
                )
                roleName = str(trajectory.role or f"Trajectory {int(trajectory.trajectoryIndex) + 1:02d}")
                coaxialRows.append(
                    {
                        "TrajectoryIndex": int(trajectory.trajectoryIndex + 1),
                        "Role": roleName,
                        "Source": "derived" if bool(trajectory.derivedFromMaster) else "master",
                        "Technique": str(coaxialSummary.technique or selectedTechnique),
                        "ActiveElementLengthMm": float(coaxialSummary.activeElementLengthMm),
                        "SpareMm": float(coaxialSummary.spareMm),
                        "PushThroughOffsetMm": float(coaxialSummary.pushThroughOffsetMm),
                        "EntryPointRAS": self._formatRasPoint(trajectory.entryPointRAS),
                        "TargetPointRAS": self._formatRasPoint(trajectory.targetPointRAS),
                        "NavigationTargetRAS": self._formatRasPoint(coaxialSummary.navigationTargetRAS),
                    }
                )
                lineSpecs.append((lineStartPointRAS, coaxialSummary.navigationTargetRAS, roleName))
                navigationTargets.append(tuple(float(value) for value in coaxialSummary.navigationTargetRAS))

            coaxialTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                COAXIAL_PLAN_TABLE_NODE_NAME,
                GENERATED_COAXIAL_PLAN_TABLE_ATTRIBUTE,
                self._parameterNode.coaxialPlanTable,
            )
            if len(coaxialRows) == 1 and firstCoaxialSummary is not None:
                self.logic.populateKeyValueTable(coaxialTable, asdict(firstCoaxialSummary))
            else:
                self.logic.populateCoaxialPlanTable(coaxialTable, coaxialRows)
            self._parameterNode.coaxialPlanTable = coaxialTable
            if len(coaxialRows) == 1 and firstCoaxialSummary is not None:
                coaxialSummaryPayload = asdict(firstCoaxialSummary)
                coaxialSummaryPayload["planningMode"] = str(self._parameterNode.trajectoryPlanningMode or "Single")
                coaxialSummaryPayload["trajectoryCount"] = 1
                coaxialSummaryPayload["rows"] = coaxialRows
            else:
                coaxialSummaryPayload = {
                    "planningMode": str(self._parameterNode.trajectoryPlanningMode or "Single"),
                    "trajectoryCount": int(len(coaxialRows)),
                    "technique": selectedTechnique,
                    "activeElementLengthMm": float(activeElementLengthMm),
                    "spareMm": float(coaxialSpareMm),
                    "pushThroughOffsetMm": float(activeElementLengthMm + coaxialSpareMm),
                    "notes": (
                        f"{selectedTechnique} coaxial plan computed for {len(coaxialRows)} trajectories."
                    ),
                    "rows": coaxialRows,
                }
            self._parameterNode.coaxialPlanSummaryJson = json.dumps(coaxialSummaryPayload, sort_keys=True)
            coaxialTargetNode = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLMarkupsFiducialNode",
                COAXIAL_NAVIGATION_TARGET_NODE_NAME,
                GENERATED_COAXIAL_TARGET_ATTRIBUTE,
                self._parameterNode.coaxialNavigationTarget,
            )
            if len(navigationTargets) <= 0:
                raise ValueError("No coaxial navigation targets were computed.")
            slicer.util.updateMarkupsControlPointsFromArray(
                coaxialTargetNode,
                np.array(navigationTargets, dtype=float),
            )
            if hasattr(coaxialTargetNode, "SetNthControlPointLabel"):
                for rowIndex, rowValues in enumerate(coaxialRows):
                    coaxialTargetNode.SetNthControlPointLabel(rowIndex, f"{str(rowValues.get('Role', '')).strip()} Nav")
            coaxialTargetNode.SetLocked(True)
            self._parameterNode.coaxialNavigationTarget = coaxialTargetNode
            self.logic.createOrUpdateCoaxialLines(lineSpecs)
            self._logStepTrace(
                "Step6",
                (
                    f"Coaxial plan computed for {len(coaxialRows)} trajectory(ies) "
                    f"(mode='{self._parameterNode.trajectoryPlanningMode or 'Single'}', "
                    f"technique='{selectedTechnique}', spareMm={coaxialSpareMm:.1f})."
                ),
            )
            if self._coaxialStatusLabel:
                self._setStatusLabelText(
                    self._coaxialStatusLabel,
                    str(coaxialSummaryPayload.get("notes", "Coaxial plan computed.")),
                )
            self._updateButtonStates()

    def onLoadSampleCaseButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to load sample case."), waitCursor=True):
            if not self.logic:
                raise RuntimeError("Module logic is not initialized.")
            selectedScenePath = self._selectedSampleCaseScenePath().strip()
            if not selectedScenePath:
                raise ValueError("Select a sample case scene before loading.")
            self._logStepTrace("Step1", f"Load bundled sample requested: {selectedScenePath}")
            self.logic.loadBundledSampleScene(selectedScenePath)
            # Scene load triggers module-scene events, but run one more initialization pass to refresh selectors.
            self.initializeParameterNode()
            self._applyDefaultCtAbdomenDisplay()
            if self._parameterNode:
                self._parameterNode.caseFolderPath = ""
                self._parameterNode.generatedProbeNodeIDs = "[]"
                self._parameterNode.generatedTrajectoryLineIDs = "[]"
            self.logic.removeGeneratedProbeNodes()
            self.logic.removeGeneratedTrajectoryLines()
            self._clearOwnedDerivedOutputs(clearReferences=True, clearBeginnerOutputs=False)
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)
            if self._parameterNode:
                tumorName = self._parameterNode.tumorSegmentation.GetName() if self._parameterNode.tumorSegmentation else "(none)"
                criticalName = (
                    self._parameterNode.riskStructuresSegmentation.GetName()
                    if self._parameterNode.riskStructuresSegmentation
                    else "(none)"
                )
                self._logStepTrace("Step1", f"Sample load completed. Tumor='{tumorName}', Critical structures='{criticalName}'")
            self._traceStep123State("onLoadSampleCaseButton", force=True)
            self._updateButtonStates()

    def onPlaceProbesButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to place probes."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            if self._parameterNode.masterTrajectoryLocked:
                raise ValueError("Reset the locked master plan before changing the applicator placement.")

            if not self._parameterNode.referenceProbeSegmentation:
                raise ValueError("Select a reference probe segmentation before placing probes.")
            if not self._parameterNode.endpointsMarkups:
                raise ValueError("Select an endpoint markups node before placing probes.")
            controlPointCount = int(self._parameterNode.endpointsMarkups.GetNumberOfControlPoints())
            if controlPointCount == 0:
                raise ValueError("Endpoint markups node has no control points.")
            if BEGINNER_WORKFLOW_MODE and controlPointCount != 2:
                raise ValueError(
                    f"Beginner workflow expects exactly 2 control points, but found {controlPointCount}. "
                    "Place one entry point and one applicator endpoint (entry first)."
                )
            if controlPointCount % 2 != 0:
                raise ValueError(
                    f"Endpoint markups has {controlPointCount} control points. Add one more point to complete entry/endpoint pairs."
                )
            resolvedReferenceProbeSegmentation = self.logic.resolveUsableReferenceProbeSegmentation(
                self._parameterNode.referenceProbeSegmentation
            )
            if not resolvedReferenceProbeSegmentation:
                raise ValueError("No usable reference probe segmentation is available.")
            if resolvedReferenceProbeSegmentation is not self._parameterNode.referenceProbeSegmentation:
                self._parameterNode.referenceProbeSegmentation = resolvedReferenceProbeSegmentation

            existingProbeNodeIDs = self.logic.resolveExistingNodeIDs(
                self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs)
            )
            trajectories = self._plannedTrajectoryBundle(requireValidatedMasterForArray=True)
            if len(trajectories) == 0:
                raise ValueError("No trajectories are available for placement.")
            self._logStepTrace(
                "Step5",
                (
                    f"Place trajectory bundle requested (count={len(trajectories)}, "
                    f"mode='{self._parameterNode.trajectoryPlanningMode or 'Single'}', "
                    f"clearPrevious={bool(self._parameterNode.clearPreviousGeneratedProbes)})."
                ),
            )
            if self._parameterNode.clearPreviousGeneratedProbes:
                keepGeneratedReferenceProbeNodeID = None
                referenceProbeIsPresent = False
                if resolvedReferenceProbeSegmentation:
                    try:
                        referenceProbeIsPresent = bool(slicer.mrmlScene.IsNodePresent(resolvedReferenceProbeSegmentation))
                    except Exception:
                        referenceProbeIsPresent = False
                if (
                    resolvedReferenceProbeSegmentation
                    and referenceProbeIsPresent
                    and str(resolvedReferenceProbeSegmentation.GetAttribute(GENERATED_PROBE_ATTRIBUTE) or "") == "1"
                ):
                    keepGeneratedReferenceProbeNodeID = resolvedReferenceProbeSegmentation.GetID()
                    self._logStepTrace(
                        "Step5",
                        (
                            "Preserving selected generated reference probe during cleanup "
                            f"(node='{self._nodeDisplayName(resolvedReferenceProbeSegmentation)}')."
                        ),
                    )
                self.logic.removeGeneratedProbeNodes(keepNodeID=keepGeneratedReferenceProbeNodeID)
                self.logic.removeGeneratedTrajectoryLines()
                self._clearOwnedDerivedOutputs(clearReferences=True, clearBeginnerOutputs=False)
                self._parameterNode.generatedProbeNodeIDs = "[]"
                self._parameterNode.generatedTrajectoryLineIDs = "[]"
                existingProbeNodeIDs = []

            generatedProbeNodeIDs = self.logic.placeProbeInstances(resolvedReferenceProbeSegmentation, trajectories)
            if self._parameterNode.clearPreviousGeneratedProbes:
                trackedProbeNodeIDs = generatedProbeNodeIDs
            else:
                trackedProbeNodeIDs = self.logic.mergeNodeIDLists(existingProbeNodeIDs, generatedProbeNodeIDs)
            self._parameterNode.generatedProbeNodeIDs = self.logic.serializeNodeIDs(trackedProbeNodeIDs)

            # Probe placement invalidates previously merged/evaluated module-owned outputs.
            self._clearOwnedDerivedOutputs(clearReferences=True, clearBeginnerOutputs=False)
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True, clearTrajectoryValidation=False)

            if self._parameterNode.createTrajectoryLinesOnPlacement:
                if self._isMultiTrajectoryArrayMode():
                    generatedLineNodeIDs = self.logic.createOrUpdateDerivedTrajectoryPreview(trajectories, clearExisting=True)
                else:
                    generatedLineNodeIDs = self.logic.createTrajectoryLines(trajectories, clearExisting=True)
                self._parameterNode.generatedTrajectoryLineIDs = self.logic.serializeNodeIDs(generatedLineNodeIDs)
                self._parameterNode.derivedTrajectoryPreviewNode = (
                    slicer.mrmlScene.GetNodeByID(generatedLineNodeIDs[0])
                    if len(generatedLineNodeIDs) > 0
                    else None
                )
            else:
                self._parameterNode.generatedTrajectoryLineIDs = "[]"
                self._parameterNode.derivedTrajectoryPreviewNode = None

            trajectorySummaryTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                TRAJECTORY_SUMMARY_TABLE_NODE_NAME,
                GENERATED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
                self._parameterNode.trajectorySummaryTable,
            )
            trajectoryMetrics = self.logic.computeTrajectoryMetrics(trajectories)
            self.logic.populateTrajectorySummaryTable(trajectorySummaryTable, trajectoryMetrics)
            self._parameterNode.trajectorySummaryTable = trajectorySummaryTable
            self._publishDerivedTrajectoryBundleSummary(
                trajectories,
                evaluateAgainstCriticalStructures=EVALUATE_DERIVED_BUNDLE_INLINE_ON_PLACEMENT,
            )

            self._logStepTrace(
                "Step5",
                (
                    f"Placement completed (placed={len(generatedProbeNodeIDs)}, "
                    f"previewLines={len(self.logic.deserializeNodeIDs(self._parameterNode.generatedTrajectoryLineIDs))})."
                ),
            )
            self._updateButtonStates()

    def onCreateTrajectoryLinesButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to create trajectory lines."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            if self._parameterNode.masterTrajectoryLocked:
                raise ValueError("Reset the locked master plan before editing the master trajectory.")
            if not self._parameterNode.endpointsMarkups:
                raise ValueError("Select an endpoint markups node before creating trajectory lines.")
            controlPointCount = int(self._parameterNode.endpointsMarkups.GetNumberOfControlPoints())
            if controlPointCount == 0:
                raise ValueError("Endpoint markups node has no control points.")
            if BEGINNER_WORKFLOW_MODE and controlPointCount != 2:
                raise ValueError(
                    f"Beginner workflow expects exactly 2 control points, but found {controlPointCount}. "
                    "Place one entry point and one applicator endpoint (entry first)."
                )
            if controlPointCount % 2 != 0:
                raise ValueError(
                    f"Endpoint markups has {controlPointCount} control points. Add one more point to complete entry/endpoint pairs."
                )
            self._logStepTrace(
                "Step3",
                f"Preview trajectories requested (points={controlPointCount}, mode='{self._parameterNode.trajectoryPlanningMode or 'Single'}').",
            )
            trajectories = self._plannedTrajectoryBundle(requireValidatedMasterForArray=True)
            if self._isMultiTrajectoryArrayMode():
                generatedLineNodeIDs = self.logic.createOrUpdateDerivedTrajectoryPreview(trajectories, clearExisting=True)
            else:
                generatedLineNodeIDs = self.logic.createTrajectoryLines(trajectories, clearExisting=True)
            self._parameterNode.generatedTrajectoryLineIDs = self.logic.serializeNodeIDs(generatedLineNodeIDs)
            self._parameterNode.derivedTrajectoryPreviewNode = (
                slicer.mrmlScene.GetNodeByID(generatedLineNodeIDs[0])
                if len(generatedLineNodeIDs) > 0
                else None
            )
            self._publishDerivedTrajectoryBundleSummary(trajectories, evaluateAgainstCriticalStructures=False)
            self._ensurePrimaryVolumeVisibleInSlices(fit=False)
            self._logStepTrace("Step3", f"Preview generated trajectory lines: {len(generatedLineNodeIDs)}")
            self._traceStep123State("onCreateTrajectoryLinesButton", force=True)
            self._updateButtonStates()

    def onPreviewDerivedArrayButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to preview derived trajectory array."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            if not self._isMultiTrajectoryArrayMode():
                raise ValueError("Switch planning mode to 'Multiple trajectories' before previewing a derived array.")
            arrayConfig = self._currentDerivedTrajectoryArrayConfig()
            self._logStepTrace(
                "Step3",
                (
                    "Preview derived array requested "
                    f"(derivedCount={int(arrayConfig.derivedTrajectoryCount)}, "
                    f"radiusMm={float(arrayConfig.radiusMm):.1f}, "
                    f"angleOffsetDeg={float(arrayConfig.angleOffsetDeg):.1f}, "
                    f"includeMaster={bool(arrayConfig.includeMasterTrajectory)})."
                ),
            )

            trajectories = self._plannedTrajectoryBundle(requireValidatedMasterForArray=True)
            if len(trajectories) == 0:
                raise ValueError("No trajectories are available to preview.")

            generatedLineNodeIDs = self.logic.createOrUpdateDerivedTrajectoryPreview(trajectories, clearExisting=True)
            self._parameterNode.generatedTrajectoryLineIDs = self.logic.serializeNodeIDs(generatedLineNodeIDs)
            firstPreviewNode = (
                slicer.mrmlScene.GetNodeByID(generatedLineNodeIDs[0])
                if len(generatedLineNodeIDs) > 0
                else None
            )
            self._parameterNode.derivedTrajectoryPreviewNode = firstPreviewNode
            self._publishDerivedTrajectoryBundleSummary(trajectories, evaluateAgainstCriticalStructures=False)
            self._ensurePrimaryVolumeVisibleInSlices(fit=False)
            self._logStepTrace("Step3", f"Preview derived array generated trajectory lines: {len(generatedLineNodeIDs)}")
            self._traceStep123State("onPreviewDerivedArrayButton", force=True)
            self._updateButtonStates()

    def onClearDerivedArrayButton(self) -> None:
        if not self.logic or not self._parameterNode:
            return
        self.logic.removeGeneratedTrajectoryLines()
        self._parameterNode.generatedTrajectoryLineIDs = "[]"
        if self.logic.removeNodeIfOwned(
            self._parameterNode.derivedTrajectorySummaryTable,
            GENERATED_DERIVED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
        ):
            self._parameterNode.derivedTrajectorySummaryTable = None
        else:
            self._parameterNode.derivedTrajectorySummaryTable = None
        self._parameterNode.derivedTrajectoryPreviewNode = None
        self._parameterNode.derivedTrajectoryBundleSummaryJson = "{}"
        if self._derivedArrayStatusLabel:
            self._setStatusLabelText(self._derivedArrayStatusLabel, "Array not generated.")
        self._logStepTrace("Step3", "Derived array preview cleared.")
        self._traceStep123State("onClearDerivedArrayButton", force=True)
        self._updateButtonStates()

    def onMergeTranslatedProbesButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to merge translated probes."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            if self._parameterNode.masterTrajectoryLocked:
                raise ValueError("Reset the locked master plan before changing the ablation volume.")
            generatedProbeNodeIDs = self.logic.resolveExistingNodeIDs(
                self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs)
            )
            if len(generatedProbeNodeIDs) == 0:
                raise ValueError("No generated probe nodes are available. Click 'Place Probes' first.")
            combinedProbeNode = self.logic.mergeProbeInstances(generatedProbeNodeIDs, self._parameterNode.combinedProbeSegmentation)
            self._parameterNode.combinedProbeSegmentation = combinedProbeNode

            # Merged ablation geometry changed, margin outputs are now stale.
            self.logic.removeNodeIfOwned(self._parameterNode.outputMarginModel, GENERATED_MARGIN_MODEL_ATTRIBUTE)
            self.logic.removeNodeIfOwned(self._parameterNode.resultTable, GENERATED_RESULT_TABLE_ATTRIBUTE)
            self.logic.removeNodeIfOwned(self._parameterNode.planSummaryTable, GENERATED_PLAN_SUMMARY_TABLE_ATTRIBUTE)
            self.logic.removeNodeIfOwned(
                self._parameterNode.marginThresholdSummaryTable,
                GENERATED_MARGIN_THRESHOLD_TABLE_ATTRIBUTE,
            )
            self._clearOwnedSafetyOutputs(clearReferences=True)
            self._clearOwnedCoordinationOutputs(clearReferences=True)
            self._parameterNode.outputMarginModel = None
            self._parameterNode.resultTable = None
            self._parameterNode.planSummaryTable = None
            self._parameterNode.marginThresholdSummaryTable = None
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True, clearTrajectoryValidation=False)
            self._updateButtonStates()

    def onRegisterTumorButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to register tumor fiducials."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            nativeCount = self._markupsPointCount(self._parameterNode.nativeFiducials)
            registeredCount = self._markupsPointCount(self._parameterNode.registeredFiducials)
            self._logStepTrace(
                "Step2",
                f"Apply fiducial registration requested (nativePts={nativeCount}, registeredPts={registeredCount}).",
            )
            transformNode = self.logic.registerTumorToFiducials(
                self._parameterNode.tumorSegmentation,
                self._parameterNode.nativeFiducials,
                self._parameterNode.registeredFiducials,
                self._parameterNode.tumorTransform,
            )
            self._parameterNode.tumorTransform = transformNode
            self._logStepTrace("Step2", f"Registration completed. Transform='{self._nodeDisplayName(transformNode)}'")
            if self._segmentEditorStatusLabel:
                self._setStatusLabelText(
                    self._segmentEditorStatusLabel,
                    "Fiducial registration applied to the tumor segmentation.",
                )
            self._traceStep123State("onRegisterTumorButton", force=True)
            self._updateButtonStates()

    def onHardenTumorTransformButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to harden tumor transform."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            self._logStepTrace(
                "Step2",
                f"Harden registration requested for tumor='{self._nodeDisplayName(self._parameterNode.tumorSegmentation)}'.",
            )
            self.logic.hardenTumorTransform(self._parameterNode.tumorSegmentation)
            self._logStepTrace("Step2", "Tumor transform hardened.")
            if self._segmentEditorStatusLabel:
                self._setStatusLabelText(
                    self._segmentEditorStatusLabel,
                    "Tumor registration hardened into the segmentation geometry.",
                )
            self._traceStep123State("onHardenTumorTransformButton", force=True)
            self._updateButtonStates()

    def onRiskStructuresSegmentationChanged(self, node=None) -> None:
        if not self.logic or not self._parameterNode:
            return
        selector = getattr(self.ui, "riskStructuresSegmentationSelector", None) if hasattr(self, "ui") else None
        sanitizedNode = self._sanitizeSelectorNodeClass(
            selector,
            node,
            "vtkMRMLSegmentationNode",
            "riskStructuresSegmentationSelector",
        )
        if self._sceneImportInProgress or self._sceneIsBusy():
            return
        if self._parameterNode.riskStructuresSegmentation is not sanitizedNode:
            self._parameterNode.riskStructuresSegmentation = sanitizedNode
        self._parameterNode.criticalStructuresSegmentation = sanitizedNode
        if self.logic.removeNodeIfOwned(
            self._parameterNode.derivedTrajectorySummaryTable,
            GENERATED_DERIVED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
        ):
            self._parameterNode.derivedTrajectorySummaryTable = None
        self._parameterNode.derivedTrajectoryBundleSummaryJson = "{}"
        self._clearOwnedSafetyOutputs(clearReferences=True)
        self._clearOwnedBeginnerOutputs(clearReferences=True)
        self._updateButtonStates()

    def _buildProbeCoordinationConstraintSettings(self) -> ProbeCoordinationConstraintSettings:
        if not self._parameterNode:
            return ProbeCoordinationConstraintSettings()
        return ProbeCoordinationConstraintSettings(
            minInterProbeDistanceMm=float(self._parameterNode.minInterProbeDistanceMm),
            maxInterProbeDistanceMm=float(self._parameterNode.maxInterProbeDistanceMm),
            minEntryPointSpacingMm=float(self._parameterNode.minEntryPointSpacingMm),
            minTargetPointSpacingMm=float(self._parameterNode.minTargetPointSpacingMm),
            maxParallelAngleDeg=float(self._parameterNode.maxParallelAngleDeg),
            maxAllowedOverlapPercentBetweenPerProbeVolumes=float(
                self._parameterNode.maxAllowedOverlapPercentBetweenPerProbeVolumes
            ),
            enableNoTouchCheck=bool(self._parameterNode.enableNoTouchCheck),
            requireAllProbePairsFeasible=bool(self._parameterNode.requireAllProbePairsFeasible),
            enableInterProbeDistanceRule=bool(self._parameterNode.enableInterProbeDistanceRule),
            enableEntrySpacingRule=bool(self._parameterNode.enableEntrySpacingRule),
            enableTargetSpacingRule=bool(self._parameterNode.enableTargetSpacingRule),
            enableAngleRule=bool(self._parameterNode.enableAngleRule),
            enableOverlapRule=bool(self._parameterNode.enableOverlapRule),
        )

    def _evaluateAndPublishProbeCoordination(
        self,
        trajectories: Sequence[ProbeTrajectory],
        updateStatusLabel: bool = True,
    ) -> dict[str, float | int | bool | str]:
        if not self.logic or not self._parameterNode:
            raise RuntimeError("Module logic is not initialized.")

        settings = self._buildProbeCoordinationConstraintSettings()
        pairRows, planSummary, noTouchSummary = self.logic.evaluatePlanProbeCoordination(
            trajectories,
            settings,
            self._parameterNode.tumorSegmentation,
        )

        settingsTable = self.logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            PROBE_COORDINATION_SETTINGS_TABLE_NODE_NAME,
            GENERATED_PROBE_COORDINATION_SETTINGS_TABLE_ATTRIBUTE,
            self._parameterNode.probeCoordinationConstraintSettingsTable,
        )
        pairSummaryTable = self.logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            PROBE_PAIR_COORDINATION_TABLE_NODE_NAME,
            GENERATED_PROBE_PAIR_COORDINATION_TABLE_ATTRIBUTE,
            self._parameterNode.probePairCoordinationSummaryTable,
        )
        planSummaryTable = self.logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            PROBE_COORDINATION_SUMMARY_TABLE_NODE_NAME,
            GENERATED_PROBE_COORDINATION_SUMMARY_TABLE_ATTRIBUTE,
            self._parameterNode.probeCoordinationSummaryTable,
        )
        noTouchSummaryTable = self.logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            NO_TOUCH_SUMMARY_TABLE_NODE_NAME,
            GENERATED_NO_TOUCH_SUMMARY_TABLE_ATTRIBUTE,
            self._parameterNode.noTouchSummaryTable,
        )
        self.logic.populateProbeCoordinationConstraintSettingsTable(settingsTable, settings)
        self.logic.populateProbePairCoordinationSummaryTable(pairSummaryTable, pairRows)
        self.logic.populateProbeCoordinationSummaryTable(planSummaryTable, planSummary)
        self.logic.populateNoTouchSummaryTable(noTouchSummaryTable, noTouchSummary)
        self._parameterNode.probeCoordinationConstraintSettingsTable = settingsTable
        self._parameterNode.probePairCoordinationSummaryTable = pairSummaryTable
        self._parameterNode.probeCoordinationSummaryTable = planSummaryTable
        self._parameterNode.noTouchSummaryTable = noTouchSummaryTable

        if updateStatusLabel and hasattr(self.ui, "probeCoordinationStatusLabel"):
            pairStatus = (
                f"{int(planSummary.get('FeasiblePairCount', 0))}/{int(planSummary.get('PairCount', 0))} pairs feasible"
            )
            noTouchStatus = "No-touch not checked"
            if bool(noTouchSummary.get("NoTouchChecked", False)):
                noTouchStatus = "No-touch pass" if bool(noTouchSummary.get("NoTouchPass", False)) else "No-touch fail"
            self.ui.probeCoordinationStatusLabel.text = f"{pairStatus}; {noTouchStatus}"

        return planSummary

    def onEvaluateProbeCoordinationButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to evaluate probe coordination."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            if not self._parameterNode.endpointsMarkups:
                raise ValueError("Select an endpoint markups node before evaluating probe coordination.")

            trajectories = self._plannedTrajectoryBundle(requireValidatedMasterForArray=True)
            if len(trajectories) == 0:
                raise ValueError("No valid trajectories are available for probe coordination evaluation.")
            self._evaluateAndPublishProbeCoordination(trajectories, updateStatusLabel=True)
            self._updateButtonStates()

    def onEvaluateMarginsButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to evaluate margins."), waitCursor=True):
            stepStartTime = time.perf_counter()
            self._logStepTrace("Step5", "Evaluate MAM requested.")
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            if not self._parameterNode.tumorSegmentation:
                raise ValueError("Select a tumor segmentation before evaluating margins.")

            probeSegmentation = self._parameterNode.combinedProbeSegmentation
            if not probeSegmentation:
                generatedProbeNodeIDs = self.logic.resolveExistingNodeIDs(
                    self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs)
                )
                if len(generatedProbeNodeIDs) == 0:
                    raise ValueError("No probe segmentation is available for margin evaluation. Place and merge probes first.")
                probeSegmentation = self.logic.mergeProbeInstances(generatedProbeNodeIDs, self._parameterNode.combinedProbeSegmentation)
                self._parameterNode.combinedProbeSegmentation = probeSegmentation
            if ENABLE_MAM_DEBUG_LOGGING:
                generatedProbeCount = len(
                    self.logic.resolveExistingNodeIDs(
                        self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs)
                    )
                )
                logging.info(
                    "MAM debug: onEvaluateMarginsButton inputs | tumor='%s' | probe='%s' | generatedProbeCount=%d | mamMm=%.2f",
                    self._nodeDisplayName(self._parameterNode.tumorSegmentation),
                    self._nodeDisplayName(probeSegmentation),
                    generatedProbeCount,
                    float(self._parameterNode.mamMm),
                )

            signedDistanceStartTime = time.perf_counter()
            outputMarginModel, resultTable, summary = self.logic.evaluateMargins(
                self._parameterNode.tumorSegmentation,
                probeSegmentation,
                self._parameterNode.outputMarginModel,
                self._parameterNode.resultTable,
            )
            self._logStepTrace(
                "Step5",
                f"Signed-margin computation finished in {time.perf_counter() - signedDistanceStartTime:.2f}s.",
            )
            self._parameterNode.outputMarginModel = outputMarginModel
            self._parameterNode.resultTable = resultTable
            logging.info("Margin summary: %s", summary)

            mamColoringStartTime = time.perf_counter()
            signedMarginValues = self.logic.getSignedMarginValuesArray(outputMarginModel)
            mamAssessmentSummary = self.logic.applyBeginnerMamColoring(outputMarginModel, float(self._parameterNode.mamMm))
            self._logStepTrace(
                "Step5",
                f"MAM coloring + summary finished in {time.perf_counter() - mamColoringStartTime:.2f}s.",
            )
            self._parameterNode.mamAssessmentSummaryJson = json.dumps(mamAssessmentSummary, sort_keys=True)
            if self._marginAssessmentStatusLabel:
                self._setStatusLabelText(
                    self._marginAssessmentStatusLabel,
                    str(mamAssessmentSummary.get("StatusText", "MAM assessment completed.")),
                )
            if bool(mamAssessmentSummary.get("MamPass", False)):
                slicer.util.infoDisplay(str(mamAssessmentSummary.get("StatusText", "MAM satisfied.")), windowTitle="MAM Validation")
            trajectoryCount = len(
                self.logic.resolveExistingNodeIDs(
                    self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs)
                )
            )
            if trajectoryCount <= 0:
                trajectoryCount = self.logic.tableNodeRowCount(self._parameterNode.trajectorySummaryTable)

            tumorSegmentID, tumorSegmentName = self.logic.getPreferredSegmentInfo(
                self._parameterNode.tumorSegmentation,
                DEFAULT_TUMOR_SEGMENT_NAMES,
                "plan summary",
            )
            planSummary = self.logic.computeSignedMarginSummary(
                signedMarginValues,
                trajectoryCount=trajectoryCount,
                tumorSegmentID=tumorSegmentID,
                tumorSegmentName=tumorSegmentName,
            )
            thresholdSummary = self.logic.computeMarginThresholdSummary(signedMarginValues)

            planSummaryTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                PLAN_SUMMARY_TABLE_NODE_NAME,
                GENERATED_PLAN_SUMMARY_TABLE_ATTRIBUTE,
                self._parameterNode.planSummaryTable,
            )
            marginThresholdSummaryTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                MARGIN_THRESHOLD_SUMMARY_TABLE_NODE_NAME,
                GENERATED_MARGIN_THRESHOLD_TABLE_ATTRIBUTE,
                self._parameterNode.marginThresholdSummaryTable,
            )
            self.logic.populatePlanSummaryTable(planSummaryTable, planSummary)
            self.logic.populateMarginThresholdSummaryTable(marginThresholdSummaryTable, thresholdSummary)
            self._parameterNode.planSummaryTable = planSummaryTable
            self._parameterNode.marginThresholdSummaryTable = marginThresholdSummaryTable

            if BEGINNER_WORKFLOW_MODE:
                self._clearOwnedSafetyOutputs(clearReferences=True)
                self._clearOwnedCoordinationOutputs(clearReferences=True)
                self._logStepTrace(
                    "Step5",
                    "Beginner workflow: MAM outputs completed; skipped structure-safety and coordination evaluations.",
                )
                self._logStepTrace("Step5", f"Evaluate MAM total time: {time.perf_counter() - stepStartTime:.2f}s.")
                self._updateButtonStates()
                return

            shouldEvaluateStructureSafety = bool(
                (not BEGINNER_WORKFLOW_MODE)
                and self._parameterNode.criticalStructuresSegmentation
            )
            if shouldEvaluateStructureSafety:
                safetyStartTime = time.perf_counter()
                try:
                    structureSafetySummaryRows, structureSafetyThresholdRows = self.logic.evaluateStructureSafety(
                        self._parameterNode.criticalStructuresSegmentation,
                        probeSegmentation,
                    )
                except Exception:
                    logging.exception("Structure-safety evaluation failed; continuing with MAM outputs only.")
                    structureSafetySummaryRows, structureSafetyThresholdRows = [], []
                self._logStepTrace(
                    "Step5",
                    (
                        f"Structure-safety evaluation finished in {time.perf_counter() - safetyStartTime:.2f}s "
                        f"for {len(structureSafetySummaryRows)} segments."
                    ),
                )
                if len(structureSafetySummaryRows) > 0:
                    structureSafetySummaryTable = self.logic.createOrReuseOwnedOutputNode(
                        "vtkMRMLTableNode",
                        STRUCTURE_SAFETY_SUMMARY_TABLE_NODE_NAME,
                        GENERATED_STRUCTURE_SAFETY_SUMMARY_TABLE_ATTRIBUTE,
                        self._parameterNode.structureSafetySummaryTable,
                    )
                    structureSafetyThresholdSummaryTable = self.logic.createOrReuseOwnedOutputNode(
                        "vtkMRMLTableNode",
                        STRUCTURE_SAFETY_THRESHOLD_SUMMARY_TABLE_NODE_NAME,
                        GENERATED_STRUCTURE_SAFETY_THRESHOLD_TABLE_ATTRIBUTE,
                        self._parameterNode.structureSafetyThresholdSummaryTable,
                    )
                    self.logic.populateStructureSafetySummaryTable(structureSafetySummaryTable, structureSafetySummaryRows)
                    self.logic.populateStructureSafetyThresholdSummaryTable(
                        structureSafetyThresholdSummaryTable,
                        structureSafetyThresholdRows,
                    )
                    self._parameterNode.structureSafetySummaryTable = structureSafetySummaryTable
                    self._parameterNode.structureSafetyThresholdSummaryTable = structureSafetyThresholdSummaryTable
                else:
                    self._clearOwnedSafetyOutputs(clearReferences=True)
            else:
                self._clearOwnedSafetyOutputs(clearReferences=True)
                self._logStepTrace(
                    "Step5",
                    "Structure-safety evaluation skipped in beginner workflow mode.",
                )

            if self._parameterNode.endpointsMarkups and not BEGINNER_WORKFLOW_MODE:
                controlPointCount = int(self._parameterNode.endpointsMarkups.GetNumberOfControlPoints())
                if controlPointCount >= 2 and controlPointCount % 2 == 0:
                    trajectories = self.logic.extractTrajectoriesFromMarkups(
                        self._parameterNode.endpointsMarkups,
                        strictEven=True,
                    )
                    self._evaluateAndPublishProbeCoordination(trajectories, updateStatusLabel=True)
                else:
                    self._clearOwnedCoordinationOutputs(clearReferences=True)
            elif BEGINNER_WORKFLOW_MODE:
                self._clearOwnedCoordinationOutputs(clearReferences=True)
            self._logStepTrace("Step5", f"Evaluate MAM total time: {time.perf_counter() - stepStartTime:.2f}s.")
            self._updateButtonStates()

    def onRecolorMarginsButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to recolor margins."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            if not self._parameterNode.outputMarginModel:
                raise ValueError("No margin model is available. Evaluate margins first.")
            thresholds = (
                self._parameterNode.recolorThresholdLow,
                self._parameterNode.recolorThresholdMid,
                self._parameterNode.recolorThresholdHigh,
            )
            self.logic.recolorMarginModel(self._parameterNode.outputMarginModel, thresholds)
            self._updateButtonStates()

    def onResetMarginColorsButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to reset margin colors."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            if not self._parameterNode.outputMarginModel:
                raise ValueError("No margin model is available. Evaluate margins first.")
            self.logic.resetMarginModelColors(self._parameterNode.outputMarginModel)
            self._updateButtonStates()

    def _buildCohortExecutionConfig(self) -> CohortExecutionConfig:
        if not self._parameterNode:
            return CohortExecutionConfig()

        cohortStudyDefinitionPath = (
            str(self.ui.cohortStudyDefinitionPathLineEdit.text)
            if hasattr(self.ui, "cohortStudyDefinitionPathLineEdit")
            else str(self._parameterNode.cohortStudyDefinitionPath)
        )
        cohortExecutionMode = (
            str(self.ui.cohortExecutionModeComboBox.currentText)
            if hasattr(self.ui, "cohortExecutionModeComboBox")
            else str(self._parameterNode.cohortExecutionMode)
        )
        cohortMaxCases = (
            int(self.ui.cohortMaxCasesSpinBox.value)
            if hasattr(self.ui, "cohortMaxCasesSpinBox")
            else int(self._parameterNode.cohortMaxCases)
        )

        self._parameterNode.cohortStudyDefinitionPath = cohortStudyDefinitionPath
        self._parameterNode.cohortExecutionMode = cohortExecutionMode
        self._parameterNode.cohortMaxCases = int(cohortMaxCases)

        return CohortExecutionConfig(
            studyDefinitionPath=cohortStudyDefinitionPath,
            executionMode=cohortExecutionMode,
            includeMarginMetrics=bool(self._parameterNode.cohortIncludeMarginMetrics),
            includeSafetyMetrics=bool(self._parameterNode.cohortIncludeSafetyMetrics),
            includeCoverageMetrics=bool(self._parameterNode.cohortIncludeCoverageMetrics),
            includeFeasibilityMetrics=bool(self._parameterNode.cohortIncludeFeasibilityMetrics),
            includeCoordinationMetrics=bool(self._parameterNode.cohortIncludeCoordinationMetrics),
            includeVerificationMetrics=bool(self._parameterNode.cohortIncludeVerificationMetrics),
            includeRecommendationMetrics=bool(self._parameterNode.cohortIncludeRecommendationMetrics),
            maxCases=int(cohortMaxCases),
        )

    def onRunCohortEvaluationButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to run cohort evaluation."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")

            executionConfig = self._buildCohortExecutionConfig()
            cohortResult = self.logic.runCohortStudy(self._parameterNode, executionConfig)

            executionSummaryTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                COHORT_EXECUTION_SUMMARY_TABLE_NODE_NAME,
                GENERATED_COHORT_EXECUTION_SUMMARY_TABLE_ATTRIBUTE,
                self._parameterNode.cohortExecutionSummaryTable,
            )
            caseSummaryTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                COHORT_CASE_SUMMARY_TABLE_NODE_NAME,
                GENERATED_COHORT_CASE_SUMMARY_TABLE_ATTRIBUTE,
                self._parameterNode.cohortCaseSummaryTable,
            )
            aggregateMetricsTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                COHORT_AGGREGATE_METRICS_TABLE_NODE_NAME,
                GENERATED_COHORT_AGGREGATE_METRICS_TABLE_ATTRIBUTE,
                self._parameterNode.cohortAggregateMetricsTable,
            )
            comparisonSummaryTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                COHORT_COMPARISON_SUMMARY_TABLE_NODE_NAME,
                GENERATED_COHORT_COMPARISON_SUMMARY_TABLE_ATTRIBUTE,
                self._parameterNode.cohortComparisonSummaryTable,
            )

            self.logic.populateCohortExecutionSummaryTable(
                executionSummaryTable,
                cohortResult["executionSummary"],
            )
            self.logic.populateCohortCaseSummaryTable(
                caseSummaryTable,
                cohortResult["caseResults"],
            )
            self.logic.populateCohortAggregateMetricsTable(
                aggregateMetricsTable,
                cohortResult["aggregateMetrics"],
            )
            self.logic.populateCohortComparisonSummaryTable(
                comparisonSummaryTable,
                cohortResult["comparisonRows"],
            )
            self._parameterNode.cohortExecutionSummaryTable = executionSummaryTable
            self._parameterNode.cohortCaseSummaryTable = caseSummaryTable
            self._parameterNode.cohortAggregateMetricsTable = aggregateMetricsTable
            self._parameterNode.cohortComparisonSummaryTable = comparisonSummaryTable

            if hasattr(self.ui, "cohortStatusLabel"):
                executionSummary = cohortResult["executionSummary"]
                self.ui.cohortStatusLabel.text = (
                    f"Study {str(executionSummary.get('StudyID', ''))}: "
                    f"{int(executionSummary.get('SuccessCount', 0))}/"
                    f"{int(executionSummary.get('CaseCount', 0))} cases succeeded"
                )
            self._updateButtonStates()

    def _buildReproducibilityPackageConfig(self) -> ReproducibilityPackageConfig:
        if not self._parameterNode:
            return ReproducibilityPackageConfig()

        packageMode = (
            str(self.ui.packageModeComboBox.currentText)
            if hasattr(self.ui, "packageModeComboBox")
            else str(self._parameterNode.packageMode)
        )
        packageBaseName = (
            str(self.ui.packageBaseNameLineEdit.text)
            if hasattr(self.ui, "packageBaseNameLineEdit")
            else str(self._parameterNode.packageBaseName)
        )
        packageOutputDirectory = (
            str(self.ui.packageOutputDirectoryLineEdit.text)
            if hasattr(self.ui, "packageOutputDirectoryLineEdit")
            else str(self._parameterNode.packageOutputDirectory)
        )

        self._parameterNode.packageMode = packageMode
        self._parameterNode.packageBaseName = packageBaseName
        self._parameterNode.packageOutputDirectory = packageOutputDirectory

        return ReproducibilityPackageConfig(
            packageMode=packageMode,
            includeBenchmarkArtifacts=bool(self._parameterNode.includeBenchmarkArtifacts),
            includeScenarioRegistry=bool(self._parameterNode.includeScenarioRegistry),
            includeCohortStudyArtifacts=bool(self._parameterNode.includeCohortStudyArtifacts),
            includeStudyAnalytics=bool(self._parameterNode.includeStudyAnalytics),
            includeReports=bool(self._parameterNode.includeReports),
            includeCanonicalJson=bool(self._parameterNode.includeCanonicalJson),
            includeValidationResults=bool(self._parameterNode.includeValidationResults),
            packageBaseName=packageBaseName,
            outputDirectory=packageOutputDirectory,
            lastPackageSequence=int(self._parameterNode.lastReproducibilityPackageSequence),
        )

    def onGenerateReproducibilityPackageButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to generate reproducibility package."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")

            packageConfig = self._buildReproducibilityPackageConfig()
            packageResult = self.logic.assembleReproducibilityPackage(self._parameterNode, packageConfig)

            self._parameterNode.lastReproducibilityPackageSequence = int(packageResult["packageSequence"])
            self._parameterNode.packageOutputDirectory = str(packageResult["packageDirectory"])
            if hasattr(self.ui, "packageOutputDirectoryLineEdit"):
                self.ui.packageOutputDirectoryLineEdit.text = str(self._parameterNode.packageOutputDirectory)

            packageSummaryTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                REPRODUCIBILITY_PACKAGE_SUMMARY_TABLE_NODE_NAME,
                GENERATED_REPRODUCIBILITY_PACKAGE_SUMMARY_TABLE_ATTRIBUTE,
                self._parameterNode.reproducibilityPackageSummaryTable,
            )
            manifestPreviewTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                REPRODUCIBILITY_MANIFEST_PREVIEW_TABLE_NODE_NAME,
                GENERATED_REPRODUCIBILITY_MANIFEST_PREVIEW_TABLE_ATTRIBUTE,
                self._parameterNode.reproducibilityManifestPreviewTable,
            )
            artifactIndexTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                REPRODUCIBILITY_ARTIFACT_INDEX_TABLE_NODE_NAME,
                GENERATED_REPRODUCIBILITY_ARTIFACT_INDEX_TABLE_ATTRIBUTE,
                self._parameterNode.reproducibilityArtifactIndexTable,
            )

            self.logic.populateReproducibilityPackageSummaryTable(
                packageSummaryTable,
                {
                    "PackageMode": packageConfig.packageMode,
                    "PackageBaseName": packageConfig.packageBaseName,
                    "PackagePath": str(packageResult["packagePath"]),
                    "PackageDirectory": str(packageResult["packageDirectory"]),
                    "ArtifactCount": int(packageResult["artifactCount"]),
                    "WarningCount": int(packageResult["warningCount"]),
                    "LastPackageStatus": str(packageResult["status"]),
                    "LastPackageSequence": int(packageResult["packageSequence"]),
                },
            )
            manifestDict = asdict(packageResult["manifest"])
            self.logic.populateReproducibilityManifestPreviewTable(manifestPreviewTable, manifestDict)
            self.logic.populateReproducibilityArtifactIndexTable(
                artifactIndexTable,
                packageResult["artifactEntries"],
            )

            self._parameterNode.reproducibilityPackageSummaryTable = packageSummaryTable
            self._parameterNode.reproducibilityManifestPreviewTable = manifestPreviewTable
            self._parameterNode.reproducibilityArtifactIndexTable = artifactIndexTable

            if hasattr(self.ui, "reproducibilityStatusLabel"):
                self.ui.reproducibilityStatusLabel.text = (
                    f"Package {int(packageResult['packageSequence']):04d}: "
                    f"{int(packageResult['artifactCount'])} artifacts, "
                    f"{int(packageResult['warningCount'])} warning(s)"
                )
            self._updateButtonStates()

    def _buildPlanExportConfig(self) -> PlanExportConfig:
        if not self._parameterNode:
            return PlanExportConfig()

        if BEGINNER_WORKFLOW_MODE:
            exportMode = "CurrentWorkingPlan"
            selectedExportScenarioID = ""
        else:
            exportMode = (
                str(self.ui.exportModeComboBox.currentText)
                if hasattr(self.ui, "exportModeComboBox")
                else str(self._parameterNode.exportMode)
            )
            selectedExportScenarioID = (
                str(self.ui.selectedExportScenarioIDLineEdit.text)
                if hasattr(self.ui, "selectedExportScenarioIDLineEdit")
                else str(self._parameterNode.selectedExportScenarioID)
            )
        exportBaseName = str(self.ui.exportBaseNameLineEdit.text) if hasattr(self.ui, "exportBaseNameLineEdit") else str(self._parameterNode.exportBaseName)
        exportDirectory = (
            str(self.ui.exportDirectoryLineEdit.text)
            if hasattr(self.ui, "exportDirectoryLineEdit")
            else str(self._parameterNode.lastExportDirectory)
        )

        self._parameterNode.exportMode = exportMode
        self._parameterNode.selectedExportScenarioID = selectedExportScenarioID
        self._parameterNode.exportBaseName = exportBaseName
        self._parameterNode.lastExportDirectory = exportDirectory

        return PlanExportConfig(
            exportMode=exportMode,
            selectedExportScenarioID=selectedExportScenarioID,
            exportBaseName=exportBaseName,
            exportDirectory=exportDirectory,
            lastExportSequence=int(self._parameterNode.lastExportSequence),
            includeWorkingPlan=bool(self._parameterNode.includeWorkingPlan),
            includeSelectedScenario=bool(self._parameterNode.includeSelectedScenario),
            includeScenarioComparison=bool(self._parameterNode.includeScenarioComparison),
            includeRecommendationOutputs=bool(self._parameterNode.includeRecommendationOutputs),
            includeTrajectoryTables=bool(self._parameterNode.includeTrajectoryTables),
            includeSafetyTables=bool(self._parameterNode.includeSafetyTables),
            includeCoverageTables=bool(self._parameterNode.includeCoverageTables),
            includeFeasibilityTables=bool(self._parameterNode.includeFeasibilityTables),
            includeCoordinationTables=bool(self._parameterNode.includeCoordinationTables),
        )

    def onExportBundleButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to export plan bundle."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")

            exportConfig = self._buildPlanExportConfig()
            exportResult = self.logic.exportPlanBundle(self._parameterNode, exportConfig)

            self._parameterNode.lastExportSequence = int(exportResult["exportSequence"])
            self._parameterNode.lastExportDirectory = str(exportResult["exportDirectory"])
            if hasattr(self.ui, "exportDirectoryLineEdit"):
                self.ui.exportDirectoryLineEdit.text = str(self._parameterNode.lastExportDirectory)

            exportSummaryTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                EXPORT_SUMMARY_TABLE_NODE_NAME,
                GENERATED_EXPORT_SUMMARY_TABLE_ATTRIBUTE,
                self._parameterNode.exportSummaryTable,
            )
            exportManifestPreviewTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                EXPORT_MANIFEST_PREVIEW_TABLE_NODE_NAME,
                GENERATED_EXPORT_MANIFEST_PREVIEW_TABLE_ATTRIBUTE,
                self._parameterNode.exportManifestPreviewTable,
            )

            self.logic.populateExportSummaryTable(
                exportSummaryTable,
                {
                    "ExportMode": exportConfig.exportMode,
                    "ExportBaseName": exportConfig.exportBaseName,
                    "SelectedScenarioID": exportConfig.selectedExportScenarioID,
                    "SelectedScenarioName": str(exportResult.get("selectedScenarioName", "")),
                    "FileCount": int(exportResult["fileCount"]),
                    "WarningCount": int(exportResult.get("warningCount", 0)),
                    "LastExportStatus": str(exportResult["status"]),
                    "LastExportDirectory": str(exportResult["bundlePath"]),
                    "LastExportSequence": int(exportResult["exportSequence"]),
                },
            )
            manifestDict = asdict(exportResult["manifest"])
            self.logic.populateExportManifestPreviewTable(exportManifestPreviewTable, manifestDict)
            self._parameterNode.exportSummaryTable = exportSummaryTable
            self._parameterNode.exportManifestPreviewTable = exportManifestPreviewTable

            if hasattr(self.ui, "exportStatusLabel"):
                warningCount = int(exportResult.get("warningCount", 0))
                warningSuffix = f" with {warningCount} warning(s)" if warningCount > 0 else ""
                self.ui.exportStatusLabel.text = f"Exported {int(exportResult['fileCount'])} files to {str(exportResult['bundlePath'])}{warningSuffix}"
            self._updateButtonStates()


#
# SurgicalVision3D_PlannerLogic
#


class SurgicalVision3D_PlannerLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return SurgicalVision3D_PlannerParameterNode(super().getParameterNode())

    @staticmethod
    def serializeNodeIDs(nodeIDs: Sequence[str]) -> str:
        return json.dumps([nodeID for nodeID in nodeIDs if nodeID])

    @staticmethod
    def deserializeNodeIDs(serializedNodeIDs: str | None) -> list[str]:
        if not serializedNodeIDs:
            return []
        try:
            parsed = json.loads(serializedNodeIDs)
            return [str(nodeID) for nodeID in parsed if nodeID]
        except Exception:
            logging.warning("Failed to parse serialized node IDs: %s", serializedNodeIDs)
            return []

    @staticmethod
    def resolveExistingNodeIDs(nodeIDs: Sequence[str]) -> list[str]:
        existingNodeIDs: list[str] = []
        for nodeID in nodeIDs:
            if slicer.mrmlScene.GetNodeByID(nodeID):
                existingNodeIDs.append(nodeID)
        return existingNodeIDs

    @staticmethod
    def mergeNodeIDLists(*nodeIDLists: Sequence[str]) -> list[str]:
        mergedNodeIDs: list[str] = []
        seenNodeIDs: set[str] = set()
        for nodeIDList in nodeIDLists:
            for nodeID in nodeIDList:
                if not nodeID or nodeID in seenNodeIDs:
                    continue
                seenNodeIDs.add(nodeID)
                mergedNodeIDs.append(nodeID)
        return mergedNodeIDs

    @staticmethod
    def _canonicalNodeName(nodeName: str) -> str:
        return "".join(character for character in str(nodeName).lower() if character.isalnum())

    @staticmethod
    def isGenericDefaultFiducialsNodeName(nodeName: str) -> bool:
        loweredName = str(nodeName or "").strip().lower()
        if loweredName == "f":
            return True
        return loweredName.startswith("f_") and loweredName[2:].isdigit()

    def findFirstNodeByClassAndPreferredNames(self, nodeClassName: str, preferredNames: Sequence[str]):
        normalizedPreferredNames = [
            self._canonicalNodeName(name)
            for name in preferredNames
            if str(name or "").strip()
        ]
        if len(normalizedPreferredNames) == 0:
            return None

        candidateNodes = slicer.util.getNodesByClass(nodeClassName)
        for preferredName in normalizedPreferredNames:
            for candidateNode in candidateNodes:
                if self._canonicalNodeName(candidateNode.GetName()) == preferredName:
                    return candidateNode
        return None

    def loadGeometryCatalog(self) -> list[GeometryCatalogEntry]:
        catalogPath = self._resolveResourcePath(BEGINNER_GEOMETRY_CATALOG_RELATIVE_PATH)
        if not catalogPath.exists():
            logging.warning("Beginner geometry catalog was not found: %s", catalogPath)
            return []

        try:
            catalogValues = json.loads(catalogPath.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Failed to parse beginner geometry catalog: %s", catalogPath)
            return []

        geometryEntries: list[GeometryCatalogEntry] = []
        for entry in catalogValues if isinstance(catalogValues, list) else []:
            if not isinstance(entry, dict):
                continue
            geometryId = str(entry.get("id", "")).strip()
            displayName = str(entry.get("label", "")).strip()
            templateRelativePath = str(entry.get("templateFile", "")).strip()
            if not geometryId or not displayName or not templateRelativePath:
                continue
            geometryEntries.append(
                GeometryCatalogEntry(
                    geometryId=geometryId,
                    displayName=displayName,
                    templateRelativePath=templateRelativePath,
                    activeElementLengthMm=float(entry.get("activeElementLengthMm", 30.0)),
                    axialPlacementOffsetMm=float(entry.get("axialPlacementOffsetMm", 0.0)),
                )
            )
        return geometryEntries

    def geometryCatalogEntryById(self, geometryId: str) -> GeometryCatalogEntry | None:
        normalizedGeometryId = str(geometryId or "").strip().lower()
        if not normalizedGeometryId:
            return None
        for geometryEntry in self.loadGeometryCatalog():
            if str(geometryEntry.geometryId).strip().lower() == normalizedGeometryId:
                return geometryEntry
        return None

    def referenceProbeTemplateNodeForCatalogEntry(self, geometryEntry: GeometryCatalogEntry | None) -> vtkMRMLSegmentationNode | None:
        if geometryEntry is None:
            return None
        templatePath = self._resolveResourcePath(geometryEntry.templateRelativePath)
        normalizedTemplatePath = self._normalizeFilesystemPath(templatePath)
        for templateNode in self.ensureReferenceProbeTemplatesLoaded():
            sourcePath = templateNode.GetAttribute(REFERENCE_PROBE_TEMPLATE_SOURCE_PATH_ATTRIBUTE) or ""
            if sourcePath and self._normalizeFilesystemPath(sourcePath) == normalizedTemplatePath:
                templateNode.SetAttribute(
                    REFERENCE_PROBE_TEMPLATE_AXIAL_PLACEMENT_OFFSET_MM_ATTRIBUTE,
                    f"{float(geometryEntry.axialPlacementOffsetMm):.6f}",
                )
                return templateNode
        return None

    def loadCaseFolder(self, caseFolderPath: str | Path) -> None:
        resolvedCaseFolderPath = Path(caseFolderPath).resolve()
        if not resolvedCaseFolderPath.exists() or not resolvedCaseFolderPath.is_dir():
            raise ValueError(f"Case folder was not found: {resolvedCaseFolderPath}")
        if not self._loadBundledCaseAssetsFromDirectory(resolvedCaseFolderPath):
            raise RuntimeError(f"No loadable .nrrd or .seg.nrrd assets were found in: {resolvedCaseFolderPath}")

    def findPreferredSegmentID(
        self,
        segmentationNode: vtkMRMLSegmentationNode | None,
        preferredNames: Sequence[str],
        operationName: str,
        fallbackToFirst: bool = True,
    ) -> str:
        if not segmentationNode:
            raise ValueError(f"{operationName}: segmentation node is required.")
        segmentation = segmentationNode.GetSegmentation()
        if not segmentation or segmentation.GetNumberOfSegments() <= 0:
            raise RuntimeError(f"{operationName}: segmentation '{segmentationNode.GetName()}' has no segments.")

        normalizedPreferredNames = {self._canonicalNodeName(name) for name in preferredNames if str(name or "").strip()}
        if normalizedPreferredNames:
            for segmentIndex in range(segmentation.GetNumberOfSegments()):
                segmentID = segmentation.GetNthSegmentID(segmentIndex)
                segment = segmentation.GetSegment(segmentID) if segmentID else None
                if segment and self._canonicalNodeName(segment.GetName()) in normalizedPreferredNames:
                    return str(segmentID)

        if not fallbackToFirst:
            raise RuntimeError(f"{operationName}: preferred segment was not found in '{segmentationNode.GetName()}'.")
        return self.getWorkingSegmentID(segmentationNode, operationName)

    def getPreferredSegmentInfo(
        self,
        segmentationNode: vtkMRMLSegmentationNode | None,
        preferredNames: Sequence[str],
        operationName: str,
        fallbackToFirst: bool = True,
    ) -> tuple[str, str]:
        segmentID = self.findPreferredSegmentID(segmentationNode, preferredNames, operationName, fallbackToFirst=fallbackToFirst)
        segment = segmentationNode.GetSegmentation().GetSegment(segmentID) if segmentationNode else None
        segmentName = segment.GetName() if segment and segment.GetName() else segmentID
        return segmentID, segmentName

    def extractSingleTrajectoryFromMarkups(self, endpointsMarkups: vtkMRMLMarkupsFiducialNode | None) -> ProbeTrajectory:
        trajectories = self.extractTrajectoriesFromMarkups(endpointsMarkups, strictEven=True)
        if len(trajectories) != 1:
            raise ValueError(
                f"Beginner workflow expects exactly one entry/endpoint pair, but found {len(trajectories)} trajectories."
            )
        return trajectories[0]

    @staticmethod
    def createOrthogonalArrayBasis(directionVectorRAS: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        direction = _normalize_vector(directionVectorRAS)
        seedAxes = (
            np.array([0.0, 0.0, 1.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        )
        for seedAxis in seedAxes:
            projected = seedAxis - (np.dot(seedAxis, direction) * direction)
            projectedNorm = float(np.linalg.norm(projected))
            if projectedNorm <= 1e-6:
                continue
            basisU = projected / projectedNorm
            basisV = np.cross(direction, basisU)
            basisVNorm = float(np.linalg.norm(basisV))
            if basisVNorm <= 1e-6:
                continue
            return basisU, (basisV / basisVNorm)
        raise RuntimeError("Failed to construct a deterministic orthogonal basis for trajectory-array generation.")

    @staticmethod
    def _derivedTrajectoryRoleName(derivedCount: int, derivedIndex: int) -> str:
        if int(derivedCount) == 4:
            cardinalRoles = ("North", "East", "South", "West")
            return cardinalRoles[int(derivedIndex) % 4]
        return f"Derived {int(derivedIndex) + 1:02d}"

    def generateDerivedParallelTrajectories(
        self,
        masterTrajectory: ProbeTrajectory,
        derivedCount: int,
        radiusMm: float,
        angleOffsetDeg: float = 0.0,
        includeMaster: bool = True,
    ) -> list[ProbeTrajectory]:
        if int(derivedCount) < 1:
            raise ValueError("Derived trajectory count must be at least 1.")
        radiusMm = float(max(0.0, radiusMm))

        masterEntry = np.asarray(masterTrajectory.entryPointRAS, dtype=float)
        masterTarget = np.asarray(masterTrajectory.targetPointRAS, dtype=float)
        masterDirection = _normalize_vector(masterTrajectory.directionVector)
        basisU, basisV = self.createOrthogonalArrayBasis(masterDirection)
        angularStepDeg = 360.0 / float(derivedCount)

        bundle: list[ProbeTrajectory] = []
        if includeMaster:
            bundle.append(
                ProbeTrajectory(
                    entryPointRAS=tuple(float(value) for value in masterEntry.tolist()),
                    targetPointRAS=tuple(float(value) for value in masterTarget.tolist()),
                    directionVector=tuple(float(value) for value in masterDirection.tolist()),
                    lengthMm=float(masterTrajectory.lengthMm),
                    trajectoryIndex=len(bundle),
                    label="Master",
                    sourceControlPointIndices=masterTrajectory.sourceControlPointIndices,
                    role="Master",
                    angleDeg=None,
                    radialOffsetMm=0.0,
                    derivedFromMaster=False,
                )
            )

        for derivedIndex in range(int(derivedCount)):
            angleDeg = float(angleOffsetDeg + (angularStepDeg * float(derivedIndex)))
            angleRad = math.radians(angleDeg)
            offset = radiusMm * ((math.cos(angleRad) * basisU) + (math.sin(angleRad) * basisV))
            childEntry = masterEntry + offset
            childTarget = masterTarget + offset
            role = self._derivedTrajectoryRoleName(int(derivedCount), int(derivedIndex))
            bundle.append(
                ProbeTrajectory(
                    entryPointRAS=tuple(float(value) for value in childEntry.tolist()),
                    targetPointRAS=tuple(float(value) for value in childTarget.tolist()),
                    directionVector=tuple(float(value) for value in masterDirection.tolist()),
                    lengthMm=float(masterTrajectory.lengthMm),
                    trajectoryIndex=len(bundle),
                    label=role,
                    sourceControlPointIndices=masterTrajectory.sourceControlPointIndices,
                    role=role,
                    angleDeg=float(((angleDeg % 360.0) + 360.0) % 360.0),
                    radialOffsetMm=radiusMm,
                    derivedFromMaster=True,
                )
            )

        for trajectoryIndex, trajectory in enumerate(bundle):
            trajectory.trajectoryIndex = int(trajectoryIndex)
        return bundle

    @staticmethod
    def tableNodeRowCount(tableNode: vtkMRMLTableNode | None) -> int:
        if not tableNode or not slicer.mrmlScene.IsNodePresent(tableNode):
            return 0
        table = tableNode.GetTable()
        if table is None:
            return 0
        return int(table.GetNumberOfRows())

    @staticmethod
    def segmentationSegmentCount(segmentationNode: vtkMRMLSegmentationNode | None) -> int:
        if not segmentationNode or not slicer.mrmlScene.IsNodePresent(segmentationNode):
            return 0
        segmentation = segmentationNode.GetSegmentation()
        if not segmentation:
            return 0
        return int(segmentation.GetNumberOfSegments())

    @staticmethod
    def sanitizeExportBaseName(exportBaseName: str) -> str:
        sanitized = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in exportBaseName)
        sanitized = sanitized.strip("_")
        return sanitized or "SV3D_Export"

    @staticmethod
    def buildDeterministicBundlePath(exportDirectory: str, exportBaseName: str, exportSequence: int) -> Path:
        bundleBaseName = SurgicalVision3D_PlannerLogic.sanitizeExportBaseName(exportBaseName)
        rootDirectory = Path(exportDirectory)
        return rootDirectory / f"{bundleBaseName}_{int(exportSequence):04d}"

    @staticmethod
    def _findFirstTableNodeByName(nodeName: str) -> vtkMRMLTableNode | None:
        tableNodes = slicer.util.getNodesByClass("vtkMRMLTableNode")
        for tableNode in tableNodes:
            if tableNode.GetName() == nodeName:
                return tableNode
        return None

    @staticmethod
    def _tableNodeToDictionaries(tableNode: vtkMRMLTableNode | None) -> list[dict[str, str]]:
        if not tableNode or not slicer.mrmlScene.IsNodePresent(tableNode):
            return []

        table = tableNode.GetTable()
        if table is None:
            return []

        columnCount = int(table.GetNumberOfColumns())
        rowCount = int(table.GetNumberOfRows())
        columnNames = [str(table.GetColumnName(columnIndex) or f"Column{columnIndex}") for columnIndex in range(columnCount)]
        rows: list[dict[str, str]] = []
        for rowIndex in range(rowCount):
            rowValues: dict[str, str] = {}
            for columnIndex, columnName in enumerate(columnNames):
                rowValues[columnName] = table.GetValue(rowIndex, columnIndex).ToString()
            rows.append(rowValues)
        return rows

    @staticmethod
    def exportTableNodeToCsv(tableNode: vtkMRMLTableNode, outputCsvPath: Path) -> None:
        if not tableNode:
            raise ValueError("Table node is required for CSV export.")

        table = tableNode.GetTable()
        if table is None:
            raise RuntimeError(f"Cannot export table '{tableNode.GetName()}': table data is unavailable.")

        outputCsvPath.parent.mkdir(parents=True, exist_ok=True)
        columnCount = int(table.GetNumberOfColumns())
        rowCount = int(table.GetNumberOfRows())
        columnNames = [str(table.GetColumnName(columnIndex) or f"Column{columnIndex}") for columnIndex in range(columnCount)]

        with outputCsvPath.open("w", encoding="utf-8", newline="") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(columnNames)
            for rowIndex in range(rowCount):
                writer.writerow([table.GetValue(rowIndex, columnIndex).ToString() for columnIndex in range(columnCount)])

    @staticmethod
    def exportStructuredSummaryToJson(outputJsonPath: Path, summaryData: dict[str, Any] | list[dict[str, Any]]) -> None:
        outputJsonPath.parent.mkdir(parents=True, exist_ok=True)
        with outputJsonPath.open("w", encoding="utf-8", newline="\n") as jsonFile:
            json.dump(summaryData, jsonFile, indent=2, sort_keys=True)

    @staticmethod
    def exportKeyValueDictionaryToCsv(outputCsvPath: Path, valuesByKey: dict[str, Any]) -> None:
        outputCsvPath.parent.mkdir(parents=True, exist_ok=True)
        with outputCsvPath.open("w", encoding="utf-8", newline="") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(("Metric", "Value"))
            for metricName in sorted(valuesByKey.keys()):
                writer.writerow((metricName, valuesByKey.get(metricName, "")))

    def collectPlanMetricsSnapshot(self, parameterNode: SurgicalVision3D_PlannerParameterNode) -> dict[str, Any]:
        planSummaryRows = self._tableNodeToDictionaries(parameterNode.planSummaryTable)
        coverageRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Coverage Summary"))
        safetyRows = self._tableNodeToDictionaries(parameterNode.structureSafetySummaryTable)
        coordinationRows = self._tableNodeToDictionaries(parameterNode.probeCoordinationSummaryTable)
        planRow = planSummaryRows[0] if len(planSummaryRows) > 0 else {}
        coverageRow = coverageRows[0] if len(coverageRows) > 0 else {}
        coordinationRow = coordinationRows[0] if len(coordinationRows) > 0 else {}

        metrics: dict[str, Any] = {
            "MasterTrajectoryLocked": bool(parameterNode.masterTrajectoryLocked),
            "TrajectoryPlanningMode": str(parameterNode.trajectoryPlanningMode or "Single"),
            "DerivedTrajectoryCount": int(parameterNode.derivedTrajectoryCount),
            "MamThresholdMm": float(parameterNode.mamMm),
            "CoaxialTechnique": str(parameterNode.coaxialTechnique or "PullBack"),
            "CoaxialSpareMm": float(parameterNode.coaxialSpareMm),
        }

        trajectoryCount = self._firstNumericValue([planRow], ["Trajectory Count", "TrajectoryCount"])
        minMargin = self._firstNumericValue([planRow], ["Minimum Signed Margin (mm)", "MinSignedMarginMm"])
        meanMargin = self._firstNumericValue([planRow], ["Mean Signed Margin (mm)", "MeanSignedMarginMm"])
        medianMargin = self._firstNumericValue([planRow], ["Median Signed Margin (mm)", "MedianSignedMarginMm"])
        coveragePercent = self._firstNumericValue([coverageRow], ["CoveragePercent", "Coverage Percent", "Coverage (%)"])
        if trajectoryCount is not None:
            metrics["TrajectoryCount"] = int(round(trajectoryCount))
        if minMargin is not None:
            metrics["MinSignedMarginMm"] = float(minMargin)
        if meanMargin is not None:
            metrics["MeanSignedMarginMm"] = float(meanMargin)
        if medianMargin is not None:
            metrics["MedianSignedMarginMm"] = float(medianMargin)
        if coveragePercent is not None:
            metrics["CoveragePercent"] = float(coveragePercent)

        coordinationGatePassRaw = self._firstStringValue([coordinationRow], ["Coordination Gate Pass", "CoordinationGatePass"])
        if coordinationGatePassRaw:
            metrics["CoordinationGatePass"] = bool(self._coerceBoolean(coordinationGatePassRaw, defaultValue=True))

        structureDistances = [
            self._firstNumericValue([row], ["Minimum Distance (mm)", "MinDistanceMm"])
            for row in safetyRows
        ]
        finiteDistances = [value for value in structureDistances if value is not None and math.isfinite(value)]
        if len(finiteDistances) > 0:
            metrics["WorstStructureMinDistanceMm"] = float(min(finiteDistances))

        try:
            mamSummary = json.loads(str(parameterNode.mamAssessmentSummaryJson or "{}"))
        except Exception:
            mamSummary = {}
        if isinstance(mamSummary, dict):
            if "MamPass" in mamSummary:
                metrics["MamPass"] = bool(mamSummary.get("MamPass"))
            minimumAchievedMargin = mamSummary.get("MinimumAchievedMarginMm")
            try:
                if minimumAchievedMargin is not None and math.isfinite(float(minimumAchievedMargin)):
                    metrics["MinimumAchievedMarginMm"] = float(minimumAchievedMargin)
            except Exception:
                pass

        try:
            coaxialSummary = json.loads(str(parameterNode.coaxialPlanSummaryJson or "{}"))
        except Exception:
            coaxialSummary = {}
        if isinstance(coaxialSummary, dict):
            activeElementLengthMm = coaxialSummary.get("activeElementLengthMm")
            pushThroughOffsetMm = coaxialSummary.get("pushThroughOffsetMm")
            try:
                if activeElementLengthMm is not None and math.isfinite(float(activeElementLengthMm)):
                    metrics["CoaxialActiveElementLengthMm"] = float(activeElementLengthMm)
            except Exception:
                pass
            try:
                if pushThroughOffsetMm is not None and math.isfinite(float(pushThroughOffsetMm)):
                    metrics["CoaxialPushThroughOffsetMm"] = float(pushThroughOffsetMm)
            except Exception:
                pass

        return metrics

    def exportPlanningScreenshots(self, outputDirectory: Path) -> tuple[list[Path], list[str], list[dict[str, str]]]:
        outputDirectory.mkdir(parents=True, exist_ok=True)
        savedCapturePaths: list[Path] = []
        warnings: list[str] = []
        captureEntries: list[dict[str, str]] = []

        def captureWidget(captureName: str, widget) -> None:
            fileName = f"{captureName}.png"
            entry = {
                "CaptureName": captureName,
                "FileName": fileName,
                "Status": "Skipped",
                "Warning": "",
            }
            if widget is None:
                warningText = f"Screenshot '{captureName}' skipped: widget is unavailable."
                warnings.append(warningText)
                entry["Status"] = "Skipped"
                entry["Warning"] = warningText
                captureEntries.append(entry)
                return

            try:
                pixmap = widget.grab() if hasattr(widget, "grab") else None
                if (pixmap is None or pixmap.isNull()) and hasattr(qt.QPixmap, "grabWidget"):
                    pixmap = qt.QPixmap.grabWidget(widget)
                if pixmap is None or pixmap.isNull():
                    warningText = f"Screenshot '{captureName}' skipped: capture returned an empty pixmap."
                    warnings.append(warningText)
                    entry["Status"] = "Skipped"
                    entry["Warning"] = warningText
                    captureEntries.append(entry)
                    return

                outputPath = outputDirectory / fileName
                if not bool(pixmap.save(str(outputPath), "PNG")):
                    warningText = f"Screenshot '{captureName}' failed to save at '{outputPath}'."
                    warnings.append(warningText)
                    entry["Status"] = "Failed"
                    entry["Warning"] = warningText
                    captureEntries.append(entry)
                    return

                savedCapturePaths.append(outputPath)
                entry["Status"] = "Saved"
                captureEntries.append(entry)
            except Exception as exc:
                warningText = f"Screenshot '{captureName}' failed: {exc}"
                warnings.append(warningText)
                entry["Status"] = "Failed"
                entry["Warning"] = warningText
                captureEntries.append(entry)

        mainWindow = None
        try:
            mainWindow = slicer.util.mainWindow()
        except Exception:
            mainWindow = None
        captureWidget("main_window", mainWindow)

        layoutManager = slicer.app.layoutManager() if hasattr(slicer.app, "layoutManager") else None
        threeDView = None
        if layoutManager and hasattr(layoutManager, "threeDViewCount") and int(layoutManager.threeDViewCount) > 0:
            try:
                threeDWidget = layoutManager.threeDWidget(0)
                if threeDWidget and hasattr(threeDWidget, "threeDView"):
                    threeDView = threeDWidget.threeDView()
            except Exception:
                threeDView = None
        captureWidget("three_d_view", threeDView)

        for sliceName in ("Red", "Yellow", "Green"):
            sliceView = None
            if layoutManager and hasattr(layoutManager, "sliceWidget"):
                try:
                    sliceWidget = layoutManager.sliceWidget(sliceName)
                    if sliceWidget and hasattr(sliceWidget, "sliceView"):
                        sliceView = sliceWidget.sliceView()
                except Exception:
                    sliceView = None
            captureWidget(f"slice_{sliceName.lower()}", sliceView)

        return savedCapturePaths, warnings, captureEntries

    def exportCurrentSceneBundle(self, outputDirectory: Path) -> tuple[list[Path], list[str], dict[str, Any]]:
        outputDirectory.mkdir(parents=True, exist_ok=True)
        exportedPaths: list[Path] = []
        warnings: list[str] = []
        sceneSummary: dict[str, Any] = {
            "SceneBundleFile": "planning_scene.mrb",
            "SceneBundleStatus": "NotAttempted",
            "NodeCount": 0,
        }

        sceneBundlePath = outputDirectory / "planning_scene.mrb"
        saveScene = getattr(slicer.util, "saveScene", None)
        if callable(saveScene):
            try:
                if bool(saveScene(str(sceneBundlePath))):
                    exportedPaths.append(sceneBundlePath)
                    sceneSummary["SceneBundleStatus"] = "Saved"
                else:
                    warningText = f"Scene bundle export failed: saveScene returned false for '{sceneBundlePath}'."
                    warnings.append(warningText)
                    sceneSummary["SceneBundleStatus"] = "Failed"
            except Exception as exc:
                warningText = f"Scene bundle export failed: {exc}"
                warnings.append(warningText)
                sceneSummary["SceneBundleStatus"] = "Failed"
        else:
            warningText = "Scene bundle export skipped: slicer.util.saveScene is unavailable."
            warnings.append(warningText)
            sceneSummary["SceneBundleStatus"] = "Skipped"

        nodeInventory: list[dict[str, str]] = []
        scene = slicer.mrmlScene
        if scene:
            for nodeIndex in range(int(scene.GetNumberOfNodes())):
                node = scene.GetNthNode(nodeIndex)
                if not node:
                    continue
                nodeInventory.append(
                    {
                        "NodeID": str(node.GetID() or ""),
                        "NodeName": str(node.GetName() or ""),
                        "NodeClassName": str(node.GetClassName() or ""),
                    }
                )
        sceneSummary["NodeCount"] = int(len(nodeInventory))

        nodeInventoryJsonPath = outputDirectory / "scene_node_inventory.json"
        self.exportStructuredSummaryToJson(nodeInventoryJsonPath, nodeInventory)
        exportedPaths.append(nodeInventoryJsonPath)

        nodeInventoryCsvPath = outputDirectory / "scene_node_inventory.csv"
        with nodeInventoryCsvPath.open("w", encoding="utf-8", newline="") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(("NodeID", "NodeName", "NodeClassName"))
            for nodeEntry in nodeInventory:
                writer.writerow(
                    (
                        nodeEntry.get("NodeID", ""),
                        nodeEntry.get("NodeName", ""),
                        nodeEntry.get("NodeClassName", ""),
                    )
                )
        exportedPaths.append(nodeInventoryCsvPath)

        return exportedPaths, warnings, sceneSummary

    def collectCurrentPlanExportData(
        self,
        parameterNode: SurgicalVision3D_PlannerParameterNode,
        exportConfig: PlanExportConfig,
    ) -> tuple[dict[str, Any], list[tuple[str, vtkMRMLTableNode]]]:
        generatedProbeCount = len(self.resolveExistingNodeIDs(self.deserializeNodeIDs(parameterNode.generatedProbeNodeIDs)))
        currentPlanSummary = {
            "ReferenceProbeSegmentationID": parameterNode.referenceProbeSegmentation.GetID() if parameterNode.referenceProbeSegmentation else "",
            "ReferenceProbeSegmentationName": parameterNode.referenceProbeSegmentation.GetName() if parameterNode.referenceProbeSegmentation else "",
            "TumorSegmentationID": parameterNode.tumorSegmentation.GetID() if parameterNode.tumorSegmentation else "",
            "TumorSegmentationName": parameterNode.tumorSegmentation.GetName() if parameterNode.tumorSegmentation else "",
            "RiskStructuresSegmentationID": parameterNode.riskStructuresSegmentation.GetID() if parameterNode.riskStructuresSegmentation else "",
            "RiskStructuresSegmentationName": parameterNode.riskStructuresSegmentation.GetName() if parameterNode.riskStructuresSegmentation else "",
            "CombinedProbeSegmentationID": parameterNode.combinedProbeSegmentation.GetID() if parameterNode.combinedProbeSegmentation else "",
            "CombinedProbeSegmentationName": parameterNode.combinedProbeSegmentation.GetName() if parameterNode.combinedProbeSegmentation else "",
            "GeneratedProbeCount": int(generatedProbeCount),
            "GeneratedTrajectoryLineCount": int(len(self.resolveExistingNodeIDs(self.deserializeNodeIDs(parameterNode.generatedTrajectoryLineIDs)))),
            "TrajectoryPlanningMode": str(parameterNode.trajectoryPlanningMode or "Single"),
            "DerivedTrajectoryCount": int(parameterNode.derivedTrajectoryCount),
            "DerivedTrajectoryRadiusMm": float(parameterNode.derivedTrajectoryRadiusMm),
            "DerivedTrajectoryAngleOffsetDeg": float(parameterNode.derivedTrajectoryAngleOffsetDeg),
            "IncludeMasterTrajectoryInArray": bool(parameterNode.includeMasterTrajectoryInArray),
            "ExportMode": exportConfig.exportMode,
            "SelectedExportScenarioID": exportConfig.selectedExportScenarioID,
        }

        tableExports: list[tuple[str, vtkMRMLTableNode]] = []
        seenNodeIDs: set[str] = set()

        def addTable(filename: str, tableNode: vtkMRMLTableNode | None) -> None:
            if not tableNode or not slicer.mrmlScene.IsNodePresent(tableNode):
                return
            if tableNode.GetID() in seenNodeIDs:
                return
            seenNodeIDs.add(tableNode.GetID())
            tableExports.append((filename, tableNode))

        if exportConfig.includeTrajectoryTables:
            addTable("trajectory_summary.csv", parameterNode.trajectorySummaryTable)
            addTable("derived_trajectory_bundle_summary.csv", parameterNode.derivedTrajectorySummaryTable)
        if exportConfig.includeWorkingPlan:
            addTable("plan_summary.csv", parameterNode.planSummaryTable)
            addTable("margin_threshold_summary.csv", parameterNode.marginThresholdSummaryTable)
        if exportConfig.includeSafetyTables:
            addTable("structure_safety_summary.csv", parameterNode.structureSafetySummaryTable)
            addTable("structure_safety_threshold_summary.csv", parameterNode.structureSafetyThresholdSummaryTable)
        if exportConfig.includeCoordinationTables:
            addTable("probe_coordination_constraint_settings.csv", parameterNode.probeCoordinationConstraintSettingsTable)
            addTable("probe_pair_coordination_summary.csv", parameterNode.probePairCoordinationSummaryTable)
            addTable("probe_coordination_summary.csv", parameterNode.probeCoordinationSummaryTable)
            addTable("no_touch_summary.csv", parameterNode.noTouchSummaryTable)
        addTable("cohort_execution_summary.csv", parameterNode.cohortExecutionSummaryTable)
        addTable("cohort_case_summary.csv", parameterNode.cohortCaseSummaryTable)
        addTable("cohort_aggregate_metrics.csv", parameterNode.cohortAggregateMetricsTable)
        addTable("cohort_comparison_summary.csv", parameterNode.cohortComparisonSummaryTable)
        addTable("reproducibility_package_summary.csv", parameterNode.reproducibilityPackageSummaryTable)
        addTable("reproducibility_manifest_preview.csv", parameterNode.reproducibilityManifestPreviewTable)
        addTable("reproducibility_artifact_index.csv", parameterNode.reproducibilityArtifactIndexTable)

        if exportConfig.includeCoverageTables:
            addTable("coverage_summary.csv", self._findFirstTableNodeByName("SV3D Coverage Summary"))
            addTable("multi_target_coverage_summary.csv", self._findFirstTableNodeByName("SV3D Multi-Target Coverage Summary"))
        if exportConfig.includeFeasibilityTables:
            addTable("trajectory_feasibility_summary.csv", self._findFirstTableNodeByName("SV3D Trajectory Feasibility Summary"))
            addTable("plan_trajectory_feasibility_summary.csv", self._findFirstTableNodeByName("SV3D Plan Trajectory Feasibility Summary"))
            addTable("candidate_feasibility_summary.csv", self._findFirstTableNodeByName("SV3D Candidate Feasibility Summary"))
        if exportConfig.includeScenarioComparison:
            addTable("scenario_comparison.csv", self._findFirstTableNodeByName("SV3D Scenario Comparison"))
            addTable("scenario_delta_comparison.csv", self._findFirstTableNodeByName("SV3D Scenario Delta Comparison"))
            addTable("scenario_frontier_summary.csv", self._findFirstTableNodeByName("SV3D Scenario Frontier Summary"))
        if exportConfig.includeRecommendationOutputs:
            addTable("feasible_candidate_recommendation.csv", self._findFirstTableNodeByName("SV3D Feasible Candidate Recommendation"))
            addTable("scenario_recommendation_summary.csv", self._findFirstTableNodeByName("SV3D Scenario Recommendation Summary"))

        return currentPlanSummary, tableExports

    def collectScenarioExportData(self, selectedScenarioID: str) -> dict[str, Any]:
        if not selectedScenarioID.strip():
            raise ValueError("Selected scenario export mode requires a non-empty scenario ID.")

        scenarioSummary: dict[str, Any] = {
            "SelectedScenarioID": selectedScenarioID,
            "ScenarioName": "",
            "Source": "Unavailable",
            "Notes": "Scenario registry table not found.",
        }

        scenarioRegistryTable = self._findFirstTableNodeByName("SV3D Scenario Registry")
        if not scenarioRegistryTable:
            return scenarioSummary

        rows = self._tableNodeToDictionaries(scenarioRegistryTable)
        for row in rows:
            scenarioIDValue = str(row.get("ScenarioID", "")).strip()
            if scenarioIDValue == selectedScenarioID.strip():
                scenarioSummary["ScenarioName"] = str(row.get("ScenarioName", ""))
                scenarioSummary["Source"] = "SV3D Scenario Registry"
                scenarioSummary["Notes"] = ""
                scenarioSummary["ScenarioRow"] = row
                return scenarioSummary

        scenarioSummary["Notes"] = f"Scenario ID '{selectedScenarioID}' was not found in SV3D Scenario Registry."
        return scenarioSummary

    def buildPlanExportManifest(
        self,
        parameterNode: SurgicalVision3D_PlannerParameterNode,
        exportConfig: PlanExportConfig,
        exportSequence: int,
        filesExported: Sequence[str],
        selectedScenarioSummary: dict[str, Any] | None = None,
    ) -> PlanExportManifest:
        selectedScenarioSummary = selectedScenarioSummary or {}
        tumorSegmentID = ""
        tumorSegmentName = ""
        if parameterNode.planSummaryTable and slicer.mrmlScene.IsNodePresent(parameterNode.planSummaryTable):
            summaryRows = self._tableNodeToDictionaries(parameterNode.planSummaryTable)
            if len(summaryRows) > 0:
                tumorSegmentID = str(summaryRows[0].get("Tumor Segment ID", ""))
                tumorSegmentName = str(summaryRows[0].get("Tumor Segment Name", ""))

        includeFlags = {
            "includeWorkingPlan": bool(exportConfig.includeWorkingPlan),
            "includeSelectedScenario": bool(exportConfig.includeSelectedScenario),
            "includeScenarioComparison": bool(exportConfig.includeScenarioComparison),
            "includeRecommendationOutputs": bool(exportConfig.includeRecommendationOutputs),
            "includeTrajectoryTables": bool(exportConfig.includeTrajectoryTables),
            "includeSafetyTables": bool(exportConfig.includeSafetyTables),
            "includeCoverageTables": bool(exportConfig.includeCoverageTables),
            "includeFeasibilityTables": bool(exportConfig.includeFeasibilityTables),
            "includeCoordinationTables": bool(exportConfig.includeCoordinationTables),
        }

        return PlanExportManifest(
            exportId=f"SV3D-Export-{int(exportSequence):04d}",
            exportTimestampISO=datetime.now().isoformat(timespec="seconds"),
            exportSequence=int(exportSequence),
            exportMode=exportConfig.exportMode,
            exportBaseName=exportConfig.exportBaseName,
            selectedScenarioID=str(exportConfig.selectedExportScenarioID),
            selectedScenarioName=str(selectedScenarioSummary.get("ScenarioName", "")),
            profileSourceMode="referenceSegmentation",
            presetID="",
            presetName="",
            targetSegmentID=tumorSegmentID,
            targetSegmentName=tumorSegmentName,
            filesExported=[str(filePath) for filePath in filesExported],
            includeFlags=includeFlags,
            notes=str(selectedScenarioSummary.get("Notes", "")),
        )

    def exportPlanBundle(
        self,
        parameterNode: SurgicalVision3D_PlannerParameterNode,
        exportConfig: PlanExportConfig,
    ) -> dict[str, Any]:
        if not exportConfig.exportBaseName.strip():
            raise ValueError("Export base name is required.")

        exportDirectory = exportConfig.exportDirectory.strip() or str(Path(slicer.app.temporaryPath) / "SurgicalVision3D_PlannerExports")
        exportRoot = Path(exportDirectory)
        exportRoot.mkdir(parents=True, exist_ok=True)

        exportSequence = max(1, int(exportConfig.lastExportSequence) + 1)
        bundlePath = self.buildDeterministicBundlePath(str(exportRoot), exportConfig.exportBaseName, exportSequence)
        while bundlePath.exists():
            exportSequence += 1
            bundlePath = self.buildDeterministicBundlePath(str(exportRoot), exportConfig.exportBaseName, exportSequence)

        bundlePath.mkdir(parents=True, exist_ok=False)
        tablesDirectory = bundlePath / "tables"
        provenanceDirectory = bundlePath / "provenance"
        metricsDirectory = bundlePath / "metrics"
        screenshotsDirectory = bundlePath / "screenshots"
        sceneDirectory = bundlePath / "scene"
        tablesDirectory.mkdir(parents=True, exist_ok=True)
        provenanceDirectory.mkdir(parents=True, exist_ok=True)
        metricsDirectory.mkdir(parents=True, exist_ok=True)
        screenshotsDirectory.mkdir(parents=True, exist_ok=True)
        sceneDirectory.mkdir(parents=True, exist_ok=True)

        currentPlanSummary, tableExports = self.collectCurrentPlanExportData(parameterNode, exportConfig)
        exportedFiles: list[str] = []
        exportedFileSet: set[str] = set()
        exportWarnings: list[str] = []

        def addExportedFile(absolutePath: Path) -> None:
            relativePath = str(absolutePath.relative_to(bundlePath)).replace("\\", "/")
            if relativePath in exportedFileSet:
                return
            exportedFileSet.add(relativePath)
            exportedFiles.append(relativePath)

        planSummaryPath = bundlePath / "plan_summary.json"
        self.exportStructuredSummaryToJson(planSummaryPath, currentPlanSummary)
        addExportedFile(planSummaryPath)

        selectedScenarioSummary: dict[str, Any] = {}
        shouldIncludeSelectedScenario = bool(exportConfig.includeSelectedScenario or exportConfig.exportMode == "SelectedScenario")
        if shouldIncludeSelectedScenario:
            selectedScenarioSummary = self.collectScenarioExportData(exportConfig.selectedExportScenarioID)
            scenarioSummaryPath = bundlePath / "scenario_summary.json"
            self.exportStructuredSummaryToJson(scenarioSummaryPath, selectedScenarioSummary)
            addExportedFile(scenarioSummaryPath)

        for tableFileName, tableNode in tableExports:
            outputCsvPath = tablesDirectory / tableFileName
            self.exportTableNodeToCsv(tableNode, outputCsvPath)
            addExportedFile(outputCsvPath)

        scenarioRegistryTable = self._findFirstTableNodeByName("SV3D Scenario Registry")
        if scenarioRegistryTable:
            scenarioRegistryJsonPath = provenanceDirectory / "scenario_registry.json"
            self.exportStructuredSummaryToJson(scenarioRegistryJsonPath, self._tableNodeToDictionaries(scenarioRegistryTable))
            addExportedFile(scenarioRegistryJsonPath)

        recommendationSummaryTable = self._findFirstTableNodeByName("SV3D Feasible Candidate Recommendation")
        if recommendationSummaryTable:
            recommendationJsonPath = provenanceDirectory / "recommendation_summary.json"
            self.exportStructuredSummaryToJson(recommendationJsonPath, self._tableNodeToDictionaries(recommendationSummaryTable))
            addExportedFile(recommendationJsonPath)

        metricsSnapshot = self.collectPlanMetricsSnapshot(parameterNode)
        metricsSnapshotJsonPath = metricsDirectory / "plan_metrics_snapshot.json"
        self.exportStructuredSummaryToJson(metricsSnapshotJsonPath, metricsSnapshot)
        addExportedFile(metricsSnapshotJsonPath)
        metricsSnapshotCsvPath = tablesDirectory / "plan_metrics_snapshot.csv"
        self.exportKeyValueDictionaryToCsv(metricsSnapshotCsvPath, metricsSnapshot)
        addExportedFile(metricsSnapshotCsvPath)

        screenshotPaths, screenshotWarnings, screenshotManifestEntries = self.exportPlanningScreenshots(screenshotsDirectory)
        exportWarnings.extend(screenshotWarnings)
        screenshotsManifestPath = screenshotsDirectory / "capture_manifest.json"
        self.exportStructuredSummaryToJson(screenshotsManifestPath, screenshotManifestEntries)
        addExportedFile(screenshotsManifestPath)
        for screenshotPath in screenshotPaths:
            addExportedFile(screenshotPath)

        scenePaths, sceneWarnings, sceneManifest = self.exportCurrentSceneBundle(sceneDirectory)
        exportWarnings.extend(sceneWarnings)
        sceneManifestPath = sceneDirectory / "scene_manifest.json"
        self.exportStructuredSummaryToJson(sceneManifestPath, sceneManifest)
        addExportedFile(sceneManifestPath)
        for scenePath in scenePaths:
            addExportedFile(scenePath)

        uniqueWarnings = sorted(set(str(warning) for warning in exportWarnings if str(warning).strip()))
        if len(uniqueWarnings) > 0:
            warningsPath = provenanceDirectory / "export_warnings.json"
            self.exportStructuredSummaryToJson(warningsPath, {"warnings": uniqueWarnings})
            addExportedFile(warningsPath)

        manifest = self.buildPlanExportManifest(
            parameterNode=parameterNode,
            exportConfig=exportConfig,
            exportSequence=exportSequence,
            filesExported=exportedFiles,
            selectedScenarioSummary=selectedScenarioSummary,
        )
        if len(uniqueWarnings) > 0:
            warningNote = f"Export warnings: {len(uniqueWarnings)} (see provenance/export_warnings.json)."
            manifest.notes = f"{str(manifest.notes or '').strip()} {warningNote}".strip()
        manifestPath = bundlePath / "manifest.json"
        manifestRelativePath = str(manifestPath.relative_to(bundlePath)).replace("\\", "/")
        manifest.filesExported = [manifestRelativePath, *exportedFiles]
        self.exportStructuredSummaryToJson(manifestPath, asdict(manifest))
        fileCount = int(len(manifest.filesExported))
        statusText = "SuccessWithWarnings" if len(uniqueWarnings) > 0 else "Success"

        return {
            "manifest": manifest,
            "bundlePath": str(bundlePath),
            "exportDirectory": str(exportRoot),
            "exportSequence": int(exportSequence),
            "fileCount": fileCount,
            "status": statusText,
            "selectedScenarioName": str(selectedScenarioSummary.get("ScenarioName", "")),
            "warningCount": int(len(uniqueWarnings)),
            "warnings": uniqueWarnings,
        }

    @staticmethod
    def buildDeterministicReproPackagePath(outputDirectory: str, packageBaseName: str, packageSequence: int) -> Path:
        bundleBaseName = SurgicalVision3D_PlannerLogic.sanitizeExportBaseName(packageBaseName)
        rootDirectory = Path(outputDirectory)
        return rootDirectory / f"{bundleBaseName}_{int(packageSequence):04d}"

    @staticmethod
    def computeArtifactIntegritySummary(artifactPath: Path, includeHash: bool = True) -> dict[str, Any]:
        sizeBytes = int(artifactPath.stat().st_size) if artifactPath.exists() else 0
        sha256Value = ""
        # Keep hashing cheap for reviewer package generation on large datasets.
        if includeHash and artifactPath.exists() and artifactPath.is_file() and sizeBytes <= 5_000_000:
            sha256Hasher = hashlib.sha256()
            with artifactPath.open("rb") as binaryFile:
                while True:
                    chunk = binaryFile.read(65536)
                    if not chunk:
                        break
                    sha256Hasher.update(chunk)
            sha256Value = sha256Hasher.hexdigest()
        return {
            "sizeBytes": sizeBytes,
            "sha256": sha256Value,
        }

    @staticmethod
    def _resolveResourcePath(relativePath: str) -> Path:
        return (Path(__file__).resolve().parent / relativePath).resolve()

    def discoverBundledSampleScenes(self) -> list[tuple[str, Path]]:
        cohortResourcesDirectory = self._resolveResourcePath("Resources/Cohorts")
        if not cohortResourcesDirectory.exists():
            return []

        bundledScenePaths = sorted(
            (
                candidatePath
                for candidatePath in cohortResourcesDirectory.rglob("*.mrml")
                if candidatePath.is_file()
            ),
            key=lambda candidatePath: (candidatePath.parent.name.lower(), candidatePath.name.lower()),
        )
        bundledScenes: list[tuple[str, Path]] = []
        for scenePath in bundledScenePaths:
            caseName = scenePath.parent.name.strip() or scenePath.parent.as_posix()
            sceneStem = scenePath.stem.strip()
            displayName = caseName if sceneStem.lower() == "demoscene" else f"{caseName} - {sceneStem}"
            bundledScenes.append((displayName, scenePath))
        return bundledScenes

    @staticmethod
    def loadBundledSampleScene(scenePath: str | Path) -> None:
        resolvedScenePath = Path(scenePath).resolve()
        if not resolvedScenePath.exists():
            raise ValueError(f"Sample scene file was not found: {resolvedScenePath}")
        if resolvedScenePath.suffix.lower() != ".mrml":
            raise ValueError(f"Unsupported sample scene format: {resolvedScenePath.name}")

        # Prefer direct case-asset loading for bundled cohort scenes to avoid noisy scene-import warnings
        # when optional subject-hierarchy references differ across Slicer builds.
        if SurgicalVision3D_PlannerLogic._loadBundledCaseAssetsFromDirectory(resolvedScenePath.parent):
            return

        unresolvedMissingFiles = SurgicalVision3D_PlannerLogic._repairMissingSceneReferences(resolvedScenePath)
        if len(unresolvedMissingFiles) > 0:
            formattedMissingFiles = "\n".join(f"- {missingPath}" for missingPath in unresolvedMissingFiles[:12])
            raise RuntimeError(
                "Sample scene has missing referenced files that could not be repaired:\n"
                f"{formattedMissingFiles}"
            )
        if slicer.util.loadScene(str(resolvedScenePath), {"clear": True}):
            return

        logging.warning(
            "Scene load returned false for '%s'. Falling back to direct asset loading.",
            resolvedScenePath,
        )
        if SurgicalVision3D_PlannerLogic._loadBundledCaseAssetsFromDirectory(resolvedScenePath.parent):
            return
        raise RuntimeError(f"Failed to load sample scene: {resolvedScenePath}")

    @staticmethod
    def _loadBundledCaseAssetsFromDirectory(caseDirectory: Path) -> bool:
        if not caseDirectory.exists():
            return False

        scene = slicer.mrmlScene
        startedStates: list[int] = []
        loadedAny = False

        for stateName in ("BatchProcessState", "ImportState"):
            stateId = getattr(scene, stateName, None)
            if stateId is None:
                continue
            try:
                scene.StartState(int(stateId))
                startedStates.append(int(stateId))
            except Exception:
                pass

        try:
            scene.Clear(0)

            volumeCandidates = sorted(
                (
                    candidatePath
                    for candidatePath in caseDirectory.glob("*.nrrd")
                    if candidatePath.is_file() and not candidatePath.name.lower().endswith(".seg.nrrd")
                ),
                key=lambda candidatePath: (-int(candidatePath.stat().st_size), candidatePath.name.lower()),
            )
            for volumePath in volumeCandidates[:1]:
                try:
                    loadedAny = bool(slicer.util.loadVolume(str(volumePath))) or loadedAny
                except Exception:
                    logging.exception("Failed to load fallback sample-case volume: %s", volumePath)

            segmentationCandidates = sorted(
                (
                    candidatePath
                    for candidatePath in caseDirectory.glob("*.seg.nrrd")
                    if candidatePath.is_file()
                ),
                key=lambda candidatePath: candidatePath.name.lower(),
            )
            for segmentationPath in segmentationCandidates:
                try:
                    loadedAny = bool(slicer.util.loadSegmentation(str(segmentationPath))) or loadedAny
                except Exception:
                    logging.exception("Failed to load fallback sample-case segmentation: %s", segmentationPath)

            markupsCandidatesByPath: dict[str, Path] = {}
            for markupsPattern in ("*.mrk.json", "*.fcsv"):
                for candidatePath in caseDirectory.glob(markupsPattern):
                    if candidatePath.is_file():
                        normalizedPath = SurgicalVision3D_PlannerLogic._normalizeFilesystemPath(candidatePath)
                        if normalizedPath not in markupsCandidatesByPath:
                            markupsCandidatesByPath[normalizedPath] = candidatePath
            markupsCandidates = sorted(
                markupsCandidatesByPath.values(),
                key=lambda candidatePath: candidatePath.name.lower(),
            )
            for markupsPath in markupsCandidates:
                try:
                    loadedAny = SurgicalVision3D_PlannerLogic._loadMarkupsFile(markupsPath) or loadedAny
                except Exception:
                    logging.exception("Failed to load fallback sample-case markups: %s", markupsPath)

            logic = SurgicalVision3D_PlannerLogic()
            for segmentationNode in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
                logic._ensureSegmentationReferenceImageGeometry(segmentationNode)
        finally:
            for stateId in reversed(startedStates):
                try:
                    scene.EndState(int(stateId))
                except Exception:
                    pass

        SurgicalVision3D_PlannerLogic().ensureReferenceProbeTemplatesLoaded()
        return loadedAny

    @staticmethod
    def _loadMarkupsFile(markupsPath: Path) -> bool:
        loadMarkups = getattr(slicer.util, "loadMarkups", None)
        if callable(loadMarkups):
            return bool(loadMarkups(str(markupsPath)))

        loadNodeFromFile = getattr(slicer.util, "loadNodeFromFile", None)
        if callable(loadNodeFromFile):
            loadedNode = loadNodeFromFile(str(markupsPath), "MarkupsFile")
            return loadedNode is not None

        loadMarkupsFiducialList = getattr(slicer.util, "loadMarkupsFiducialList", None)
        if callable(loadMarkupsFiducialList):
            return bool(loadMarkupsFiducialList(str(markupsPath)))

        return False

    def resolveUsableReferenceProbeSegmentation(
        self,
        selectedReferenceProbeSegmentation: vtkMRMLSegmentationNode | None,
    ) -> vtkMRMLSegmentationNode | None:
        if selectedReferenceProbeSegmentation and self.segmentationSegmentCount(selectedReferenceProbeSegmentation) > 0:
            return selectedReferenceProbeSegmentation

        templateNodes = self.ensureReferenceProbeTemplatesLoaded()
        if len(templateNodes) == 0:
            return selectedReferenceProbeSegmentation

        if not selectedReferenceProbeSegmentation:
            for templateNode in templateNodes:
                if self.segmentationSegmentCount(templateNode) > 0:
                    return templateNode
            return None

        selectedSourcePath = selectedReferenceProbeSegmentation.GetAttribute(REFERENCE_PROBE_TEMPLATE_SOURCE_PATH_ATTRIBUTE) or ""
        normalizedSelectedSourcePath = (
            self._normalizeFilesystemPath(selectedSourcePath)
            if selectedSourcePath
            else ""
        )
        selectedName = str(selectedReferenceProbeSegmentation.GetName() or "").strip().lower()
        for templateNode in templateNodes:
            if self.segmentationSegmentCount(templateNode) <= 0:
                continue
            templateSourcePath = templateNode.GetAttribute(REFERENCE_PROBE_TEMPLATE_SOURCE_PATH_ATTRIBUTE) or ""
            normalizedTemplateSourcePath = (
                self._normalizeFilesystemPath(templateSourcePath)
                if templateSourcePath
                else ""
            )
            if normalizedSelectedSourcePath and normalizedTemplateSourcePath == normalizedSelectedSourcePath:
                return templateNode
            if selectedName and str(templateNode.GetName() or "").strip().lower() == selectedName:
                return templateNode

        return selectedReferenceProbeSegmentation

    @staticmethod
    def _repairMissingSceneReferences(scenePath: Path) -> list[Path]:
        referencedFiles = SurgicalVision3D_PlannerLogic._referencedSceneFiles(scenePath)
        if len(referencedFiles) == 0:
            return []

        unresolvedMissingFiles: list[Path] = []
        for referencedFilePath in referencedFiles:
            if referencedFilePath.exists():
                continue
            if SurgicalVision3D_PlannerLogic._tryRepairMissingSegmentationFileFromStl(scenePath, referencedFilePath):
                continue
            unresolvedMissingFiles.append(referencedFilePath)
        return unresolvedMissingFiles

    @staticmethod
    def _referencedSceneFiles(scenePath: Path) -> list[Path]:
        try:
            sceneTree = ET.parse(str(scenePath))
        except Exception:
            logging.exception("Failed to parse sample scene for missing-reference preflight: %s", scenePath)
            return []

        referencedFilePaths: set[Path] = set()
        for sceneElement in sceneTree.iter():
            for attributeName, attributeValue in sceneElement.attrib.items():
                if attributeName != "fileName" and not attributeName.startswith("fileListMember"):
                    continue
                decodedPath = str(unquote(attributeValue or "")).strip()
                if not decodedPath:
                    continue
                candidatePath = Path(decodedPath)
                if not candidatePath.is_absolute():
                    candidatePath = (scenePath.parent / candidatePath).resolve()
                else:
                    candidatePath = candidatePath.resolve()
                referencedFilePaths.add(candidatePath)
        return sorted(referencedFilePaths, key=lambda candidatePath: candidatePath.as_posix().lower())

    @staticmethod
    def _tryRepairMissingSegmentationFileFromStl(scenePath: Path, missingFilePath: Path) -> bool:
        if not str(missingFilePath.name).lower().endswith(".seg.vtm"):
            return False

        baseName = missingFilePath.name[:-len(".seg.vtm")]
        moduleResourcesDirectory = Path(__file__).resolve().parent / "Resources"
        sceneResourcesDirectory = scenePath.parent.parent.parent if len(scenePath.parents) >= 3 else moduleResourcesDirectory
        stlCandidates = [
            (missingFilePath.parent / f"{baseName}.stl").resolve(),
            (sceneResourcesDirectory / "geometries" / f"{baseName}.stl").resolve(),
            (moduleResourcesDirectory / "geometries" / f"{baseName}.stl").resolve(),
        ]
        uniqueStlCandidates: list[Path] = []
        seenCandidates: set[str] = set()
        for stlCandidate in stlCandidates:
            candidateKey = stlCandidate.as_posix().lower()
            if candidateKey in seenCandidates:
                continue
            seenCandidates.add(candidateKey)
            uniqueStlCandidates.append(stlCandidate)

        for stlCandidatePath in uniqueStlCandidates:
            if not stlCandidatePath.exists():
                continue
            if SurgicalVision3D_PlannerLogic._buildSegmentationFileFromStl(stlCandidatePath, missingFilePath):
                logging.info(
                    "Repaired missing sample-scene segmentation file '%s' using STL '%s'.",
                    missingFilePath,
                    stlCandidatePath,
                )
                return True
        return False

    @staticmethod
    def _buildSegmentationFileFromStl(stlPath: Path, outputSegmentationPath: Path) -> bool:
        stlReader = vtk.vtkSTLReader()
        stlReader.SetFileName(str(stlPath))
        stlReader.Update()
        stlSurface = stlReader.GetOutput()
        if not stlSurface or stlSurface.GetNumberOfPoints() <= 0:
            return False

        closedSurface = vtk.vtkPolyData()
        closedSurface.DeepCopy(stlSurface)
        temporarySegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", stlPath.stem)
        try:
            temporarySegmentationNode.CreateDefaultDisplayNodes()
            SurgicalVision3D_PlannerLogic._preferClosedSurfaceSourceRepresentation(temporarySegmentationNode)
            SurgicalVision3D_PlannerLogic()._ensureSegmentationReferenceImageGeometry(temporarySegmentationNode)
            segmentID = temporarySegmentationNode.AddSegmentFromClosedSurfaceRepresentation(
                closedSurface,
                stlPath.stem,
                [0.2, 0.8, 1.0],
            )
            if not segmentID:
                return False
            outputSegmentationPath.parent.mkdir(parents=True, exist_ok=True)
            return bool(slicer.util.saveNode(temporarySegmentationNode, str(outputSegmentationPath)))
        finally:
            if temporarySegmentationNode and slicer.mrmlScene.IsNodePresent(temporarySegmentationNode):
                slicer.mrmlScene.RemoveNode(temporarySegmentationNode)

    @staticmethod
    def _normalizeFilesystemPath(pathValue: str | Path) -> str:
        try:
            normalizedPath = Path(pathValue).resolve()
        except Exception:
            normalizedPath = Path(pathValue)
        return normalizedPath.as_posix().lower()

    @staticmethod
    def _referenceProbeTemplateDisplayName(templatePath: Path) -> str:
        templateName = templatePath.stem.strip()
        return templateName if templateName else templatePath.name

    @staticmethod
    def _loadClosedSurfaceFromStl(stlPath: Path) -> vtk.vtkPolyData | None:
        stlReader = vtk.vtkSTLReader()
        stlReader.SetFileName(str(stlPath))
        stlReader.Update()
        outputSurface = stlReader.GetOutput()
        if not outputSurface or outputSurface.GetNumberOfPoints() <= 0:
            return None
        closedSurface = vtk.vtkPolyData()
        closedSurface.DeepCopy(outputSurface)
        return closedSurface

    def discoverReferenceProbeTemplateFiles(self) -> list[Path]:
        geometriesDirectory = self._resolveResourcePath(REFERENCE_PROBE_TEMPLATE_DIRECTORY_RELATIVE_PATH)
        if not geometriesDirectory.exists():
            logging.warning("Reference probe template directory was not found: %s", geometriesDirectory)
            return []
        return sorted(
            (
                candidatePath
                for candidatePath in geometriesDirectory.iterdir()
                if candidatePath.is_file() and candidatePath.suffix.lower() == ".stl"
            ),
            key=lambda candidatePath: candidatePath.name.lower(),
        )

    @staticmethod
    def isReferenceProbeTemplateSegmentation(segmentationNode: vtkMRMLSegmentationNode | None) -> bool:
        if not segmentationNode:
            return False
        return str(segmentationNode.GetAttribute(REFERENCE_PROBE_TEMPLATE_ATTRIBUTE) or "") == "1"

    def ensureReferenceProbeTemplatesLoaded(self) -> list[vtkMRMLSegmentationNode]:
        templatePaths = self.discoverReferenceProbeTemplateFiles()
        if len(templatePaths) == 0:
            return []

        catalogOffsetsByTemplatePath: dict[str, float] = {}
        for geometryEntry in self.loadGeometryCatalog():
            resolvedTemplatePath = self._resolveResourcePath(geometryEntry.templateRelativePath)
            normalizedCatalogTemplatePath = self._normalizeFilesystemPath(resolvedTemplatePath)
            if normalizedCatalogTemplatePath not in catalogOffsetsByTemplatePath:
                catalogOffsetsByTemplatePath[normalizedCatalogTemplatePath] = float(geometryEntry.axialPlacementOffsetMm)

        existingTemplatesByPath: dict[str, vtkMRMLSegmentationNode] = {}
        for segmentationNode in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
            sourcePath = segmentationNode.GetAttribute(REFERENCE_PROBE_TEMPLATE_SOURCE_PATH_ATTRIBUTE)
            if not sourcePath:
                continue
            normalizedSourcePath = self._normalizeFilesystemPath(sourcePath)
            if normalizedSourcePath not in existingTemplatesByPath:
                existingTemplatesByPath[normalizedSourcePath] = segmentationNode

        loadedTemplates: list[vtkMRMLSegmentationNode] = []
        for templatePath in templatePaths:
            normalizedTemplatePath = self._normalizeFilesystemPath(templatePath)
            existingTemplate = existingTemplatesByPath.get(normalizedTemplatePath)
            if existingTemplate and slicer.mrmlScene.IsNodePresent(existingTemplate):
                existingTemplate.SetHideFromEditors(True)
                existingTemplate.SetAttribute(
                    REFERENCE_PROBE_TEMPLATE_AXIAL_PLACEMENT_OFFSET_MM_ATTRIBUTE,
                    f"{float(catalogOffsetsByTemplatePath.get(normalizedTemplatePath, 0.0)):.6f}",
                )
                loadedTemplates.append(existingTemplate)
                continue

            closedSurface = self._loadClosedSurfaceFromStl(templatePath)
            if not closedSurface or closedSurface.GetNumberOfPoints() <= 0:
                logging.warning("Failed to load reference probe STL template: %s", templatePath)
                continue

            templateName = self._referenceProbeTemplateDisplayName(templatePath)
            templateNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", templateName)
            templateNode.CreateDefaultDisplayNodes()
            self._preferClosedSurfaceSourceRepresentation(templateNode)
            self._ensureSegmentationReferenceImageGeometry(templateNode)
            segmentID = templateNode.AddSegmentFromClosedSurfaceRepresentation(closedSurface, templateName, [0.2, 0.8, 1.0])
            if not segmentID:
                logging.warning("Failed to import reference probe template geometry into segmentation node: %s", templatePath)
                slicer.mrmlScene.RemoveNode(templateNode)
                continue

            templateNode.SetAttribute(REFERENCE_PROBE_TEMPLATE_ATTRIBUTE, "1")
            templateNode.SetAttribute(REFERENCE_PROBE_TEMPLATE_SOURCE_PATH_ATTRIBUTE, str(templatePath))
            templateNode.SetAttribute(
                REFERENCE_PROBE_TEMPLATE_AXIAL_PLACEMENT_OFFSET_MM_ATTRIBUTE,
                f"{float(catalogOffsetsByTemplatePath.get(normalizedTemplatePath, 0.0)):.6f}",
            )
            templateNode.SetHideFromEditors(True)
            if templateNode.GetDisplayNode():
                templateNode.GetDisplayNode().SetVisibility(False)
            loadedTemplates.append(templateNode)
            existingTemplatesByPath[normalizedTemplatePath] = templateNode

        return loadedTemplates

    def collectReproducibilityArtifacts(
        self,
        parameterNode: SurgicalVision3D_PlannerParameterNode,
        packageConfig: ReproducibilityPackageConfig,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        artifactPlans: list[dict[str, Any]] = []
        warnings: list[str] = []

        def addCopyArtifact(
            artifactKey: str,
            category: str,
            relativePath: str,
            sourcePath: Path,
            requiredByMode: bool = False,
        ) -> None:
            artifactPlans.append(
                {
                    "artifactKey": artifactKey,
                    "category": category,
                    "relativePath": relativePath.replace("\\", "/"),
                    "mode": "copy",
                    "sourcePath": str(sourcePath),
                    "requiredByMode": bool(requiredByMode),
                }
            )

        def addTableArtifact(
            artifactKey: str,
            category: str,
            relativePath: str,
            tableNode: vtkMRMLTableNode | None,
            requiredByMode: bool = False,
        ) -> None:
            artifactPlans.append(
                {
                    "artifactKey": artifactKey,
                    "category": category,
                    "relativePath": relativePath.replace("\\", "/"),
                    "mode": "table_csv",
                    "tableNode": tableNode,
                    "requiredByMode": bool(requiredByMode),
                }
            )

        def addJsonArtifact(
            artifactKey: str,
            category: str,
            relativePath: str,
            payload: dict[str, Any] | list[dict[str, Any]],
        ) -> None:
            artifactPlans.append(
                {
                    "artifactKey": artifactKey,
                    "category": category,
                    "relativePath": relativePath.replace("\\", "/"),
                    "mode": "json_data",
                    "payload": payload,
                    "requiredByMode": False,
                }
            )

        reproducibilityResourceRoot = self._resolveResourcePath("Resources/Reproducibility")
        addCopyArtifact(
            "schema_package_v1",
            "schemas",
            "schemas/reproducibility_package_schema_v1.json",
            reproducibilityResourceRoot / "reproducibility_package_schema_v1.json",
            requiredByMode=True,
        )
        addCopyArtifact(
            "schema_layout_v1",
            "schemas",
            "schemas/reproducibility_package_layout_v1.json",
            reproducibilityResourceRoot / "reproducibility_package_layout_v1.json",
            requiredByMode=True,
        )
        addCopyArtifact(
            "example_config_v1",
            "schemas",
            "schemas/example_reviewer_package_config_v1.json",
            reproducibilityResourceRoot / "example_reviewer_package_config_v1.json",
            requiredByMode=False,
        )
        addCopyArtifact(
            "package_readme_template",
            "reports",
            "README_package.md",
            reproducibilityResourceRoot / "README_package_template.md",
            requiredByMode=False,
        )

        if packageConfig.includeBenchmarkArtifacts:
            benchmarkResourceRoot = self._resolveResourcePath("Resources/Benchmarks")
            if benchmarkResourceRoot.exists():
                for benchmarkFile in sorted(path for path in benchmarkResourceRoot.rglob("*") if path.is_file()):
                    relativeBenchmarkPath = benchmarkFile.relative_to(benchmarkResourceRoot).as_posix()
                    addCopyArtifact(
                        f"benchmark_resource::{relativeBenchmarkPath}",
                        "benchmarks",
                        f"benchmarks/{relativeBenchmarkPath}",
                        benchmarkFile,
                        requiredByMode=False,
                    )
            else:
                warnings.append("Benchmark resources folder was not found; benchmark definitions were skipped.")

            addTableArtifact(
                "benchmark_case_summary",
                "validation",
                "validation/benchmark_case_summary.csv",
                self._findFirstTableNodeByName("SV3D Benchmark Case Summary"),
                requiredByMode=False,
            )
            addTableArtifact(
                "benchmark_validation_summary",
                "validation",
                "validation/benchmark_validation_summary.csv",
                self._findFirstTableNodeByName("SV3D Benchmark Validation Summary"),
                requiredByMode=False,
            )
            addTableArtifact(
                "benchmark_metric_delta_summary",
                "validation",
                "validation/benchmark_metric_delta_summary.csv",
                self._findFirstTableNodeByName("SV3D Benchmark Metric Delta Summary"),
                requiredByMode=False,
            )
            addTableArtifact(
                "benchmark_reproducibility_summary",
                "validation",
                "validation/benchmark_reproducibility_summary.csv",
                self._findFirstTableNodeByName("SV3D Benchmark Reproducibility Summary"),
                requiredByMode=False,
            )

        if packageConfig.includeValidationResults:
            addTableArtifact(
                "trajectory_verification_summary",
                "validation",
                "validation/trajectory_verification_summary.csv",
                self._findFirstTableNodeByName("SV3D Trajectory Verification Summary"),
                requiredByMode=False,
            )
            addTableArtifact(
                "plan_verification_summary",
                "validation",
                "validation/plan_verification_summary.csv",
                self._findFirstTableNodeByName("SV3D Plan Verification Summary"),
                requiredByMode=False,
            )

        if packageConfig.includeCohortStudyArtifacts:
            cohortResourceRoot = self._resolveResourcePath("Resources/Cohorts")
            if cohortResourceRoot.exists():
                for cohortFile in sorted(path for path in cohortResourceRoot.rglob("*") if path.is_file()):
                    relativeCohortPath = cohortFile.relative_to(cohortResourceRoot).as_posix()
                    addCopyArtifact(
                        f"cohort_resource::{relativeCohortPath}",
                        "cohorts",
                        f"cohorts/resources/{relativeCohortPath}",
                        cohortFile,
                        requiredByMode=False,
                    )
            addTableArtifact(
                "cohort_execution_summary",
                "cohorts",
                "cohorts/cohort_execution_summary.csv",
                parameterNode.cohortExecutionSummaryTable,
                requiredByMode=False,
            )
            addTableArtifact(
                "cohort_case_summary",
                "cohorts",
                "cohorts/cohort_case_summary.csv",
                parameterNode.cohortCaseSummaryTable,
                requiredByMode=False,
            )
            addTableArtifact(
                "cohort_aggregate_metrics",
                "cohorts",
                "cohorts/cohort_aggregate_metrics.csv",
                parameterNode.cohortAggregateMetricsTable,
                requiredByMode=False,
            )
            addTableArtifact(
                "cohort_comparison_summary",
                "cohorts",
                "cohorts/cohort_comparison_summary.csv",
                parameterNode.cohortComparisonSummaryTable,
                requiredByMode=False,
            )
            resolvedCohortDefinitionPath = self._resolveModuleRelativePath(str(parameterNode.cohortStudyDefinitionPath or ""))
            if str(parameterNode.cohortStudyDefinitionPath or "").strip():
                addCopyArtifact(
                    "selected_cohort_definition",
                    "cohorts",
                    "cohorts/selected_cohort_definition.json",
                    resolvedCohortDefinitionPath,
                    requiredByMode=False,
                )

        if packageConfig.includeStudyAnalytics:
            addTableArtifact(
                "study_summary",
                "study_analytics",
                "study_analytics/study_summary.csv",
                self._findFirstTableNodeByName("SV3D Study Summary"),
                requiredByMode=False,
            )
            addTableArtifact(
                "study_aggregate_table",
                "study_analytics",
                "study_analytics/study_aggregate_table.csv",
                self._findFirstTableNodeByName("SV3D Study Aggregate Table"),
                requiredByMode=False,
            )
            addTableArtifact(
                "study_group_comparison",
                "study_analytics",
                "study_analytics/study_group_comparison.csv",
                self._findFirstTableNodeByName("SV3D Study Group Comparison"),
                requiredByMode=False,
            )
            addTableArtifact(
                "study_publication_table_pack",
                "study_analytics",
                "study_analytics/study_publication_table_pack.csv",
                self._findFirstTableNodeByName("SV3D Study Publication Table Pack"),
                requiredByMode=False,
            )

        if packageConfig.includeReports:
            addTableArtifact(
                "report_summary",
                "reports",
                "reports/report_summary.csv",
                self._findFirstTableNodeByName("SV3D Report Summary"),
                requiredByMode=False,
            )
            addTableArtifact(
                "report_sections",
                "reports",
                "reports/report_sections.csv",
                self._findFirstTableNodeByName("SV3D Report Sections"),
                requiredByMode=False,
            )
            addTableArtifact(
                "report_metrics",
                "reports",
                "reports/report_metrics.csv",
                self._findFirstTableNodeByName("SV3D Report Metrics"),
                requiredByMode=False,
            )

            lastExportDirectory = str(parameterNode.lastExportDirectory or "").strip()
            if lastExportDirectory and int(parameterNode.lastExportSequence) > 0:
                latestExportBundlePath = self.buildDeterministicBundlePath(
                    lastExportDirectory,
                    str(parameterNode.exportBaseName or "SV3D_Export"),
                    int(parameterNode.lastExportSequence),
                )
                if latestExportBundlePath.exists() and latestExportBundlePath.is_dir():
                    for bundleFile in sorted(path for path in latestExportBundlePath.rglob("*") if path.is_file()):
                        relativeBundlePath = bundleFile.relative_to(latestExportBundlePath).as_posix()
                        addCopyArtifact(
                            f"latest_export_bundle::{relativeBundlePath}",
                            "exports",
                            f"exports/latest_export_bundle/{relativeBundlePath}",
                            bundleFile,
                            requiredByMode=False,
                        )

        if packageConfig.includeScenarioRegistry:
            scenarioRegistryTable = self._findFirstTableNodeByName("SV3D Scenario Registry")
            recommendationTable = self._findFirstTableNodeByName("SV3D Feasible Candidate Recommendation")
            addTableArtifact(
                "scenario_registry_csv",
                "exports",
                "exports/provenance/scenario_registry.csv",
                scenarioRegistryTable,
                requiredByMode=False,
            )
            addTableArtifact(
                "recommendation_summary_csv",
                "exports",
                "exports/provenance/recommendation_summary.csv",
                recommendationTable,
                requiredByMode=False,
            )

        if packageConfig.includeCanonicalJson:
            currentPlanSummary, _ = self.collectCurrentPlanExportData(
                parameterNode,
                PlanExportConfig(
                    includeWorkingPlan=True,
                    includeSelectedScenario=False,
                    includeScenarioComparison=False,
                    includeRecommendationOutputs=False,
                    includeTrajectoryTables=False,
                    includeSafetyTables=False,
                    includeCoverageTables=False,
                    includeFeasibilityTables=False,
                    includeCoordinationTables=False,
                ),
            )
            addJsonArtifact(
                "canonical_current_plan_summary",
                "canonical_json",
                "canonical_json/current_plan_summary.json",
                currentPlanSummary,
            )
            scenarioRegistryTable = self._findFirstTableNodeByName("SV3D Scenario Registry")
            if scenarioRegistryTable:
                addJsonArtifact(
                    "canonical_scenario_registry",
                    "canonical_json",
                    "canonical_json/scenario_registry.json",
                    self._tableNodeToDictionaries(scenarioRegistryTable),
                )
            recommendationTable = self._findFirstTableNodeByName("SV3D Feasible Candidate Recommendation")
            if recommendationTable:
                addJsonArtifact(
                    "canonical_recommendation_summary",
                    "canonical_json",
                    "canonical_json/recommendation_summary.json",
                    self._tableNodeToDictionaries(recommendationTable),
                )

        return artifactPlans, warnings

    def copyOrRegenerateArtifactSet(
        self,
        packagePath: Path,
        artifactPlans: Sequence[dict[str, Any]],
    ) -> tuple[list[ReproducibilityArtifactEntry], list[str]]:
        artifactEntries: list[ReproducibilityArtifactEntry] = []
        warnings: list[str] = []

        for artifactPlan in sorted(
            artifactPlans,
            key=lambda plan: (str(plan.get("relativePath", "")), str(plan.get("artifactKey", ""))),
        ):
            artifactKey = str(artifactPlan.get("artifactKey", ""))
            category = str(artifactPlan.get("category", "uncategorized"))
            relativePath = str(artifactPlan.get("relativePath", "")).replace("\\", "/")
            mode = str(artifactPlan.get("mode", ""))
            requiredByMode = bool(artifactPlan.get("requiredByMode", False))
            destinationPath = packagePath / relativePath

            entry = ReproducibilityArtifactEntry(
                artifactKey=artifactKey,
                category=category,
                relativePath=relativePath,
                status="Missing",
            )

            try:
                if mode == "copy":
                    sourcePath = Path(str(artifactPlan.get("sourcePath", "")))
                    entry.sourcePath = str(sourcePath)
                    if sourcePath.exists() and sourcePath.is_file():
                        destinationPath.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(sourcePath, destinationPath)
                        integrity = self.computeArtifactIntegritySummary(destinationPath, includeHash=True)
                        entry.status = "Copied"
                        entry.sizeBytes = int(integrity["sizeBytes"])
                        entry.sha256 = str(integrity["sha256"])
                    else:
                        warningText = f"Artifact source is missing: {sourcePath}"
                        entry.warning = warningText
                        warnings.append(warningText)
                        if requiredByMode:
                            raise ValueError(warningText)
                elif mode == "table_csv":
                    tableNode = artifactPlan.get("tableNode")
                    if tableNode and slicer.mrmlScene.IsNodePresent(tableNode):
                        self.exportTableNodeToCsv(tableNode, destinationPath)
                        integrity = self.computeArtifactIntegritySummary(destinationPath, includeHash=True)
                        entry.status = "GeneratedCSV"
                        entry.sourcePath = str(tableNode.GetID())
                        entry.sizeBytes = int(integrity["sizeBytes"])
                        entry.sha256 = str(integrity["sha256"])
                    else:
                        warningText = f"Source table is not available for artifact '{artifactKey}'."
                        entry.warning = warningText
                        warnings.append(warningText)
                        if requiredByMode:
                            raise ValueError(warningText)
                elif mode == "json_data":
                    payload = artifactPlan.get("payload")
                    self.exportStructuredSummaryToJson(destinationPath, payload if payload is not None else {})
                    integrity = self.computeArtifactIntegritySummary(destinationPath, includeHash=True)
                    entry.status = "GeneratedJSON"
                    entry.sourcePath = "InSceneSummary"
                    entry.sizeBytes = int(integrity["sizeBytes"])
                    entry.sha256 = str(integrity["sha256"])
                else:
                    warningText = f"Unsupported artifact mode '{mode}' for artifact '{artifactKey}'."
                    entry.warning = warningText
                    warnings.append(warningText)
                    if requiredByMode:
                        raise ValueError(warningText)
            except Exception as exc:
                if requiredByMode:
                    raise
                warningText = f"{artifactKey}: {exc}"
                entry.warning = warningText
                warnings.append(warningText)

            artifactEntries.append(entry)

        return artifactEntries, warnings

    def buildReproducibilityManifest(
        self,
        packageConfig: ReproducibilityPackageConfig,
        packageSequence: int,
        artifactEntries: Sequence[ReproducibilityArtifactEntry],
        warnings: Sequence[str],
    ) -> ReproducibilityManifest:
        scenarioRegistryRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Scenario Registry"))
        benchmarkRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Benchmark Case Summary"))
        cohortExecutionRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName(COHORT_EXECUTION_SUMMARY_TABLE_NODE_NAME))
        reportRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Report Summary"))

        def collectValues(rows: Sequence[dict[str, Any]], keyCandidates: Sequence[str]) -> list[str]:
            values: list[str] = []
            seen: set[str] = set()
            for row in rows:
                for keyCandidate in keyCandidates:
                    rawValue = str(row.get(keyCandidate, "")).strip()
                    if rawValue and rawValue not in seen:
                        seen.add(rawValue)
                        values.append(rawValue)
                        break
            return sorted(values)

        studyIds: list[str] = []
        for row in cohortExecutionRows:
            if "Field" in row and "Value" in row and str(row.get("Field", "")).strip() == "StudyID":
                value = str(row.get("Value", "")).strip()
                if value:
                    studyIds.append(value)
        studyIds = sorted(set(studyIds))

        manifestArtifacts = [asdict(entry) for entry in sorted(artifactEntries, key=lambda item: item.relativePath)]
        manifestWarnings = sorted(set(str(warning) for warning in warnings if str(warning).strip()))
        return ReproducibilityManifest(
            packageId=f"SV3D-Repro-{int(packageSequence):04d}",
            packageTimestampISO=datetime.now().isoformat(timespec="seconds"),
            packageSequence=int(packageSequence),
            packageMode=str(packageConfig.packageMode),
            packageBaseName=str(packageConfig.packageBaseName),
            createdByModule="SurgicalVision3D_Planner",
            schemaVersions={
                "reproducibilityPackageSchema": "v1",
                "reproducibilityLayoutSchema": "v1",
                "phaseMarker": "Phase12B",
            },
            includedArtifacts=manifestArtifacts,
            benchmarkCaseIds=collectValues(benchmarkRows, ["CaseID", "caseId"]),
            studyIds=studyIds,
            scenarioIds=collectValues(scenarioRegistryRows, ["ScenarioID", "scenarioId"]),
            reportIds=collectValues(reportRows, ["ReportID", "reportId"]),
            warnings=manifestWarnings,
            notes="Deterministic frozen reproducibility package; missing optional artifacts are listed in warnings.",
        )

    def assembleReproducibilityPackage(
        self,
        parameterNode: SurgicalVision3D_PlannerParameterNode,
        packageConfig: ReproducibilityPackageConfig,
    ) -> dict[str, Any]:
        if not str(packageConfig.packageBaseName).strip():
            raise ValueError("Reproducibility package base name is required.")

        packageDirectory = str(packageConfig.outputDirectory).strip() or str(
            Path(slicer.app.temporaryPath) / "SurgicalVision3D_PlannerReproducibilityPackages"
        )
        packageRoot = Path(packageDirectory)
        packageRoot.mkdir(parents=True, exist_ok=True)

        packageSequence = max(1, int(packageConfig.lastPackageSequence) + 1)
        packagePath = self.buildDeterministicReproPackagePath(packageDirectory, packageConfig.packageBaseName, packageSequence)
        while packagePath.exists():
            packageSequence += 1
            packagePath = self.buildDeterministicReproPackagePath(packageDirectory, packageConfig.packageBaseName, packageSequence)
        packagePath.mkdir(parents=True, exist_ok=False)

        # Keep the layout explicit and stable even when specific sections are empty.
        for subDirectory in (
            "schemas",
            "benchmarks",
            "validation",
            "cohorts",
            "study_analytics",
            "reports",
            "exports",
            "canonical_json",
        ):
            (packagePath / subDirectory).mkdir(parents=True, exist_ok=True)

        artifactPlans, collectionWarnings = self.collectReproducibilityArtifacts(parameterNode, packageConfig)
        artifactEntries, assemblyWarnings = self.copyOrRegenerateArtifactSet(packagePath, artifactPlans)
        combinedWarnings = sorted(set(collectionWarnings + assemblyWarnings))

        manifest = self.buildReproducibilityManifest(
            packageConfig=packageConfig,
            packageSequence=packageSequence,
            artifactEntries=artifactEntries,
            warnings=combinedWarnings,
        )
        manifestPath = packagePath / "manifest.json"
        self.exportStructuredSummaryToJson(manifestPath, asdict(manifest))
        manifestIntegrity = self.computeArtifactIntegritySummary(manifestPath, includeHash=True)
        artifactEntries.append(
            ReproducibilityArtifactEntry(
                artifactKey="manifest",
                category="schemas",
                relativePath="manifest.json",
                status="GeneratedJSON",
                sourcePath="InSceneSummary",
                sizeBytes=int(manifestIntegrity["sizeBytes"]),
                sha256=str(manifestIntegrity["sha256"]),
                warning="",
            )
        )

        # Rewrite manifest to include manifest.json in the artifact list deterministically.
        manifest.includedArtifacts = [asdict(entry) for entry in sorted(artifactEntries, key=lambda item: item.relativePath)]
        self.exportStructuredSummaryToJson(manifestPath, asdict(manifest))

        statusText = "SuccessWithWarnings" if len(combinedWarnings) > 0 else "Success"
        return {
            "manifest": manifest,
            "packagePath": str(packagePath),
            "packageDirectory": str(packageRoot),
            "packageSequence": int(packageSequence),
            "artifactEntries": sorted(artifactEntries, key=lambda item: item.relativePath),
            "artifactCount": int(len(artifactEntries)),
            "warningCount": int(len(combinedWarnings)),
            "status": statusText,
        }

    @staticmethod
    def _coerceBoolean(value: Any, defaultValue: bool = False) -> bool:
        if value is None:
            return defaultValue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        valueText = str(value).strip().lower()
        if valueText in ("1", "true", "yes", "y", "pass"):
            return True
        if valueText in ("0", "false", "no", "n", "fail"):
            return False
        return defaultValue

    @staticmethod
    def _resolveModuleRelativePath(inputPath: str) -> Path:
        path = Path(inputPath).expanduser()
        if path.is_absolute():
            return path
        return (Path(__file__).resolve().parent / path).resolve()

    def loadCohortStudyDefinition(self, studyDefinitionPath: str) -> CohortStudyDefinition:
        if not studyDefinitionPath.strip():
            raise ValueError("Cohort study definition path is required.")

        resolvedPath = self._resolveModuleRelativePath(studyDefinitionPath)
        if not resolvedPath.exists():
            raise ValueError(f"Cohort study definition was not found: {resolvedPath}")

        with resolvedPath.open("r", encoding="utf-8") as inputFile:
            rawDefinition = json.load(inputFile)

        studyID = str(rawDefinition.get("studyId", "")).strip()
        if not studyID:
            raise ValueError("Cohort study definition is missing required field 'studyId'.")

        rawCases = rawDefinition.get("cases", [])
        if not isinstance(rawCases, list) or len(rawCases) == 0:
            raise ValueError(f"Cohort study '{studyID}' does not define any case members.")

        caseMembers: list[CohortCaseMember] = []
        for caseIndex, rawCase in enumerate(rawCases):
            if not isinstance(rawCase, dict):
                raise ValueError(f"Cohort case at index {caseIndex} must be a JSON object.")
            caseID = str(rawCase.get("caseId", f"Case{caseIndex + 1:03d}")).strip()
            caseMembers.append(
                CohortCaseMember(
                    caseId=caseID,
                    displayName=str(rawCase.get("displayName", caseID)),
                    inputReference=str(rawCase.get("inputReference", "ScenarioID")),
                    scenarioId=str(rawCase.get("scenarioId", "")),
                    presetId=str(rawCase.get("presetId", "")),
                    targetSegmentId=str(rawCase.get("targetSegmentId", "")),
                    notes=str(rawCase.get("notes", "")),
                )
            )

        return CohortStudyDefinition(
            studyId=studyID,
            displayName=str(rawDefinition.get("displayName", studyID)),
            description=str(rawDefinition.get("description", "")),
            cases=caseMembers,
        )

    @staticmethod
    def _findRowByColumnValue(
        rows: Sequence[dict[str, Any]],
        columnCandidates: Sequence[str],
        expectedValue: str,
    ) -> dict[str, Any] | None:
        expected = str(expectedValue).strip()
        for row in rows:
            for columnName in columnCandidates:
                if str(row.get(columnName, "")).strip() == expected:
                    return row
        return None

    @staticmethod
    def _firstNumericValue(rowCandidates: Sequence[dict[str, Any]], columnCandidates: Sequence[str]) -> float | None:
        for row in rowCandidates:
            if not row:
                continue
            for columnName in columnCandidates:
                rawValue = row.get(columnName)
                if rawValue is None or str(rawValue).strip() == "":
                    continue
                try:
                    numericValue = float(rawValue)
                except Exception:
                    continue
                if math.isfinite(numericValue):
                    return numericValue
        return None

    @staticmethod
    def _firstStringValue(rowCandidates: Sequence[dict[str, Any]], columnCandidates: Sequence[str]) -> str:
        for row in rowCandidates:
            if not row:
                continue
            for columnName in columnCandidates:
                valueText = str(row.get(columnName, "")).strip()
                if valueText:
                    return valueText
        return ""

    def _collectCurrentPlanMetricsForCohort(
        self,
        parameterNode: SurgicalVision3D_PlannerParameterNode,
        executionConfig: CohortExecutionConfig,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        planSummaryRows = self._tableNodeToDictionaries(parameterNode.planSummaryTable)
        coverageRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Coverage Summary"))
        safetyRows = self._tableNodeToDictionaries(parameterNode.structureSafetySummaryTable)
        coordinationRows = self._tableNodeToDictionaries(parameterNode.probeCoordinationSummaryTable)
        verificationRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Plan Verification Summary"))

        planRow = planSummaryRows[0] if len(planSummaryRows) > 0 else {}
        coverageRow = coverageRows[0] if len(coverageRows) > 0 else {}
        coordinationRow = coordinationRows[0] if len(coordinationRows) > 0 else {}
        verificationRow = verificationRows[0] if len(verificationRows) > 0 else {}

        trajectoryCount = self._firstNumericValue([planRow], ["Trajectory Count", "TrajectoryCount"])
        if trajectoryCount is not None:
            metrics["TrajectoryCount"] = int(round(trajectoryCount))

        minMargin = self._firstNumericValue([planRow], ["Minimum Signed Margin (mm)", "MinSignedMarginMm"])
        medianMargin = self._firstNumericValue([planRow], ["Median Signed Margin (mm)", "MedianSignedMarginMm"])
        coveragePercent = self._firstNumericValue([coverageRow], ["CoveragePercent", "Coverage Percent", "Coverage (%)"])
        coordinationGatePass = self._coerceBoolean(
            self._firstStringValue([coordinationRow], ["Coordination Gate Pass", "CoordinationGatePass"]),
            defaultValue=True,
        )
        verificationMeanTargetDeviation = self._firstNumericValue(
            [verificationRow],
            ["MeanTargetDeviationMm", "Mean Target Deviation (mm)"],
        )

        worstStructureDistance = None
        if len(safetyRows) > 0:
            distanceValues = [
                self._firstNumericValue([row], ["Minimum Distance (mm)", "MinDistanceMm"])
                for row in safetyRows
            ]
            finiteDistances = [value for value in distanceValues if value is not None and math.isfinite(value)]
            if len(finiteDistances) > 0:
                worstStructureDistance = min(finiteDistances)

        if executionConfig.includeMarginMetrics and minMargin is not None:
            metrics["MinSignedMarginMm"] = float(minMargin)
        if executionConfig.includeMarginMetrics and medianMargin is not None:
            metrics["MedianSignedMarginMm"] = float(medianMargin)
        if executionConfig.includeCoverageMetrics and coveragePercent is not None:
            metrics["CoveragePercent"] = float(coveragePercent)
        if executionConfig.includeSafetyMetrics and worstStructureDistance is not None:
            metrics["WorstStructureMinDistanceMm"] = float(worstStructureDistance)
        if executionConfig.includeCoordinationMetrics:
            metrics["CoordinationGatePass"] = bool(coordinationGatePass)
        if executionConfig.includeVerificationMetrics and verificationMeanTargetDeviation is not None:
            metrics["MeanTargetDeviationMm"] = float(verificationMeanTargetDeviation)
        return metrics

    def collectCohortCaseMetrics(
        self,
        parameterNode: SurgicalVision3D_PlannerParameterNode,
        caseMember: CohortCaseMember,
        executionConfig: CohortExecutionConfig,
    ) -> dict[str, Any]:
        executionModeText = str(executionConfig.executionMode or "").strip().lower()
        sourceMode = str(caseMember.inputReference or "ScenarioID").strip().lower()
        if executionModeText == "currentworkingplan":
            sourceMode = "currentworkingplan"
        if sourceMode in ("currentworkingplan", "currentplan"):
            return self._collectCurrentPlanMetricsForCohort(parameterNode, executionConfig)

        scenarioID = str(caseMember.scenarioId).strip()
        if not scenarioID:
            raise ValueError(f"Cohort case '{caseMember.caseId}' requires a non-empty scenarioId.")

        scenarioRegistryRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Scenario Registry"))
        scenarioRow = self._findRowByColumnValue(scenarioRegistryRows, ["ScenarioID"], scenarioID)
        if scenarioRow is None:
            raise ValueError(f"Cohort case '{caseMember.caseId}' references unknown scenario ID '{scenarioID}'.")

        comparisonRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Scenario Comparison"))
        comparisonRow = self._findRowByColumnValue(comparisonRows, ["ScenarioID"], scenarioID) or {}

        feasibilityRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Candidate Feasibility Summary"))
        feasibilityRow = self._findRowByColumnValue(feasibilityRows, ["ScenarioID"], scenarioID) or {}

        verificationRows = self._tableNodeToDictionaries(self._findFirstTableNodeByName("SV3D Trajectory Verification Summary"))
        verificationRow = verificationRows[0] if len(verificationRows) > 0 else {}

        metrics: dict[str, Any] = {
            "ScenarioID": scenarioID,
        }
        metrics["PresetID"] = self._firstStringValue(
            [comparisonRow, scenarioRow],
            ["ApplicatorPresetID", "PresetID", "Preset ID"],
        )
        if executionConfig.includeRecommendationMetrics:
            metrics["RecommendationTag"] = self._firstStringValue(
                [feasibilityRow, comparisonRow],
                ["RecommendationTag", "Tag"],
            )
        if executionConfig.includeFeasibilityMetrics:
            metrics["IsFeasible"] = self._coerceBoolean(
                self._firstStringValue([feasibilityRow], ["IsFeasible", "Is Feasible"]),
                defaultValue=False,
            )
        if executionConfig.includeCoordinationMetrics:
            metrics["CoordinationGatePass"] = self._coerceBoolean(
                self._firstStringValue(
                    [feasibilityRow, comparisonRow],
                    ["CoordinationGatePass", "Coordination Gate Pass"],
                ),
                defaultValue=True,
            )

        if executionConfig.includeMarginMetrics:
            minMargin = self._firstNumericValue([comparisonRow], ["MinSignedMarginMm", "Minimum Signed Margin (mm)"])
            medianMargin = self._firstNumericValue([comparisonRow], ["MedianSignedMarginMm", "Median Signed Margin (mm)"])
            trajectoryCount = self._firstNumericValue([comparisonRow], ["TrajectoryCount", "Trajectory Count"])
            if minMargin is not None:
                metrics["MinSignedMarginMm"] = float(minMargin)
            if medianMargin is not None:
                metrics["MedianSignedMarginMm"] = float(medianMargin)
            if trajectoryCount is not None:
                metrics["TrajectoryCount"] = int(round(trajectoryCount))

        if executionConfig.includeCoverageMetrics:
            coveragePercent = self._firstNumericValue([comparisonRow], ["CoveragePercent", "Coverage Percent"])
            if coveragePercent is not None:
                metrics["CoveragePercent"] = float(coveragePercent)

        if executionConfig.includeSafetyMetrics:
            worstStructure = self._firstNumericValue(
                [comparisonRow],
                ["WorstStructureMinDistanceMm", "Worst Structure Min Distance (mm)"],
            )
            if worstStructure is not None:
                metrics["WorstStructureMinDistanceMm"] = float(worstStructure)

        if executionConfig.includeRecommendationMetrics:
            compositeScore = self._firstNumericValue([comparisonRow], ["CompositeScore"])
            if compositeScore is not None:
                metrics["CompositeScore"] = float(compositeScore)

        if executionConfig.includeVerificationMetrics:
            meanTargetDeviation = self._firstNumericValue(
                [verificationRow],
                ["MeanTargetDeviationMm", "Mean Target Deviation (mm)"],
            )
            if meanTargetDeviation is not None:
                metrics["MeanTargetDeviationMm"] = float(meanTargetDeviation)
        return metrics

    def runCaseMemberEvaluation(
        self,
        parameterNode: SurgicalVision3D_PlannerParameterNode,
        caseMember: CohortCaseMember,
        executionConfig: CohortExecutionConfig,
    ) -> CohortCaseResult:
        try:
            metricValues = self.collectCohortCaseMetrics(parameterNode, caseMember, executionConfig)
            statusMessage = "Completed"
            if len(metricValues) <= 2:
                statusMessage = "Completed with limited metrics"
            return CohortCaseResult(
                caseId=caseMember.caseId,
                displayName=caseMember.displayName,
                inputReference=caseMember.inputReference,
                scenarioId=caseMember.scenarioId,
                executionStatus="Success",
                statusMessage=statusMessage,
                presetId=caseMember.presetId,
                targetSegmentId=caseMember.targetSegmentId,
                metricValues=metricValues,
            )
        except Exception as exc:
            return CohortCaseResult(
                caseId=caseMember.caseId,
                displayName=caseMember.displayName,
                inputReference=caseMember.inputReference,
                scenarioId=caseMember.scenarioId,
                executionStatus="Failed",
                statusMessage=str(exc),
                presetId=caseMember.presetId,
                targetSegmentId=caseMember.targetSegmentId,
                metricValues={},
            )

    @staticmethod
    def aggregateCohortMetrics(caseResults: Sequence[CohortCaseResult]) -> dict[str, float | int]:
        successfulResults = [result for result in caseResults if result.executionStatus == "Success"]

        def numericValues(metricName: str) -> list[float]:
            values: list[float] = []
            for caseResult in successfulResults:
                rawValue = caseResult.metricValues.get(metricName)
                if rawValue is None:
                    continue
                try:
                    numericValue = float(rawValue)
                except Exception:
                    continue
                if math.isfinite(numericValue):
                    values.append(numericValue)
            return values

        aggregated: dict[str, float | int] = {
            "CaseCount": int(len(caseResults)),
            "SuccessCount": int(len(successfulResults)),
            "FailureCount": int(len(caseResults) - len(successfulResults)),
        }
        for metricName in (
            "CoveragePercent",
            "MinSignedMarginMm",
            "MedianSignedMarginMm",
            "WorstStructureMinDistanceMm",
            "CompositeScore",
            "TrajectoryCount",
            "MeanTargetDeviationMm",
        ):
            values = numericValues(metricName)
            if len(values) == 0:
                continue
            aggregated[f"Mean{metricName}"] = float(np.mean(values))
            aggregated[f"Median{metricName}"] = float(np.median(values))
            aggregated[f"Min{metricName}"] = float(np.min(values))
            aggregated[f"Max{metricName}"] = float(np.max(values))

        feasibleCount = int(
            sum(
                1
                for caseResult in successfulResults
                if SurgicalVision3D_PlannerLogic._coerceBoolean(caseResult.metricValues.get("IsFeasible"), defaultValue=False)
            )
        )
        recommendationTagCount = int(
            sum(1 for caseResult in successfulResults if str(caseResult.metricValues.get("RecommendationTag", "")).strip())
        )
        aggregated["FeasibleCaseCount"] = feasibleCount
        aggregated["RecommendationTaggedCaseCount"] = recommendationTagCount
        return aggregated

    @staticmethod
    def groupCohortResultsByPreset(caseResults: Sequence[CohortCaseResult]) -> dict[str, list[CohortCaseResult]]:
        groupedResults: dict[str, list[CohortCaseResult]] = {}
        for caseResult in caseResults:
            groupKey = str(caseResult.metricValues.get("PresetID", "")).strip() or "UnspecifiedPreset"
            groupedResults.setdefault(groupKey, []).append(caseResult)
        return groupedResults

    def computeCohortComparisonSummary(self, caseResults: Sequence[CohortCaseResult]) -> list[dict[str, float | int | str]]:
        groupedByPreset = self.groupCohortResultsByPreset(caseResults)
        summaryRows: list[dict[str, float | int | str]] = []
        for presetID in sorted(groupedByPreset.keys()):
            groupResults = groupedByPreset[presetID]
            successfulGroup = [result for result in groupResults if result.executionStatus == "Success"]
            coverageValues: list[float] = []
            compositeValues: list[float] = []
            for result in successfulGroup:
                coverageRaw = result.metricValues.get("CoveragePercent")
                if coverageRaw is not None:
                    try:
                        coverageValue = float(coverageRaw)
                        if math.isfinite(coverageValue):
                            coverageValues.append(coverageValue)
                    except Exception:
                        pass
                compositeRaw = result.metricValues.get("CompositeScore")
                if compositeRaw is not None:
                    try:
                        compositeValue = float(compositeRaw)
                        if math.isfinite(compositeValue):
                            compositeValues.append(compositeValue)
                    except Exception:
                        pass
            summaryRows.append(
                {
                    "PresetID": presetID,
                    "CaseCount": int(len(groupResults)),
                    "SuccessCount": int(len(successfulGroup)),
                    "MeanCoveragePercent": float(np.mean(coverageValues)) if len(coverageValues) > 0 else float("nan"),
                    "MeanCompositeScore": float(np.mean(compositeValues)) if len(compositeValues) > 0 else float("nan"),
                }
            )
        return summaryRows

    def runCohortStudy(
        self,
        parameterNode: SurgicalVision3D_PlannerParameterNode,
        executionConfig: CohortExecutionConfig,
    ) -> dict[str, Any]:
        studyDefinition = self.loadCohortStudyDefinition(executionConfig.studyDefinitionPath)
        caseMembers = list(studyDefinition.cases)
        if executionConfig.maxCases > 0:
            caseMembers = caseMembers[: int(executionConfig.maxCases)]

        caseResults = [
            self.runCaseMemberEvaluation(parameterNode, caseMember, executionConfig)
            for caseMember in caseMembers
        ]
        successCount = int(sum(1 for result in caseResults if result.executionStatus == "Success"))
        executionSummary = {
            "StudyID": studyDefinition.studyId,
            "StudyDisplayName": studyDefinition.displayName,
            "ExecutionMode": executionConfig.executionMode,
            "CaseCount": int(len(caseResults)),
            "SuccessCount": successCount,
            "FailureCount": int(len(caseResults) - successCount),
            "SuccessRatePercent": float(100.0 * successCount / len(caseResults)) if len(caseResults) > 0 else 0.0,
            "StudyDescription": studyDefinition.description,
        }
        aggregateMetrics = self.aggregateCohortMetrics(caseResults)
        comparisonRows = self.computeCohortComparisonSummary(caseResults)
        return {
            "studyDefinition": studyDefinition,
            "caseResults": caseResults,
            "executionSummary": executionSummary,
            "aggregateMetrics": aggregateMetrics,
            "comparisonRows": comparisonRows,
        }

    @staticmethod
    def computeTrajectoryMetrics(trajectories: Sequence[ProbeTrajectory]) -> list[dict[str, float | int | str]]:
        metrics: list[dict[str, float | int | str]] = []
        for trajectory in trajectories:
            role = str(trajectory.role or f"Trajectory {int(trajectory.trajectoryIndex) + 1:02d}")
            source = "derived" if bool(trajectory.derivedFromMaster) else ("master" if role.lower() == "master" else "manual")
            metrics.append(
                {
                    "TrajectoryIndex": int(trajectory.trajectoryIndex + 1),
                    "Role": role,
                    "Source": source,
                    "EntryR": float(trajectory.entryPointRAS[0]),
                    "EntryA": float(trajectory.entryPointRAS[1]),
                    "EntryS": float(trajectory.entryPointRAS[2]),
                    "TargetR": float(trajectory.targetPointRAS[0]),
                    "TargetA": float(trajectory.targetPointRAS[1]),
                    "TargetS": float(trajectory.targetPointRAS[2]),
                    "DirR": float(trajectory.directionVector[0]),
                    "DirA": float(trajectory.directionVector[1]),
                    "DirS": float(trajectory.directionVector[2]),
                    "LengthMm": float(trajectory.lengthMm),
                    "AngleDeg": float(trajectory.angleDeg) if trajectory.angleDeg is not None else float("nan"),
                    "RadialOffsetMm": float(trajectory.radialOffsetMm),
                }
            )
        return metrics

    @staticmethod
    def computeSignedMarginSummary(
        signedMarginValues: Sequence[float],
        trajectoryCount: int,
        tumorSegmentID: str,
        tumorSegmentName: str,
    ) -> dict[str, float | int | str]:
        if len(signedMarginValues) == 0:
            raise ValueError("Signed margin summary cannot be computed because no signed margin values are available.")

        signedValues = np.asarray(signedMarginValues, dtype=float)
        signedValues = signedValues[np.isfinite(signedValues)]
        if signedValues.size == 0:
            raise ValueError("Signed margin summary cannot be computed because signed margin values are invalid.")

        return {
            "TrajectoryCount": int(trajectoryCount),
            "TumorSegmentID": tumorSegmentID,
            "TumorSegmentName": tumorSegmentName,
            "MinSignedMarginMm": float(np.min(signedValues)),
            "MeanSignedMarginMm": float(np.mean(signedValues)),
            "MedianSignedMarginMm": float(np.median(signedValues)),
            "P20SignedMarginMm": float(np.quantile(signedValues, 0.20)),
            "P80SignedMarginMm": float(np.quantile(signedValues, 0.80)),
        }

    @staticmethod
    def computeMarginThresholdSummary(signedMarginValues: Sequence[float]) -> list[dict[str, float | int | str]]:
        if len(signedMarginValues) == 0:
            raise ValueError("Margin threshold summary cannot be computed because no signed margin values are available.")

        signedValues = np.asarray(signedMarginValues, dtype=float)
        signedValues = signedValues[np.isfinite(signedValues)]
        if signedValues.size == 0:
            raise ValueError("Margin threshold summary cannot be computed because signed margin values are invalid.")

        totalValueCount = int(signedValues.size)
        thresholdBuckets = [
            ("< 0 mm", int(np.count_nonzero(signedValues < 0.0))),
            ("< 2 mm", int(np.count_nonzero(signedValues < 2.0))),
            ("< 5 mm", int(np.count_nonzero(signedValues < 5.0))),
            (">= 5 mm", int(np.count_nonzero(signedValues >= 5.0))),
        ]

        summaryRows: list[dict[str, float | int | str]] = []
        for bucketLabel, bucketCount in thresholdBuckets:
            percentage = (100.0 * bucketCount / totalValueCount) if totalValueCount else 0.0
            summaryRows.append(
                {
                    "Bucket": bucketLabel,
                    "Count": int(bucketCount),
                    "Percent": float(percentage),
                }
            )
        return summaryRows

    @staticmethod
    def computeDistanceSummary(distanceValues: Sequence[float]) -> dict[str, float]:
        if len(distanceValues) == 0:
            raise ValueError("Distance summary cannot be computed because no distance values are available.")

        values = np.asarray(distanceValues, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Distance summary cannot be computed because distance values are invalid.")

        return {
            "MinDistanceMm": float(np.min(values)),
            "MeanDistanceMm": float(np.mean(values)),
            "MedianDistanceMm": float(np.median(values)),
            "P20DistanceMm": float(np.quantile(values, 0.20)),
            "P80DistanceMm": float(np.quantile(values, 0.80)),
        }

    @staticmethod
    def computeDistanceThresholdSummary(distanceValues: Sequence[float]) -> dict[str, float | int]:
        if len(distanceValues) == 0:
            raise ValueError("Distance threshold summary cannot be computed because no distance values are available.")

        values = np.asarray(distanceValues, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Distance threshold summary cannot be computed because distance values are invalid.")

        totalValueCount = int(values.size)
        countBelow0 = int(np.count_nonzero(values < 0.0))
        countBelow2 = int(np.count_nonzero(values < 2.0))
        countBelow5 = int(np.count_nonzero(values < 5.0))
        countAtLeast5 = int(np.count_nonzero(values >= 5.0))

        return {
            "CountBelow0Mm": countBelow0,
            "PercentBelow0Mm": float(100.0 * countBelow0 / totalValueCount),
            "CountBelow2Mm": countBelow2,
            "PercentBelow2Mm": float(100.0 * countBelow2 / totalValueCount),
            "CountBelow5Mm": countBelow5,
            "PercentBelow5Mm": float(100.0 * countBelow5 / totalValueCount),
            "CountAtLeast5Mm": countAtLeast5,
            "PercentAtLeast5Mm": float(100.0 * countAtLeast5 / totalValueCount),
        }

    @staticmethod
    def computeEntryPointSpacingMm(trajectoryA: ProbeTrajectory, trajectoryB: ProbeTrajectory) -> float:
        pointA = np.asarray(trajectoryA.entryPointRAS, dtype=float)
        pointB = np.asarray(trajectoryB.entryPointRAS, dtype=float)
        return float(np.linalg.norm(pointA - pointB))

    @staticmethod
    def computeTargetPointSpacingMm(trajectoryA: ProbeTrajectory, trajectoryB: ProbeTrajectory) -> float:
        pointA = np.asarray(trajectoryA.targetPointRAS, dtype=float)
        pointB = np.asarray(trajectoryB.targetPointRAS, dtype=float)
        return float(np.linalg.norm(pointA - pointB))

    @staticmethod
    def _segmentToSegmentDistanceMm(
        pointP0: Sequence[float],
        pointP1: Sequence[float],
        pointQ0: Sequence[float],
        pointQ1: Sequence[float],
    ) -> float:
        p0 = np.asarray(pointP0, dtype=float)
        p1 = np.asarray(pointP1, dtype=float)
        q0 = np.asarray(pointQ0, dtype=float)
        q1 = np.asarray(pointQ1, dtype=float)
        u = p1 - p0
        v = q1 - q0
        w = p0 - q0

        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w))
        e = float(np.dot(v, w))
        determinant = (a * c) - (b * b)
        epsilon = 1e-8

        sN = 0.0
        sD = determinant
        tN = 0.0
        tD = determinant

        if determinant < epsilon:
            sN = 0.0
            sD = 1.0
            tN = e
            tD = c
        else:
            sN = (b * e) - (c * d)
            tN = (a * e) - (b * d)
            if sN < 0.0:
                sN = 0.0
                tN = e
                tD = c
            elif sN > sD:
                sN = sD
                tN = e + b
                tD = c

        if tN < 0.0:
            tN = 0.0
            if -d < 0.0:
                sN = 0.0
            elif -d > a:
                sN = sD
            else:
                sN = -d
                sD = a
        elif tN > tD:
            tN = tD
            if (-d + b) < 0.0:
                sN = 0.0
            elif (-d + b) > a:
                sN = sD
            else:
                sN = -d + b
                sD = a

        segmentAParam = 0.0 if abs(sN) < epsilon else (sN / sD)
        segmentBParam = 0.0 if abs(tN) < epsilon else (tN / tD)
        delta = w + (segmentAParam * u) - (segmentBParam * v)
        return float(np.linalg.norm(delta))

    @staticmethod
    def computeInterProbeDistanceMm(trajectoryA: ProbeTrajectory, trajectoryB: ProbeTrajectory) -> float:
        # First-pass definition: minimum distance between trajectory centerline line segments.
        return SurgicalVision3D_PlannerLogic._segmentToSegmentDistanceMm(
            trajectoryA.entryPointRAS,
            trajectoryA.targetPointRAS,
            trajectoryB.entryPointRAS,
            trajectoryB.targetPointRAS,
        )

    @staticmethod
    def computeProbeAxisAngleDeg(trajectoryA: ProbeTrajectory, trajectoryB: ProbeTrajectory) -> float:
        directionA = _normalize_vector(trajectoryA.directionVector)
        directionB = _normalize_vector(trajectoryB.directionVector)
        # Treat opposite directions as parallel for inter-probe parallelism checks.
        cosineValue = float(np.clip(abs(np.dot(directionA, directionB)), 0.0, 1.0))
        return float(np.degrees(np.arccos(cosineValue)))

    @staticmethod
    def computePairwiseProbeVolumeOverlap(trajectoryA: ProbeTrajectory, trajectoryB: ProbeTrajectory) -> float:
        interProbeDistance = SurgicalVision3D_PlannerLogic.computeInterProbeDistanceMm(trajectoryA, trajectoryB)
        referenceLength = max(min(float(trajectoryA.lengthMm), float(trajectoryB.lengthMm)), 1e-3)
        overlapProxy = max(0.0, 1.0 - (interProbeDistance / referenceLength))
        # Conservative proxy in percent: 0 means no estimated redundancy, 100 means maximal redundancy.
        return float(100.0 * overlapProxy)

    @staticmethod
    def formatProbePairFailedConstraintNames(failedConstraintNames: Sequence[str]) -> str:
        if len(failedConstraintNames) == 0:
            return ""
        return ";".join(sorted(set(str(name) for name in failedConstraintNames if name)))

    def evaluateProbePairCoordination(
        self,
        trajectoryA: ProbeTrajectory,
        trajectoryB: ProbeTrajectory,
        settings: ProbeCoordinationConstraintSettings,
    ) -> dict[str, float | int | bool | str]:
        interProbeDistance = self.computeInterProbeDistanceMm(trajectoryA, trajectoryB)
        entrySpacing = self.computeEntryPointSpacingMm(trajectoryA, trajectoryB)
        targetSpacing = self.computeTargetPointSpacingMm(trajectoryA, trajectoryB)
        axisAngleDeg = self.computeProbeAxisAngleDeg(trajectoryA, trajectoryB)
        overlapPercent = self.computePairwiseProbeVolumeOverlap(trajectoryA, trajectoryB)

        failedConstraints: list[str] = []
        if settings.enableInterProbeDistanceRule:
            if interProbeDistance < float(settings.minInterProbeDistanceMm):
                failedConstraints.append("InterProbeDistanceBelowMin")
            if interProbeDistance > float(settings.maxInterProbeDistanceMm):
                failedConstraints.append("InterProbeDistanceAboveMax")
        if settings.enableEntrySpacingRule and entrySpacing < float(settings.minEntryPointSpacingMm):
            failedConstraints.append("EntryPointSpacingBelowMin")
        if settings.enableTargetSpacingRule and targetSpacing < float(settings.minTargetPointSpacingMm):
            failedConstraints.append("TargetPointSpacingBelowMin")
        if settings.enableAngleRule and axisAngleDeg < float(settings.maxParallelAngleDeg):
            failedConstraints.append("ProbeAxesTooParallel")
        if settings.enableOverlapRule and overlapPercent > float(settings.maxAllowedOverlapPercentBetweenPerProbeVolumes):
            failedConstraints.append("OverlapRedundancyAboveMax")

        probeAIndex = min(int(trajectoryA.trajectoryIndex) + 1, int(trajectoryB.trajectoryIndex) + 1)
        probeBIndex = max(int(trajectoryA.trajectoryIndex) + 1, int(trajectoryB.trajectoryIndex) + 1)
        failedConstraintNames = self.formatProbePairFailedConstraintNames(failedConstraints)
        return {
            "ProbeAIndex": probeAIndex,
            "ProbeBIndex": probeBIndex,
            "IsFeasible": len(failedConstraints) == 0,
            "FailedConstraintCount": int(len(failedConstraints)),
            "FailedConstraintNames": failedConstraintNames,
            "InterProbeDistanceMm": float(interProbeDistance),
            "EntryPointSpacingMm": float(entrySpacing),
            "TargetPointSpacingMm": float(targetSpacing),
            "ProbeAxisAngleDeg": float(axisAngleDeg),
            "OverlapRedundancyPercent": float(overlapPercent),
        }

    @staticmethod
    def _isPointInsideClosedSurface(pointRAS: Sequence[float], closedSurface: vtk.vtkPolyData) -> bool:
        points = vtk.vtkPoints()
        points.InsertNextPoint(float(pointRAS[0]), float(pointRAS[1]), float(pointRAS[2]))
        vertices = vtk.vtkCellArray()
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(0)
        pointPolyData = vtk.vtkPolyData()
        pointPolyData.SetPoints(points)
        pointPolyData.SetVerts(vertices)

        enclosedPoints = vtk.vtkSelectEnclosedPoints()
        enclosedPoints.SetSurfaceData(closedSurface)
        enclosedPoints.SetInputData(pointPolyData)
        enclosedPoints.Update()
        return bool(enclosedPoints.IsInside(0) == 1)

    def evaluateNoTouchArrangement(
        self,
        trajectories: Sequence[ProbeTrajectory],
        tumorSegmentation: vtkMRMLSegmentationNode | None,
    ) -> dict[str, float | int | bool | str]:
        if tumorSegmentation is None:
            return {
                "NoTouchChecked": False,
                "NoTouchPass": False,
                "Reason": "Tumor segmentation is required when no-touch checking is enabled.",
                "EntryPointsInsideTumorCount": 0,
                "FailedTrajectoryIndices": "",
            }

        tumorSegmentID = self.findPreferredSegmentID(
            tumorSegmentation,
            DEFAULT_TUMOR_SEGMENT_NAMES,
            "no-touch evaluation",
            fallbackToFirst=True,
        )
        self._ensureSegmentationHasClosedSurface(tumorSegmentation)
        tumorSurface = vtk.vtkPolyData()
        tumorSegmentation.GetClosedSurfaceRepresentation(tumorSegmentID, tumorSurface)
        if tumorSurface.GetNumberOfPoints() <= 0:
            raise RuntimeError("No-touch evaluation failed because tumor closed-surface representation is empty.")

        failedTrajectoryIndices: list[int] = []
        for trajectory in trajectories:
            if self._isPointInsideClosedSurface(trajectory.entryPointRAS, tumorSurface):
                failedTrajectoryIndices.append(int(trajectory.trajectoryIndex) + 1)

        return {
            "NoTouchChecked": True,
            "NoTouchPass": len(failedTrajectoryIndices) == 0,
            "Reason": "" if len(failedTrajectoryIndices) == 0 else "Entry point is inside tumor for one or more trajectories.",
            "EntryPointsInsideTumorCount": int(len(failedTrajectoryIndices)),
            "FailedTrajectoryIndices": ",".join(str(index) for index in failedTrajectoryIndices),
        }

    @staticmethod
    def aggregateProbeCoordinationFailures(pairRows: Sequence[dict[str, float | int | bool | str]]) -> str:
        failureNames: set[str] = set()
        for row in pairRows:
            rowFailures = str(row.get("FailedConstraintNames", ""))
            for name in rowFailures.split(";"):
                cleaned = name.strip()
                if cleaned:
                    failureNames.add(cleaned)
        return ";".join(sorted(failureNames))

    def evaluatePlanProbeCoordination(
        self,
        trajectories: Sequence[ProbeTrajectory],
        settings: ProbeCoordinationConstraintSettings,
        tumorSegmentation: vtkMRMLSegmentationNode | None,
    ) -> tuple[list[dict[str, float | int | bool | str]], dict[str, float | int | bool | str], dict[str, float | int | bool | str]]:
        if len(trajectories) == 0:
            raise ValueError("At least one trajectory is required for probe coordination evaluation.")

        pairRows: list[dict[str, float | int | bool | str]] = []
        for trajectoryIndexA in range(len(trajectories)):
            for trajectoryIndexB in range(trajectoryIndexA + 1, len(trajectories)):
                pairRows.append(
                    self.evaluateProbePairCoordination(
                        trajectories[trajectoryIndexA],
                        trajectories[trajectoryIndexB],
                        settings,
                    )
                )
        pairRows.sort(key=lambda row: (int(row.get("ProbeAIndex", 0)), int(row.get("ProbeBIndex", 0))))

        pairCount = int(len(pairRows))
        feasiblePairCount = int(sum(1 for row in pairRows if bool(row.get("IsFeasible", False))))
        infeasiblePairCount = int(pairCount - feasiblePairCount)
        allPairsFeasible = infeasiblePairCount == 0

        noTouchSummary = {
            "NoTouchChecked": False,
            "NoTouchPass": True,
            "Reason": "",
            "EntryPointsInsideTumorCount": 0,
            "FailedTrajectoryIndices": "",
        }
        if settings.enableNoTouchCheck:
            noTouchSummary = self.evaluateNoTouchArrangement(trajectories, tumorSegmentation)

        noTouchPass = bool(noTouchSummary.get("NoTouchPass", True))
        coordinationFailureReasons: list[str] = []
        if settings.requireAllProbePairsFeasible and not allPairsFeasible:
            coordinationFailureReasons.append("ProbePairCoordinationFailed")
        if settings.enableNoTouchCheck and not noTouchPass:
            coordinationFailureReasons.append("NoTouchCheckFailed")

        coordinationGatePass = (
            (not settings.requireAllProbePairsFeasible or allPairsFeasible)
            and (not settings.enableNoTouchCheck or noTouchPass)
        )

        planSummary: dict[str, float | int | bool | str] = {
            "ScenarioOrPlanName": "CurrentPlan",
            "ProbeCount": int(len(trajectories)),
            "PairCount": pairCount,
            "FeasiblePairCount": feasiblePairCount,
            "InfeasiblePairCount": infeasiblePairCount,
            "AllPairsFeasible": bool(allPairsFeasible),
            "AggregatedFailedConstraintNames": self.aggregateProbeCoordinationFailures(pairRows),
            "NoTouchPass": bool(noTouchPass),
            "CoordinationGatePass": bool(coordinationGatePass),
            "CoordinationFailureSummary": ";".join(coordinationFailureReasons),
        }
        return pairRows, planSummary, noTouchSummary

    @staticmethod
    def extractTrajectoriesFromPointPairs(
        pointsRAS: Sequence[Sequence[float]],
        strictEven: bool = True,
    ) -> list[ProbeTrajectory]:
        pointCount = len(pointsRAS)
        if pointCount < 2:
            raise ValueError("At least two control points are required.")
        if pointCount % 2 != 0:
            message = f"Expected an even number of control points but found {pointCount}."
            if strictEven:
                raise ValueError(message)
            logging.warning("%s The last point is ignored.", message)
            pointCount -= 1

        trajectories: list[ProbeTrajectory] = []
        for pointIndex in range(0, pointCount, 2):
            # Point-pair order is entry first, applicator endpoint second.
            entry = np.array(pointsRAS[pointIndex], dtype=float)
            endpoint = np.array(pointsRAS[pointIndex + 1], dtype=float)
            if endpoint.size != 3 or entry.size != 3:
                raise ValueError(
                    f"Control-point pair {pointIndex}-{pointIndex + 1} must contain 3D coordinates."
                )
            if not np.all(np.isfinite(endpoint)) or not np.all(np.isfinite(entry)):
                raise ValueError(
                    f"Control-point pair {pointIndex}-{pointIndex + 1} contains non-finite coordinates."
                )
            direction = endpoint - entry
            length = float(np.linalg.norm(direction))
            if not math.isfinite(length) or length <= 1e-8:
                raise ValueError(f"Control-point pair {pointIndex}-{pointIndex + 1} has zero length.")

            trajectory = ProbeTrajectory(
                entryPointRAS=tuple(entry.tolist()),
                targetPointRAS=tuple(endpoint.tolist()),
                directionVector=tuple((direction / length).tolist()),
                lengthMm=length,
                trajectoryIndex=len(trajectories),
                label=f"Trajectory {len(trajectories) + 1}",
                sourceControlPointIndices=(pointIndex, pointIndex + 1),
                role=f"Trajectory {len(trajectories) + 1:02d}",
                angleDeg=None,
                radialOffsetMm=0.0,
                derivedFromMaster=False,
            )
            trajectories.append(trajectory)
        return trajectories

    @staticmethod
    def _validateTrajectoryGeometryForPlacement(trajectory: ProbeTrajectory) -> None:
        entry = np.asarray(trajectory.entryPointRAS, dtype=float)
        target = np.asarray(trajectory.targetPointRAS, dtype=float)
        direction = np.asarray(trajectory.directionVector, dtype=float)
        if entry.size != 3 or target.size != 3 or direction.size != 3:
            raise ValueError(
                f"Trajectory {int(trajectory.trajectoryIndex) + 1} has invalid coordinate dimensionality."
            )
        if not np.all(np.isfinite(entry)) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(direction)):
            raise ValueError(f"Trajectory {int(trajectory.trajectoryIndex) + 1} has non-finite coordinates.")
        lengthMm = float(trajectory.lengthMm)
        if not math.isfinite(lengthMm) or lengthMm <= 1e-8:
            raise ValueError(f"Trajectory {int(trajectory.trajectoryIndex) + 1} has an invalid length.")
        directionNorm = float(np.linalg.norm(direction))
        if not math.isfinite(directionNorm) or directionNorm <= 1e-8:
            raise ValueError(f"Trajectory {int(trajectory.trajectoryIndex) + 1} has an invalid direction vector.")

    def extractTrajectoriesFromMarkups(
        self,
        endpointsMarkups: vtkMRMLMarkupsFiducialNode | None,
        strictEven: bool = True,
    ) -> list[ProbeTrajectory]:
        if not endpointsMarkups:
            raise ValueError("Endpoint markups node is required.")

        controlPoints: list[tuple[float, float, float]] = []
        for pointIndex in range(endpointsMarkups.GetNumberOfControlPoints()):
            pointPosition = [0.0, 0.0, 0.0]
            endpointsMarkups.GetNthControlPointPosition(pointIndex, pointPosition)
            controlPoints.append((pointPosition[0], pointPosition[1], pointPosition[2]))

        trajectories = self.extractTrajectoriesFromPointPairs(controlPoints, strictEven=strictEven)
        for trajectory in trajectories:
            if trajectory.sourceControlPointIndices is None:
                continue
            endpointLabelIndex = int(trajectory.sourceControlPointIndices[1])
            label = endpointsMarkups.GetNthControlPointLabel(endpointLabelIndex)
            if not label:
                label = endpointsMarkups.GetNthControlPointLabel(int(trajectory.sourceControlPointIndices[0]))
            if label:
                trajectory.label = label
        return trajectories

    def placeProbeInstances(
        self,
        referenceProbeSegmentation: vtkMRMLSegmentationNode | None,
        trajectories: Sequence[ProbeTrajectory],
        axialPlacementOffsetMm: float | None = None,
    ) -> list[str]:
        referenceProbeSegmentation = self.resolveUsableReferenceProbeSegmentation(referenceProbeSegmentation)
        if not referenceProbeSegmentation:
            raise ValueError("Reference probe segmentation is required.")
        if len(trajectories) == 0:
            raise ValueError("No trajectories were provided.")
        if self.segmentationSegmentCount(referenceProbeSegmentation) <= 0:
            raise RuntimeError(
                f"Reference probe segmentation '{referenceProbeSegmentation.GetName()}' has no segments. "
                "Select a template from 'Reference probe segmentation'."
            )

        self._ensureSegmentationHasClosedSurface(referenceProbeSegmentation)
        sourceSegmentID = self.getWorkingSegmentID(referenceProbeSegmentation, "reference probe placement")
        sourceSurface = vtk.vtkPolyData()
        referenceProbeSegmentation.GetClosedSurfaceRepresentation(sourceSegmentID, sourceSurface)
        if sourceSurface.GetNumberOfPoints() <= 0 or sourceSurface.GetNumberOfCells() <= 0:
            raise RuntimeError("Reference probe segmentation has no usable closed-surface geometry.")
        if not self._polyDataHasFinitePoints(sourceSurface):
            raise RuntimeError("Reference probe segmentation has non-finite closed-surface coordinates.")

        if axialPlacementOffsetMm is None:
            try:
                axialPlacementOffsetMm = float(
                    referenceProbeSegmentation.GetAttribute(REFERENCE_PROBE_TEMPLATE_AXIAL_PLACEMENT_OFFSET_MM_ATTRIBUTE) or 0.0
                )
            except Exception:
                axialPlacementOffsetMm = 0.0
        axialPlacementOffsetMm = float(axialPlacementOffsetMm or 0.0)

        generatedProbeNodeIDs: list[str] = []
        try:
            with self._mrmlBatchState("BatchProcessState"):
                for trajectory in trajectories:
                    self._validateTrajectoryGeometryForPlacement(trajectory)
                    probeNode = self._cloneReferenceProbe(
                        referenceProbeSegmentation,
                        trajectory.trajectoryIndex,
                        sourceSurface=sourceSurface,
                    )
                    self._placeProbeNodeAlongTrajectory(
                        probeNode,
                        trajectory,
                        axialPlacementOffsetMm=axialPlacementOffsetMm,
                    )
                    trajectory.generatedProbeNodeID = probeNode.GetID()
                    trajectory.status = "placed"
                    generatedProbeNodeIDs.append(probeNode.GetID())
        except Exception:
            self.removeNodesByIDs(generatedProbeNodeIDs)
            raise

        return generatedProbeNodeIDs

    @staticmethod
    def _configureTrajectoryLineDisplay(
        lineDisplayNode,
        colorRGB: tuple[float, float, float] | None = None,
    ) -> None:
        if not lineDisplayNode:
            return
        lineDisplayNode.SetVisibility(True)
        if hasattr(lineDisplayNode, "SetVisibility2D"):
            lineDisplayNode.SetVisibility2D(True)
        if hasattr(lineDisplayNode, "SetVisibility3D"):
            lineDisplayNode.SetVisibility3D(True)
        if hasattr(lineDisplayNode, "SetSliceProjection"):
            lineDisplayNode.SetSliceProjection(True)
        if hasattr(lineDisplayNode, "SetSliceProjectionOpacity"):
            lineDisplayNode.SetSliceProjectionOpacity(1.0)
        if hasattr(lineDisplayNode, "SetSliceProjectionOutlinedBehindSlicePlane"):
            lineDisplayNode.SetSliceProjectionOutlinedBehindSlicePlane(False)
        if hasattr(lineDisplayNode, "SetLineThickness"):
            lineDisplayNode.SetLineThickness(0.9 * LINE_THICKNESS_SCALE)
        elif hasattr(lineDisplayNode, "SetLineWidth"):
            lineDisplayNode.SetLineWidth(4.0 * LINE_THICKNESS_SCALE)
        if hasattr(lineDisplayNode, "SetOpacity"):
            lineDisplayNode.SetOpacity(1.0)
        lineDisplayNode.SetPropertiesLabelVisibility(False)
        lineDisplayNode.SetPointLabelsVisibility(False)
        lineDisplayNode.SetSelectable(False)
        if colorRGB is not None:
            lineDisplayNode.SetColor(float(colorRGB[0]), float(colorRGB[1]), float(colorRGB[2]))

    def createTrajectoryLines(self, trajectories: Sequence[ProbeTrajectory], clearExisting: bool = True) -> list[str]:
        if clearExisting:
            self.removeGeneratedTrajectoryLines()

        generatedLineNodeIDs: list[str] = []
        for trajectory in trajectories:
            self._validateTrajectoryGeometryForPlacement(trajectory)
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", f"SV3D Trajectory {trajectory.trajectoryIndex + 1:02d}")
            pointArray = np.array([trajectory.entryPointRAS, trajectory.targetPointRAS], dtype=float)
            slicer.util.updateMarkupsControlPointsFromArray(lineNode, pointArray)
            lineNode.SetAttribute(GENERATED_TRAJECTORY_LINE_ATTRIBUTE, "1")
            if hasattr(lineNode, "CreateDefaultDisplayNodes"):
                lineNode.CreateDefaultDisplayNodes()

            lineDisplayNode = lineNode.GetDisplayNode()
            self._configureTrajectoryLineDisplay(lineDisplayNode)

            generatedLineNodeIDs.append(lineNode.GetID())
        return generatedLineNodeIDs

    def createOrUpdateDerivedTrajectoryPreview(
        self,
        trajectories: Sequence[ProbeTrajectory],
        clearExisting: bool = True,
    ) -> list[str]:
        if clearExisting:
            self.removeGeneratedTrajectoryLines()

        generatedLineNodeIDs: list[str] = []
        derivedOrdinal = 0
        for trajectory in trajectories:
            self._validateTrajectoryGeometryForPlacement(trajectory)
            isMaster = (not trajectory.derivedFromMaster) and (str(trajectory.role or "").strip().lower() == "master")
            if isMaster:
                nodeName = "SV3D Master Trajectory"
            elif trajectory.derivedFromMaster:
                derivedOrdinal += 1
                nodeName = f"SV3D Derived Trajectory {derivedOrdinal:02d}"
            else:
                nodeName = f"SV3D Trajectory {int(trajectory.trajectoryIndex) + 1:02d}"

            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", nodeName)
            pointArray = np.array([trajectory.entryPointRAS, trajectory.targetPointRAS], dtype=float)
            slicer.util.updateMarkupsControlPointsFromArray(lineNode, pointArray)
            lineNode.SetAttribute(GENERATED_TRAJECTORY_LINE_ATTRIBUTE, "1")
            lineNode.SetAttribute("SurgicalVision3D_Planner.TrajectoryRole", str(trajectory.role or ""))
            lineNode.SetAttribute("SurgicalVision3D_Planner.DerivedFromMaster", "1" if trajectory.derivedFromMaster else "0")
            if hasattr(lineNode, "CreateDefaultDisplayNodes"):
                lineNode.CreateDefaultDisplayNodes()

            lineDisplayNode = lineNode.GetDisplayNode()
            if isMaster:
                lineColor = (0.10, 0.75, 0.95)
            elif trajectory.derivedFromMaster:
                lineColor = (1.00, 0.62, 0.15)
            else:
                lineColor = (0.25, 0.80, 0.35)
            self._configureTrajectoryLineDisplay(lineDisplayNode, colorRGB=lineColor)

            generatedLineNodeIDs.append(lineNode.GetID())

        return generatedLineNodeIDs

    @staticmethod
    def _lineIntersectsClosedSurface(
        lineStartRAS: Sequence[float],
        lineEndRAS: Sequence[float],
        closedSurface: vtk.vtkPolyData,
    ) -> bool:
        if (
            closedSurface is None
            or closedSurface.GetNumberOfPoints() <= 0
            or closedSurface.GetNumberOfCells() <= 0
        ):
            return False

        # Avoid OBBTree eigen-decomposition instability on degenerate surfaces
        # (seen as vtkMath::Jacobi warnings) by using a static cell locator.
        cellLocator = vtk.vtkStaticCellLocator()
        cellLocator.SetDataSet(closedSurface)
        cellLocator.BuildLocator()

        intersectionPoints = vtk.vtkPoints()
        intersectionCellIds = vtk.vtkIdList()
        try:
            intersectionCount = cellLocator.IntersectWithLine(
                lineStartRAS,
                lineEndRAS,
                1e-6,
                intersectionPoints,
                intersectionCellIds,
            )
        except TypeError:
            intersectionCount = cellLocator.IntersectWithLine(
                lineStartRAS,
                lineEndRAS,
                intersectionPoints,
                intersectionCellIds,
            )
        return int(intersectionCount) > 0 or intersectionPoints.GetNumberOfPoints() > 0

    @staticmethod
    def _lineToClosedSurfaceMinDistanceMm(
        lineStartRAS: Sequence[float],
        lineEndRAS: Sequence[float],
        closedSurface: vtk.vtkPolyData,
        sampleCount: int = BEGINNER_TRAJECTORY_DISTANCE_SAMPLE_COUNT,
    ) -> float:
        if closedSurface is None or closedSurface.GetNumberOfPoints() <= 0:
            return float("nan")
        implicitDistance = vtk.vtkImplicitPolyDataDistance()
        implicitDistance.SetInput(closedSurface)
        startPoint = np.array(lineStartRAS, dtype=float)
        endPoint = np.array(lineEndRAS, dtype=float)
        if sampleCount <= 1:
            sampleCount = 2
        distances = [
            abs(float(implicitDistance.EvaluateFunction((startPoint + ((endPoint - startPoint) * sampleIndex / float(sampleCount - 1))).tolist())))
            for sampleIndex in range(sampleCount)
        ]
        finiteDistances = [distance for distance in distances if math.isfinite(distance)]
        return float(min(finiteDistances)) if len(finiteDistances) > 0 else float("nan")

    @staticmethod
    def _polyDataHasFinitePoints(polyData: vtk.vtkPolyData | None) -> bool:
        if polyData is None:
            return False
        pointCount = int(polyData.GetNumberOfPoints())
        if pointCount <= 0:
            return False
        bounds = [0.0] * 6
        polyData.GetBounds(bounds)
        if any(not math.isfinite(float(bound)) for bound in bounds):
            return False
        point = [0.0, 0.0, 0.0]
        for pointIndex in range(pointCount):
            polyData.GetPoint(pointIndex, point)
            if not (
                math.isfinite(float(point[0]))
                and math.isfinite(float(point[1]))
                and math.isfinite(float(point[2]))
            ):
                return False
        return True

    @staticmethod
    def _polyDataSummaryText(polyData: vtk.vtkPolyData | None) -> str:
        if polyData is None:
            return "none"
        pointCount = int(polyData.GetNumberOfPoints())
        cellCount = int(polyData.GetNumberOfCells())
        bounds = [0.0] * 6
        try:
            polyData.GetBounds(bounds)
        except Exception:
            boundsText = "(unavailable)"
        else:
            boundsText = (
                f"({bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, "
                f"{bounds[3]:.3f}, {bounds[4]:.3f}, {bounds[5]:.3f})"
            )
        return f"points={pointCount}, cells={cellCount}, bounds={boundsText}"

    @staticmethod
    def _modelPolyDataSummaryText(modelNode: vtkMRMLModelNode | None) -> str:
        if not modelNode:
            return "none"
        mesh = modelNode.GetMesh() if hasattr(modelNode, "GetMesh") else None
        polyData = vtk.vtkPolyData.SafeDownCast(mesh)
        modelName = str(modelNode.GetName() or "(unnamed)")
        return f"name='{modelName}', {SurgicalVision3D_PlannerLogic._polyDataSummaryText(polyData)}"

    @classmethod
    def _prepareClosedSurfaceForTrajectoryValidation(
        cls,
        closedSurface: vtk.vtkPolyData | None,
    ) -> vtk.vtkPolyData | None:
        if (
            closedSurface is None
            or closedSurface.GetNumberOfPoints() <= 0
            or closedSurface.GetNumberOfCells() <= 0
        ):
            return None

        surfaceCopy = vtk.vtkPolyData()
        surfaceCopy.DeepCopy(closedSurface)

        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(surfaceCopy)
        if hasattr(triangleFilter, "PassLinesOff"):
            triangleFilter.PassLinesOff()
        if hasattr(triangleFilter, "PassVertsOff"):
            triangleFilter.PassVertsOff()
        triangleFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(triangleFilter.GetOutputPort())
        cleanFilter.Update()

        preparedSurface = vtk.vtkPolyData()
        preparedSurface.DeepCopy(cleanFilter.GetOutput())
        if preparedSurface.GetNumberOfPoints() <= 2 or preparedSurface.GetNumberOfCells() <= 0:
            return None
        if not cls._polyDataHasFinitePoints(preparedSurface):
            return None
        return preparedSurface

    def autoAdjustMasterTrajectoryEndpoint(
        self,
        trajectory: ProbeTrajectory,
        criticalStructuresSegmentation: vtkMRMLSegmentationNode | None,
        tumorSegmentation: vtkMRMLSegmentationNode | None,
        maxEndpointShiftMm: float = AUTO_ADJUST_MAX_ENDPOINT_SHIFT_MM,
        shiftStepMm: float = AUTO_ADJUST_ENDPOINT_SHIFT_STEP_MM,
        azimuthSampleCount: int = AUTO_ADJUST_AZIMUTH_SAMPLE_COUNT,
    ) -> EndpointAutoAdjustResult:
        if tumorSegmentation is None:
            raise ValueError("Tumor segmentation is required for endpoint auto-adjust.")
        if criticalStructuresSegmentation is None:
            raise ValueError("Critical-structures segmentation is required for endpoint auto-adjust.")

        trajectoryLengthMm = float(trajectory.lengthMm)
        if trajectoryLengthMm <= 1e-6:
            raise ValueError("Master trajectory must have non-zero length for endpoint auto-adjust.")

        maxEndpointShiftMm = float(max(0.0, maxEndpointShiftMm))
        shiftStepMm = float(max(0.0, shiftStepMm))
        azimuthSampleCount = max(1, int(azimuthSampleCount))
        statusTextNoSolution = f"No safe adjustment found within {maxEndpointShiftMm:.1f} mm; endpoint unchanged."
        if maxEndpointShiftMm <= 0.0 or shiftStepMm <= 0.0:
            return EndpointAutoAdjustResult(
                applied=False,
                reason="InvalidSearchConfiguration",
                statusText=statusTextNoSolution,
                maxEndpointShiftMm=maxEndpointShiftMm,
                shiftStepMm=shiftStepMm,
                azimuthSampleCount=azimuthSampleCount,
                checkedCandidateCount=0,
                insideTumorCandidateCount=0,
                zeroIntersectionCandidateCount=0,
                selectedTargetPointRAS=tuple(float(value) for value in trajectory.targetPointRAS),
            )

        tumorSegmentID = self.findPreferredSegmentID(
            tumorSegmentation,
            DEFAULT_TUMOR_SEGMENT_NAMES,
            "master trajectory endpoint auto-adjust",
            fallbackToFirst=True,
        )
        self._ensureSegmentationHasClosedSurface(tumorSegmentation)
        tumorClosedSurface = vtk.vtkPolyData()
        tumorSegmentation.GetClosedSurfaceRepresentation(tumorSegmentID, tumorClosedSurface)
        preparedTumorSurface = self._prepareClosedSurfaceForTrajectoryValidation(tumorClosedSurface)
        if preparedTumorSurface is None:
            raise RuntimeError("master trajectory endpoint auto-adjust: tumor closed-surface geometry is unavailable.")

        excludedSegmentID = ""
        if criticalStructuresSegmentation.GetID() == tumorSegmentation.GetID():
            excludedSegmentID = tumorSegmentID

        preparedCriticalSurfaces: list[vtk.vtkPolyData] = []
        segmentInfos = self.getValidSegmentationSegments(
            criticalStructuresSegmentation,
            "master trajectory endpoint auto-adjust",
        )
        for segmentInfo in segmentInfos:
            segmentID = str(segmentInfo["segmentID"])
            if excludedSegmentID and segmentID == excludedSegmentID:
                continue
            closedSurface = vtk.vtkPolyData()
            criticalStructuresSegmentation.GetClosedSurfaceRepresentation(segmentID, closedSurface)
            preparedSurface = self._prepareClosedSurfaceForTrajectoryValidation(closedSurface)
            if preparedSurface is None:
                logging.warning(
                    "master trajectory endpoint auto-adjust: segment '%s' has unstable closed-surface geometry and will be skipped.",
                    segmentID,
                )
                continue
            preparedCriticalSurfaces.append(preparedSurface)
        if len(preparedCriticalSurfaces) <= 0:
            raise RuntimeError(
                "master trajectory endpoint auto-adjust: no usable critical-structure closed-surface segments were available."
            )

        entryPoint = np.array(trajectory.entryPointRAS, dtype=float)
        originalTargetPoint = np.array(trajectory.targetPointRAS, dtype=float)
        originalDirection = _normalize_vector(trajectory.directionVector)
        basisU, basisV = self.createOrthogonalArrayBasis(originalDirection)

        checkedCandidateCount = 0
        insideTumorCandidateCount = 0
        zeroIntersectionCandidateCount = 0
        bestScore: tuple[float, float, int, int] | None = None
        bestCandidateTargetPoint: np.ndarray | None = None
        bestCandidateShiftMm = 0.0
        bestCandidateMinDistanceMm = float("nan")
        bestShellIndex = -1
        bestAzimuthIndex = -1

        shellCount = int(math.floor((maxEndpointShiftMm + 1e-6) / shiftStepMm))
        for shellIndex in range(1, shellCount + 1):
            requestedShiftMm = float(shellIndex) * shiftStepMm
            if requestedShiftMm > (maxEndpointShiftMm + 1e-6):
                break
            if requestedShiftMm > (2.0 * trajectoryLengthMm):
                continue

            cappedRatio = min(1.0, max(0.0, requestedShiftMm / (2.0 * trajectoryLengthMm)))
            polarAngleRad = 2.0 * math.asin(cappedRatio)
            sinPolar = math.sin(polarAngleRad)
            cosPolar = math.cos(polarAngleRad)

            for azimuthIndex in range(azimuthSampleCount):
                azimuthRad = (2.0 * math.pi * float(azimuthIndex)) / float(azimuthSampleCount)
                azimuthDirection = (math.cos(azimuthRad) * basisU) + (math.sin(azimuthRad) * basisV)
                candidateDirection = (cosPolar * originalDirection) + (sinPolar * azimuthDirection)
                candidateDirectionNorm = float(np.linalg.norm(candidateDirection))
                if candidateDirectionNorm <= 1e-8:
                    continue
                candidateDirection /= candidateDirectionNorm
                candidateTargetPoint = entryPoint + (trajectoryLengthMm * candidateDirection)
                checkedCandidateCount += 1

                if not self._isPointInsideClosedSurface(candidateTargetPoint, preparedTumorSurface):
                    continue
                insideTumorCandidateCount += 1

                intersectsCriticalStructure = False
                minDistanceMm = float("inf")
                for preparedSurface in preparedCriticalSurfaces:
                    if self._lineIntersectsClosedSurface(entryPoint, candidateTargetPoint, preparedSurface):
                        intersectsCriticalStructure = True
                    candidateDistanceMm = self._lineToClosedSurfaceMinDistanceMm(
                        entryPoint,
                        candidateTargetPoint,
                        preparedSurface,
                    )
                    if math.isfinite(candidateDistanceMm):
                        minDistanceMm = min(minDistanceMm, float(candidateDistanceMm))
                if intersectsCriticalStructure:
                    continue

                zeroIntersectionCandidateCount += 1
                if not math.isfinite(minDistanceMm):
                    minDistanceMm = float("nan")
                actualShiftMm = float(np.linalg.norm(candidateTargetPoint - originalTargetPoint))
                distanceScore = float(minDistanceMm) if math.isfinite(minDistanceMm) else -math.inf
                candidateScore = (-distanceScore, actualShiftMm, int(shellIndex), int(azimuthIndex))
                if bestScore is None or candidateScore < bestScore:
                    bestScore = candidateScore
                    bestCandidateTargetPoint = candidateTargetPoint
                    bestCandidateShiftMm = actualShiftMm
                    bestCandidateMinDistanceMm = float(minDistanceMm)
                    bestShellIndex = int(shellIndex)
                    bestAzimuthIndex = int(azimuthIndex)

        if bestCandidateTargetPoint is None:
            return EndpointAutoAdjustResult(
                applied=False,
                reason="NoZeroIntersectionCandidateWithinCap",
                statusText=statusTextNoSolution,
                maxEndpointShiftMm=maxEndpointShiftMm,
                shiftStepMm=shiftStepMm,
                azimuthSampleCount=azimuthSampleCount,
                checkedCandidateCount=int(checkedCandidateCount),
                insideTumorCandidateCount=int(insideTumorCandidateCount),
                zeroIntersectionCandidateCount=int(zeroIntersectionCandidateCount),
                selectedTargetPointRAS=tuple(float(value) for value in originalTargetPoint.tolist()),
            )

        return EndpointAutoAdjustResult(
            applied=True,
            reason="AppliedBestZeroIntersectionCandidate",
            statusText="Auto-adjust found a safe endpoint candidate.",
            maxEndpointShiftMm=maxEndpointShiftMm,
            shiftStepMm=shiftStepMm,
            azimuthSampleCount=azimuthSampleCount,
            checkedCandidateCount=int(checkedCandidateCount),
            insideTumorCandidateCount=int(insideTumorCandidateCount),
            zeroIntersectionCandidateCount=int(zeroIntersectionCandidateCount),
            selectedEndpointShiftMm=float(bestCandidateShiftMm),
            selectedMinDistanceMm=float(bestCandidateMinDistanceMm),
            selectedTargetPointRAS=tuple(float(value) for value in bestCandidateTargetPoint.tolist()),
            selectedShellIndex=int(bestShellIndex),
            selectedAzimuthIndex=int(bestAzimuthIndex),
        )

    def evaluateMasterTrajectoryAgainstCriticalStructures(
        self,
        trajectory: ProbeTrajectory,
        criticalStructuresSegmentation: vtkMRMLSegmentationNode | None,
        tumorSegmentation: vtkMRMLSegmentationNode | None = None,
    ) -> tuple[list[dict[str, float | int | bool | str]], dict[str, float | int | bool | str]]:
        if criticalStructuresSegmentation is None:
            raise ValueError("Critical-structures segmentation is required for trajectory validation.")

        excludedSegmentID = ""
        if tumorSegmentation and criticalStructuresSegmentation.GetID() == tumorSegmentation.GetID():
            excludedSegmentID = self.findPreferredSegmentID(
                tumorSegmentation,
                DEFAULT_TUMOR_SEGMENT_NAMES,
                "master trajectory validation",
                fallbackToFirst=True,
            )

        validationRows: list[dict[str, float | int | bool | str]] = []
        try:
            segmentInfos = self.getValidSegmentationSegments(
                criticalStructuresSegmentation,
                "master trajectory validation",
            )
        except RuntimeError:
            segmentInfos = []

        for segmentInfo in segmentInfos:
            segmentID = str(segmentInfo["segmentID"])
            if excludedSegmentID and segmentID == excludedSegmentID:
                continue

            closedSurface = vtk.vtkPolyData()
            criticalStructuresSegmentation.GetClosedSurfaceRepresentation(segmentID, closedSurface)
            preparedSurface = self._prepareClosedSurfaceForTrajectoryValidation(closedSurface)
            if preparedSurface is None:
                logging.warning(
                    "master trajectory validation: segment '%s' has unstable closed-surface geometry and will be skipped.",
                    segmentID,
                )
                continue
            intersects = self._lineIntersectsClosedSurface(
                trajectory.entryPointRAS,
                trajectory.targetPointRAS,
                preparedSurface,
            )
            minDistanceMm = self._lineToClosedSurfaceMinDistanceMm(
                trajectory.entryPointRAS,
                trajectory.targetPointRAS,
                preparedSurface,
            )
            validationRows.append(
                {
                    "StructureSegmentID": segmentID,
                    "StructureName": str(segmentInfo["segmentName"]),
                    "TrajectoryIntersects": bool(intersects),
                    "MinDistanceMm": float(minDistanceMm),
                }
            )

        intersectingRows = [row for row in validationRows if bool(row.get("TrajectoryIntersects", False))]
        minDistanceValues = [
            float(row.get("MinDistanceMm", float("nan")))
            for row in validationRows
            if math.isfinite(float(row.get("MinDistanceMm", float("nan"))))
        ]
        trajectoryPass = len(intersectingRows) == 0 and len(validationRows) > 0
        if len(validationRows) == 0:
            statusText = "No critical-structure segments were available for validation."
        elif trajectoryPass:
            statusText = "Trajectory valid: no critical-structure intersections detected."
        else:
            failedStructureNames = ", ".join(str(row.get("StructureName", "")) for row in intersectingRows)
            statusText = f"Trajectory invalid: intersects {failedStructureNames}."

        summary = {
            "TrajectoryPass": bool(trajectoryPass),
            "IntersectedStructureCount": int(len(intersectingRows)),
            "CheckedStructureCount": int(len(validationRows)),
            "MinDistanceMm": float(min(minDistanceValues)) if len(minDistanceValues) > 0 else float("nan"),
            "IntersectedStructureNames": ";".join(str(row.get("StructureName", "")) for row in intersectingRows),
            "StatusText": statusText,
        }
        return validationRows, summary

    @staticmethod
    def _formatRasPoint(pointRAS: Sequence[float]) -> str:
        return ",".join(f"{float(value):.3f}" for value in pointRAS)

    def evaluateDerivedTrajectoryBundleAgainstCriticalStructures(
        self,
        trajectories: Sequence[ProbeTrajectory],
        criticalStructuresSegmentation: vtkMRMLSegmentationNode | None,
        tumorSegmentation: vtkMRMLSegmentationNode | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        bundleRows: list[dict[str, Any]] = []
        intersectingTrajectoryCount = 0
        minDistanceValues: list[float] = []
        preparedStructureSurfaces: list[vtk.vtkPolyData] = []

        if criticalStructuresSegmentation:
            excludedSegmentID = ""
            if tumorSegmentation and criticalStructuresSegmentation.GetID() == tumorSegmentation.GetID():
                excludedSegmentID = self.findPreferredSegmentID(
                    tumorSegmentation,
                    DEFAULT_TUMOR_SEGMENT_NAMES,
                    "derived trajectory bundle validation",
                    fallbackToFirst=True,
                )

            try:
                segmentInfos = self.getValidSegmentationSegments(
                    criticalStructuresSegmentation,
                    "derived trajectory bundle validation",
                )
            except RuntimeError:
                segmentInfos = []

            for segmentInfo in segmentInfos:
                segmentID = str(segmentInfo["segmentID"])
                if excludedSegmentID and segmentID == excludedSegmentID:
                    continue
                closedSurface = vtk.vtkPolyData()
                criticalStructuresSegmentation.GetClosedSurfaceRepresentation(segmentID, closedSurface)
                preparedSurface = self._prepareClosedSurfaceForTrajectoryValidation(closedSurface)
                if preparedSurface is None:
                    logging.warning(
                        "derived trajectory bundle validation: segment '%s' has unstable closed-surface geometry and will be skipped.",
                        segmentID,
                    )
                    continue
                preparedStructureSurfaces.append(preparedSurface)

        for trajectory in trajectories:
            intersects = False
            minDistanceMm = float("nan")
            checkedStructureCount = int(len(preparedStructureSurfaces))
            if checkedStructureCount > 0:
                trajectoryDistanceValues: list[float] = []
                for preparedSurface in preparedStructureSurfaces:
                    if self._lineIntersectsClosedSurface(
                        trajectory.entryPointRAS,
                        trajectory.targetPointRAS,
                        preparedSurface,
                    ):
                        intersects = True
                    surfaceDistanceMm = self._lineToClosedSurfaceMinDistanceMm(
                        trajectory.entryPointRAS,
                        trajectory.targetPointRAS,
                        preparedSurface,
                    )
                    if math.isfinite(surfaceDistanceMm):
                        trajectoryDistanceValues.append(float(surfaceDistanceMm))
                if len(trajectoryDistanceValues) > 0:
                    minDistanceMm = float(min(trajectoryDistanceValues))
            if intersects:
                intersectingTrajectoryCount += 1
            if math.isfinite(minDistanceMm):
                minDistanceValues.append(minDistanceMm)

            bundleRows.append(
                {
                    "TrajectoryIndex": int(trajectory.trajectoryIndex + 1),
                    "Role": str(trajectory.role or ""),
                    "Source": "derived" if bool(trajectory.derivedFromMaster) else "master",
                    "AngleDeg": float(trajectory.angleDeg) if trajectory.angleDeg is not None else float("nan"),
                    "RadialOffsetMm": float(trajectory.radialOffsetMm),
                    "EntryPointRAS": self._formatRasPoint(trajectory.entryPointRAS),
                    "TargetPointRAS": self._formatRasPoint(trajectory.targetPointRAS),
                    "IntersectsCriticalStructures": bool(intersects),
                    "MinDistanceToCriticalStructuresMm": float(minDistanceMm),
                    "CheckedStructureCount": int(checkedStructureCount),
                }
            )

        trajectoryCount = int(len(trajectories))
        if not criticalStructuresSegmentation:
            statusText = "Critical-structures segmentation is not selected; bundle intersections were not evaluated."
        elif len(preparedStructureSurfaces) <= 0:
            statusText = "No usable critical-structure closed-surface segments were available for bundle evaluation."
        elif trajectoryCount <= 0:
            statusText = "No trajectories were available for bundle evaluation."
        elif intersectingTrajectoryCount <= 0:
            statusText = "Bundle valid: no critical-structure intersections detected."
        else:
            statusText = (
                f"Bundle warning: {intersectingTrajectoryCount}/{trajectoryCount} trajectories intersect critical structures."
            )

        summary: dict[str, Any] = {
            "TrajectoryCount": int(trajectoryCount),
            "IntersectingTrajectoryCount": int(intersectingTrajectoryCount),
            "NonIntersectingTrajectoryCount": int(max(0, trajectoryCount - intersectingTrajectoryCount)),
            "MinDistanceToCriticalStructuresMm": (
                float(min(minDistanceValues))
                if len(minDistanceValues) > 0
                else float("nan")
            ),
            "StatusText": statusText,
            "CriticalStructuresAvailable": bool(criticalStructuresSegmentation is not None),
        }
        return bundleRows, summary

    def computeCoaxialPlanFromTrajectory(
        self,
        trajectory: ProbeTrajectory,
        technique: str,
        activeElementLengthMm: float,
        spareMm: float = 0.0,
    ) -> CoaxialPlanSummary:
        entryPoint = np.array(trajectory.entryPointRAS, dtype=float)
        endpoint = np.array(trajectory.targetPointRAS, dtype=float)
        axisVector = endpoint - entryPoint
        axisNorm = float(np.linalg.norm(axisVector))
        if not math.isfinite(axisNorm) or axisNorm <= 1e-8:
            normalizedDirection = _normalize_vector(trajectory.directionVector)
        else:
            normalizedDirection = axisVector / axisNorm
        techniqueName = str(technique or "PullBack")
        activeElementLengthMm = float(max(0.0, activeElementLengthMm))
        spareMm = float(max(0.0, spareMm))
        pushThroughOffsetMm = float(activeElementLengthMm + spareMm)
        if techniqueName == "PushThrough":
            # Offset from endpoint back toward entry so endpoint-to-navigation distance equals active element + spare.
            navigationTarget = endpoint - (normalizedDirection * pushThroughOffsetMm)
            noteText = (
                f"Push-through: navigate the coaxial needle to the derived target, then advance the applicator "
                f"{pushThroughOffsetMm:.1f} mm "
                f"({activeElementLengthMm:.1f} mm active element + {spareMm:.1f} mm spare)."
            )
        else:
            navigationTarget = endpoint
            noteText = (
                f"Pull-back: navigate the coaxial needle to the locked endpoint, then retract the sheath to expose "
                f"{activeElementLengthMm:.1f} mm of active element."
            )

        return CoaxialPlanSummary(
            technique=techniqueName,
            activeElementLengthMm=float(activeElementLengthMm),
            navigationTargetRAS=tuple(float(value) for value in navigationTarget.tolist()),
            masterEntryPointRAS=tuple(float(value) for value in trajectory.entryPointRAS),
            masterTargetPointRAS=tuple(float(value) for value in trajectory.targetPointRAS),
            spareMm=spareMm,
            pushThroughOffsetMm=pushThroughOffsetMm,
            notes=noteText,
        )

    def createOrUpdateCoaxialLine(
        self,
        entryPointRAS: Sequence[float],
        navigationTargetRAS: Sequence[float],
    ):
        createdLineNodeIDs = self.createOrUpdateCoaxialLines(
            (
                (entryPointRAS, navigationTargetRAS, "Master"),
            )
        )
        if len(createdLineNodeIDs) <= 0:
            return None
        return slicer.mrmlScene.GetNodeByID(createdLineNodeIDs[0])

    def createOrUpdateCoaxialLines(
        self,
        lineSpecifications: Sequence[tuple[Sequence[float], Sequence[float], str]],
    ) -> list[str]:
        self.removeNodesByAttribute("vtkMRMLMarkupsLineNode", GENERATED_COAXIAL_LINE_ATTRIBUTE)
        if len(lineSpecifications) <= 0:
            return []

        createdLineNodeIDs: list[str] = []
        isMultiLinePlan = len(lineSpecifications) > 1
        for lineIndex, lineSpec in enumerate(lineSpecifications):
            lineStartPointRAS, navigationTargetRAS, roleName = lineSpec
            lineNodeName = (
                f"{COAXIAL_NAVIGATION_LINE_NODE_NAME} {lineIndex + 1:02d}"
                if isMultiLinePlan
                else COAXIAL_NAVIGATION_LINE_NODE_NAME
            )
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", lineNodeName)
            slicer.util.updateMarkupsControlPointsFromArray(
                lineNode,
                np.array([lineStartPointRAS, navigationTargetRAS], dtype=float),
            )
            lineNode.SetAttribute(GENERATED_COAXIAL_LINE_ATTRIBUTE, "1")
            lineNode.SetAttribute("SurgicalVision3D_Planner.TrajectoryRole", str(roleName or ""))
            lineDisplayNode = lineNode.GetDisplayNode()
            if lineDisplayNode:
                lineDisplayNode.SetPropertiesLabelVisibility(False)
                lineDisplayNode.SetPointLabelsVisibility(False)
                lineDisplayNode.SetOpacity(1.0)
                if hasattr(lineDisplayNode, "SetLineThickness"):
                    lineDisplayNode.SetLineThickness(0.8 * LINE_THICKNESS_SCALE)
                elif hasattr(lineDisplayNode, "SetLineWidth"):
                    lineDisplayNode.SetLineWidth(3.0 * LINE_THICKNESS_SCALE)
            createdLineNodeIDs.append(lineNode.GetID())
        return createdLineNodeIDs

    @staticmethod
    @contextmanager
    def _mrmlBatchState(*stateNames: str):
        scene = slicer.mrmlScene
        startedStateIds: list[int] = []
        try:
            for stateName in stateNames:
                stateId = getattr(scene, stateName, None)
                if stateId is None:
                    continue
                try:
                    scene.StartState(int(stateId))
                    startedStateIds.append(int(stateId))
                except Exception:
                    pass
            yield
        finally:
            for stateId in reversed(startedStateIds):
                try:
                    scene.EndState(int(stateId))
                except Exception:
                    pass

    def removeNodesByAttribute(self, className: str, attributeName: str, attributeValue: str = "1", keepNodeID: str | None = None) -> None:
        nodesToRemove = [
            node
            for node in slicer.util.getNodesByClass(className)
            if not (keepNodeID and node.GetID() == keepNodeID)
            and node.GetAttribute(attributeName) == attributeValue
        ]
        if len(nodesToRemove) <= 0:
            return
        useBatchState = str(className or "") == "vtkMRMLSegmentationNode"
        stateContext = self._mrmlBatchState("BatchProcessState") if useBatchState else nullcontext()
        with stateContext:
            for node in nodesToRemove:
                if node and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)

    def removeGeneratedProbeNodes(self, keepNodeID: str | None = None) -> None:
        self.removeNodesByAttribute("vtkMRMLSegmentationNode", GENERATED_PROBE_ATTRIBUTE, keepNodeID=keepNodeID)

    def removeGeneratedTrajectoryLines(self) -> None:
        self.removeNodesByAttribute("vtkMRMLMarkupsLineNode", GENERATED_TRAJECTORY_LINE_ATTRIBUTE)

    def removeNodesByIDs(self, nodeIDs: Sequence[str]) -> None:
        if len(nodeIDs) <= 0:
            return
        useBatchState = False
        for nodeID in nodeIDs:
            node = slicer.mrmlScene.GetNodeByID(nodeID)
            if node and node.IsA("vtkMRMLSegmentationNode"):
                useBatchState = True
                break
        stateContext = self._mrmlBatchState("BatchProcessState") if useBatchState else nullcontext()
        with stateContext:
            for nodeID in nodeIDs:
                node = slicer.mrmlScene.GetNodeByID(nodeID)
                if node and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)

    def removeNodeIfOwned(self, node: vtk.vtkObject | None, ownershipAttribute: str, ownershipValue: str = "1") -> bool:
        if not node:
            return False
        if not slicer.mrmlScene.IsNodePresent(node):
            return False
        if node.GetAttribute(ownershipAttribute) != ownershipValue:
            return False
        useBatchState = bool(hasattr(node, "IsA") and node.IsA("vtkMRMLSegmentationNode"))
        stateContext = self._mrmlBatchState("BatchProcessState") if useBatchState else nullcontext()
        with stateContext:
            if not slicer.mrmlScene.IsNodePresent(node):
                return False
            slicer.mrmlScene.RemoveNode(node)
            return True

    def createOrReuseOwnedOutputNode(
        self,
        className: str,
        preferredName: str,
        ownershipAttribute: str,
        existingNode: vtk.vtkObject | None = None,
    ):
        outputNode = existingNode if existingNode and slicer.mrmlScene.IsNodePresent(existingNode) else None
        if outputNode and outputNode.GetAttribute(ownershipAttribute) != "1":
            outputNode = None

        if outputNode is None:
            outputNode = slicer.mrmlScene.AddNewNodeByClass(className, preferredName)

        outputNode.SetName(preferredName)
        outputNode.SetAttribute(ownershipAttribute, "1")
        if className == "vtkMRMLSegmentationNode" and hasattr(outputNode, "GetSegmentation"):
            self._preferClosedSurfaceSourceRepresentation(outputNode)
        self.removeNodesByAttribute(className, ownershipAttribute, keepNodeID=outputNode.GetID())
        return outputNode

    @staticmethod
    def _preferClosedSurfaceSourceRepresentation(segmentationNode: vtkMRMLSegmentationNode | None) -> None:
        if not segmentationNode:
            return
        segmentation = segmentationNode.GetSegmentation()
        if not segmentation:
            return
        try:
            if hasattr(segmentation, "SetSourceRepresentationName"):
                segmentation.SetSourceRepresentationName("Closed surface")
            elif hasattr(segmentation, "SetMasterRepresentationName"):
                segmentation.SetMasterRepresentationName("Closed surface")
        except Exception:
            pass

    @staticmethod
    def _definedMarkupsPointCount(markupsNode: vtkMRMLMarkupsFiducialNode | None) -> int:
        if not markupsNode:
            return 0
        if hasattr(markupsNode, "GetNumberOfDefinedControlPoints"):
            return int(markupsNode.GetNumberOfDefinedControlPoints())
        return int(markupsNode.GetNumberOfControlPoints())

    @staticmethod
    def primaryScalarVolumeNode():
        volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if len(volumeNodes) == 0:
            return None

        def voxelCount(volumeNode) -> int:
            if not volumeNode or not hasattr(volumeNode, "GetImageData"):
                return 0
            imageData = volumeNode.GetImageData()
            if imageData is None:
                return 0
            dimensions = imageData.GetDimensions()
            if dimensions is None or len(dimensions) < 3:
                return 0
            return int(dimensions[0]) * int(dimensions[1]) * int(dimensions[2])

        return max(
            volumeNodes,
            key=lambda volumeNode: (
                voxelCount(volumeNode),
                str(volumeNode.GetName() or "").lower(),
            ),
        )

    def applyCtAbdomenDisplayToVolume(self, volumeNode) -> bool:
        if not volumeNode:
            return False
        if hasattr(volumeNode, "CreateDefaultDisplayNodes"):
            volumeNode.CreateDefaultDisplayNodes()
        if not hasattr(volumeNode, "GetDisplayNode"):
            return False

        displayNode = volumeNode.GetDisplayNode()
        if not displayNode:
            return False

        if hasattr(displayNode, "SetAutoWindowLevel"):
            displayNode.SetAutoWindowLevel(False)
        elif hasattr(displayNode, "AutoWindowLevelOff"):
            displayNode.AutoWindowLevelOff()

        if hasattr(displayNode, "SetWindowLevel"):
            displayNode.SetWindowLevel(float(DEFAULT_CT_ABDOMEN_WINDOW), float(DEFAULT_CT_ABDOMEN_LEVEL))
        else:
            if hasattr(displayNode, "SetWindow"):
                displayNode.SetWindow(float(DEFAULT_CT_ABDOMEN_WINDOW))
            if hasattr(displayNode, "SetLevel"):
                displayNode.SetLevel(float(DEFAULT_CT_ABDOMEN_LEVEL))

        displayNode.Modified()
        volumeNode.Modified()
        return True

    def _ensureSegmentationReferenceImageGeometry(self, segmentationNode: vtkMRMLSegmentationNode | None) -> None:
        if not segmentationNode:
            return
        segmentation = segmentationNode.GetSegmentation()
        if not segmentation:
            return

        referenceParameterName = ""
        if hasattr(slicer, "vtkSegmentationConverter") and hasattr(slicer.vtkSegmentationConverter, "GetReferenceImageGeometryParameterName"):
            referenceParameterName = str(slicer.vtkSegmentationConverter.GetReferenceImageGeometryParameterName() or "")
        existingReferenceGeometry = ""
        if referenceParameterName and hasattr(segmentation, "GetConversionParameter"):
            existingReferenceGeometry = str(segmentation.GetConversionParameter(referenceParameterName) or "")
        if existingReferenceGeometry.strip():
            return

        referenceVolumeNode = self.primaryScalarVolumeNode()
        if referenceVolumeNode and hasattr(segmentationNode, "SetReferenceImageGeometryParameterFromVolumeNode"):
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolumeNode)

    def prepareSegmentationForEditing(self, segmentationNode: vtkMRMLSegmentationNode | None) -> None:
        if not segmentationNode or not slicer.mrmlScene.IsNodePresent(segmentationNode):
            return
        segmentation = segmentationNode.GetSegmentation()
        if not segmentation:
            return

        self._ensureSegmentationReferenceImageGeometry(segmentationNode)
        try:
            segmentation.CreateRepresentation("Binary labelmap")
        except Exception:
            pass
        try:
            if hasattr(segmentation, "SetSourceRepresentationName"):
                segmentation.SetSourceRepresentationName("Binary labelmap")
            elif hasattr(segmentation, "SetMasterRepresentationName"):
                segmentation.SetMasterRepresentationName("Binary labelmap")
        except Exception:
            # Keep non-fatal: this call should never block UI flow even if representation backends differ.
            logging.debug("Could not force Binary labelmap source representation for '%s'.", segmentationNode.GetName())

    def mergeProbeInstances(
        self,
        generatedProbeNodeIDs: Sequence[str],
        outputSegmentation: vtkMRMLSegmentationNode | None = None,
    ) -> vtkMRMLSegmentationNode:
        validProbeNodes: list[vtkMRMLSegmentationNode] = []
        for nodeID in generatedProbeNodeIDs:
            probeNode = slicer.mrmlScene.GetNodeByID(nodeID)
            if probeNode and probeNode.IsA("vtkMRMLSegmentationNode"):
                validProbeNodes.append(probeNode)

        if len(validProbeNodes) == 0:
            raise ValueError("No translated probe segmentations were found to merge.")

        combinedSegmentation = self.createOrReuseOwnedOutputNode(
            "vtkMRMLSegmentationNode",
            COMBINED_PROBE_NODE_NAME,
            GENERATED_COMBINED_PROBE_ATTRIBUTE,
            outputSegmentation,
        )

        combinedSegmentation.CreateDefaultDisplayNodes()
        self._ensureSegmentationReferenceImageGeometry(combinedSegmentation)
        self._clearSegmentationSegments(combinedSegmentation)
        probeSurfaces: list[vtk.vtkPolyData] = []
        for translatedProbeNode in validProbeNodes:
            self._ensureSegmentationHasClosedSurface(translatedProbeNode)
            sourceSegmentID = self.getWorkingSegmentID(
                translatedProbeNode,
                "probe merge input",
            )
            closedSurface = vtk.vtkPolyData()
            translatedProbeNode.GetClosedSurfaceRepresentation(sourceSegmentID, closedSurface)
            if (
                closedSurface.GetNumberOfPoints() <= 0
                or closedSurface.GetNumberOfCells() <= 0
                or not self._polyDataHasFinitePoints(closedSurface)
            ):
                continue
            surfaceCopy = vtk.vtkPolyData()
            surfaceCopy.DeepCopy(closedSurface)
            probeSurfaces.append(surfaceCopy)

        if len(probeSurfaces) <= 0:
            raise RuntimeError("Unable to build combined ablation segmentation from generated probes.")

        if USE_SEGMENT_EDITOR_UNION_FOR_PROBE_MERGE:
            for surfaceIndex, surface in enumerate(probeSurfaces):
                combinedSegmentation.AddSegmentFromClosedSurfaceRepresentation(
                    surface,
                    f"Probe_{surfaceIndex + 1:02d}",
                    [1.0, 0.3, 0.1],
                )
            try:
                self._unionSegmentsWithLogicalOperators(combinedSegmentation)
            except Exception:
                logging.exception(
                    "Segment Editor union probe merge failed; falling back to closed-surface append merge."
                )
                self._mergeSegmentsByAppendingSurfaces(combinedSegmentation)
        else:
            appendFilter = vtk.vtkAppendPolyData()
            for surface in probeSurfaces:
                appendFilter.AddInputData(surface)
            appendFilter.Update()

            cleanFilter = vtk.vtkCleanPolyData()
            cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
            cleanFilter.Update()

            mergedSurface = vtk.vtkPolyData()
            mergedSurface.DeepCopy(cleanFilter.GetOutput())
            if (
                mergedSurface.GetNumberOfPoints() <= 0
                or mergedSurface.GetNumberOfCells() <= 0
                or not self._polyDataHasFinitePoints(mergedSurface)
            ):
                raise RuntimeError("Probe merge produced an invalid combined closed surface.")

            combinedSegmentation.AddSegmentFromClosedSurfaceRepresentation(
                mergedSurface,
                "CombinedAblationZone",
                [1.0, 0.3, 0.1],
            )

        if combinedSegmentation.GetSegmentation().GetNumberOfSegments() > 1:
            self._mergeSegmentsByAppendingSurfaces(combinedSegmentation)

        if combinedSegmentation.GetSegmentation().GetNumberOfSegments() == 1:
            combinedSegmentID = combinedSegmentation.GetSegmentation().GetNthSegmentID(0)
            combinedSegment = (
                combinedSegmentation.GetSegmentation().GetSegment(combinedSegmentID)
                if combinedSegmentID
                else None
            )
            if combinedSegment:
                combinedSegment.SetName("CombinedAblationZone")

        combinedSegmentation.GetSegmentation().CreateRepresentation("Closed surface")
        if combinedSegmentation.GetSegmentation().GetNumberOfSegments() != 1:
            raise RuntimeError("Probe merge did not produce a single deterministic combined segment.")

        combinedSegmentation.SetName(COMBINED_PROBE_NODE_NAME)
        displayNode = combinedSegmentation.GetDisplayNode()
        if displayNode:
            displayNode.SetOpacity(0.35)
            displayNode.SetVisibility(True)
        return combinedSegmentation

    def registerTumorToFiducials(
        self,
        tumorSegmentation: vtkMRMLSegmentationNode | None,
        nativeFiducials: vtkMRMLMarkupsFiducialNode | None,
        registeredFiducials: vtkMRMLMarkupsFiducialNode | None,
        outputTransformNode: vtkMRMLTransformNode | None = None,
    ) -> vtkMRMLTransformNode:
        if not tumorSegmentation:
            raise ValueError("Tumor segmentation node is required.")
        if not nativeFiducials or not registeredFiducials:
            raise ValueError("Both native and registered fiducial markups are required.")
        if not hasattr(slicer.modules, "fiducialregistration"):
            raise RuntimeError("fiducialregistration CLI module is not available.")

        nativeCount = self._definedMarkupsPointCount(nativeFiducials)
        registeredCount = self._definedMarkupsPointCount(registeredFiducials)
        if nativeCount != registeredCount:
            raise ValueError(
                f"Native and registered fiducial counts must match. Found {nativeCount} native and {registeredCount} registered points."
            )
        if nativeCount < 3:
            raise ValueError(
                f"Rigid fiducial registration requires at least 3 points. Found {nativeCount}."
            )

        transformNode = outputTransformNode
        if not transformNode or not slicer.mrmlScene.IsNodePresent(transformNode):
            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "TumorRegistrationTransform")

        registrationParameters = {
            "fixedLandmarks": registeredFiducials.GetID(),
            "movingLandmarks": nativeFiducials.GetID(),
            "saveTransform": transformNode.GetID(),
            "transformType": "Rigid",
        }
        cliNode = slicer.cli.runSync(slicer.modules.fiducialregistration, None, registrationParameters)
        if cliNode and hasattr(cliNode, "GetStatusString"):
            statusText = str(cliNode.GetStatusString() or "")
            if statusText and ("error" in statusText.lower()):
                errorText = str(cliNode.GetErrorText() or "") if hasattr(cliNode, "GetErrorText") else ""
                raise RuntimeError(errorText or f"Fiducial registration failed: {statusText}")
        if cliNode:
            slicer.mrmlScene.RemoveNode(cliNode)

        tumorSegmentation.SetAndObserveTransformNodeID(transformNode.GetID())
        return transformNode

    def hardenTumorTransform(self, tumorSegmentation: vtkMRMLSegmentationNode | None) -> None:
        if not tumorSegmentation:
            raise ValueError("Tumor segmentation node is required.")
        if tumorSegmentation.GetTransformNodeID():
            tumorSegmentation.HardenTransform()

    def evaluateMargins(
        self,
        tumorSegmentation: vtkMRMLSegmentationNode | None,
        probeSegmentation: vtkMRMLSegmentationNode | None,
        outputMarginModel: vtkMRMLModelNode | None = None,
        outputTableNode: vtkMRMLTableNode | None = None,
    ) -> tuple[vtkMRMLModelNode, vtkMRMLTableNode, dict[str, float]]:
        if not tumorSegmentation:
            raise ValueError("Tumor segmentation node is required.")
        if not probeSegmentation:
            raise ValueError("Combined probe segmentation node is required.")

        tempProbeModel = self.createOrReuseOwnedOutputNode(
            "vtkMRMLModelNode",
            TEMP_PROBE_MODEL_NODE_NAME,
            TEMP_PROBE_MARGIN_INPUT_ATTRIBUTE,
        )
        tempTumorModel = self.createOrReuseOwnedOutputNode(
            "vtkMRMLModelNode",
            TEMP_TUMOR_MODEL_NODE_NAME,
            TEMP_TUMOR_MARGIN_INPUT_ATTRIBUTE,
        )

        probeModel = self.segmentationFirstSegmentToModel(
            probeSegmentation,
            TEMP_PROBE_MODEL_NODE_NAME,
            outputModelNode=tempProbeModel,
        )
        tumorSegmentID = self.findPreferredSegmentID(
            tumorSegmentation,
            DEFAULT_TUMOR_SEGMENT_NAMES,
            "margin evaluation",
            fallbackToFirst=True,
        )
        tumorModel = self.segmentationSegmentToModel(
            tumorSegmentation,
            tumorSegmentID,
            TEMP_TUMOR_MODEL_NODE_NAME,
            outputModelNode=tempTumorModel,
        )

        marginModel = self.createOrReuseOwnedOutputNode(
            "vtkMRMLModelNode",
            MARGIN_MODEL_NODE_NAME,
            GENERATED_MARGIN_MODEL_ATTRIBUTE,
            outputMarginModel,
        )
        marginModel.CreateDefaultDisplayNodes()

        if ENABLE_MAM_DEBUG_LOGGING:
            logging.info(
                "MAM debug: evaluateMargins geometry | tumorModel: %s | probeModel: %s",
                self._modelPolyDataSummaryText(tumorModel),
                self._modelPolyDataSummaryText(probeModel),
            )

        try:
            self.computeSignedDistanceModel(tumorModel, probeModel, marginModel)

            signedDistanceArray = self.getSignedDistanceArray(marginModel)
            self.backupSignedDistanceArray(marginModel, signedDistanceArray)
            self.configureMarginDisplayNode(marginModel, autoRange=True)

            resultTable = self.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                MARGIN_TABLE_NODE_NAME,
                GENERATED_RESULT_TABLE_ATTRIBUTE,
                outputTableNode,
            )
            self.populateResultTableFromMarginModel(marginModel, resultTable)

            summary = self.signedDistanceSummary(signedDistanceArray)
            return marginModel, resultTable, summary
        finally:
            self.removeNodeIfOwned(tempProbeModel, TEMP_PROBE_MARGIN_INPUT_ATTRIBUTE)
            self.removeNodeIfOwned(tempTumorModel, TEMP_TUMOR_MARGIN_INPUT_ATTRIBUTE)

    def evaluateStructureSafety(
        self,
        riskStructuresSegmentation: vtkMRMLSegmentationNode | None,
        probeSegmentation: vtkMRMLSegmentationNode | None,
    ) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
        if riskStructuresSegmentation is None:
            return [], []
        if not probeSegmentation:
            raise ValueError("Combined probe segmentation node is required for structure safety evaluation.")

        riskSegments = self.getValidSegmentationSegments(
            riskStructuresSegmentation,
            "structure safety evaluation",
        )
        if ENABLE_MAM_DEBUG_LOGGING:
            logging.info(
                "MAM debug: evaluateStructureSafety start | riskSegmentation='%s' | segmentCount=%d",
                str(riskStructuresSegmentation.GetName() or "(unnamed)"),
                len(riskSegments),
            )
        tempProbeModel = self.createOrReuseOwnedOutputNode(
            "vtkMRMLModelNode",
            TEMP_PROBE_SAFETY_MODEL_NODE_NAME,
            TEMP_PROBE_SAFETY_INPUT_ATTRIBUTE,
        )
        tempStructureModel = self.createOrReuseOwnedOutputNode(
            "vtkMRMLModelNode",
            TEMP_STRUCTURE_SAFETY_MODEL_NODE_NAME,
            TEMP_STRUCTURE_SAFETY_INPUT_ATTRIBUTE,
        )
        tempDistanceModel = self.createOrReuseOwnedOutputNode(
            "vtkMRMLModelNode",
            TEMP_STRUCTURE_SAFETY_DISTANCE_MODEL_NODE_NAME,
            TEMP_STRUCTURE_SAFETY_DISTANCE_OUTPUT_ATTRIBUTE,
        )

        probeModel = self.segmentationFirstSegmentToModel(
            probeSegmentation,
            TEMP_PROBE_SAFETY_MODEL_NODE_NAME,
            outputModelNode=tempProbeModel,
        )
        preparedProbeTargetPolyData = self._preparePolyDataForDistanceComputation(
            self._requireModelPolyData(probeModel, "Target"),
            "Target",
            orientNormals=True,
        )

        structureSafetySummaryRows: list[dict[str, float | int | str]] = []
        structureSafetyThresholdRows: list[dict[str, float | int | str]] = []
        try:
            for segmentInfo in riskSegments:
                segmentID = str(segmentInfo["segmentID"])
                segmentName = str(segmentInfo["segmentName"])
                try:
                    self.segmentationSegmentToModel(
                        riskStructuresSegmentation,
                        segmentID,
                        TEMP_STRUCTURE_SAFETY_MODEL_NODE_NAME,
                        outputModelNode=tempStructureModel,
                    )
                    if ENABLE_MAM_DEBUG_LOGGING:
                        logging.info(
                            "MAM debug: structure segment '%s' (%s) geometry | %s",
                            segmentName,
                            segmentID,
                            self._modelPolyDataSummaryText(tempStructureModel),
                        )

                    # Negative values indicate structure points inside/overlapping ablation geometry.
                    self.computeSignedDistanceModel(
                        tempStructureModel,
                        probeModel,
                        tempDistanceModel,
                        preparedTargetPolyData=preparedProbeTargetPolyData,
                    )

                    signedDistanceValues = self.getSignedMarginValuesArray(tempDistanceModel)
                    distanceSummary = self.computeDistanceSummary(signedDistanceValues)
                    thresholdSummary = self.computeDistanceThresholdSummary(signedDistanceValues)
                except Exception:
                    logging.exception(
                        "Skipping structure-safety segment '%s' (%s) due to invalid geometry.",
                        segmentName,
                        segmentID,
                    )
                    continue

                structureSafetySummaryRows.append(
                    {
                        "StructureSegmentID": segmentID,
                        "StructureName": segmentName,
                        **distanceSummary,
                    }
                )
                structureSafetyThresholdRows.append(
                    {
                        "StructureSegmentID": segmentID,
                        "StructureName": segmentName,
                        **thresholdSummary,
                    }
                )
        finally:
            self.removeNodeIfOwned(tempProbeModel, TEMP_PROBE_SAFETY_INPUT_ATTRIBUTE)
            self.removeNodeIfOwned(tempStructureModel, TEMP_STRUCTURE_SAFETY_INPUT_ATTRIBUTE)
            self.removeNodeIfOwned(tempDistanceModel, TEMP_STRUCTURE_SAFETY_DISTANCE_OUTPUT_ATTRIBUTE)

        return structureSafetySummaryRows, structureSafetyThresholdRows

    def recolorMarginModel(self, marginModelNode: vtkMRMLModelNode | None, thresholds: Sequence[float]) -> None:
        if not marginModelNode:
            raise ValueError("Margin model node is required.")

        signedDistanceArray = self.getSignedDistanceArray(marginModelNode)
        backupArray = self.getSignedDistanceBackupArray(marginModelNode)
        if backupArray is None:
            self.backupSignedDistanceArray(marginModelNode, signedDistanceArray)

        thresholdCount = self.recolorSignedDistanceArray(signedDistanceArray, thresholds)
        self.configureMarginDisplayNode(marginModelNode, autoRange=False, scalarRange=(0.0, float(thresholdCount - 1)))
        self.refreshNodeDisplay(marginModelNode)

    def resetMarginModelColors(self, marginModelNode: vtkMRMLModelNode | None) -> None:
        if not marginModelNode:
            raise ValueError("Margin model node is required.")

        signedDistanceArray = self.getSignedDistanceArray(marginModelNode)
        signedDistanceBackup = self.getSignedDistanceBackupArray(marginModelNode)
        if signedDistanceBackup is None:
            raise RuntimeError("No signed-distance backup is available. Evaluate margins before resetting colors.")

        self.restoreSignedDistanceArray(signedDistanceArray, signedDistanceBackup)
        self.configureMarginDisplayNode(marginModelNode, autoRange=True)
        self.refreshNodeDisplay(marginModelNode)

    def ensureBeginnerMamColorNode(self):
        for colorNode in slicer.util.getNodesByClass("vtkMRMLColorTableNode"):
            if str(colorNode.GetName() or "").strip() == BEGINNER_MAM_COLOR_NODE_NAME:
                return colorNode

        colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", BEGINNER_MAM_COLOR_NODE_NAME)
        colorNode.SetAttribute(GENERATED_MARGIN_MODEL_ATTRIBUTE, "1")
        colorNode.SetTypeToUser()
        colorNode.SetNumberOfColors(3)
        colorNode.SetColor(0, "BelowHalfMAM", 0.84, 0.15, 0.16, 1.0)
        colorNode.SetColor(1, "BetweenHalfMAMAndMAM", 0.95, 0.55, 0.18, 1.0)
        colorNode.SetColor(2, "AtLeastMAM", 0.18, 0.62, 0.31, 1.0)
        return colorNode

    @staticmethod
    def computeMamAssessmentSummary(signedMarginValues: Sequence[float], mamMm: float) -> dict[str, float | int | bool | str]:
        if len(signedMarginValues) == 0:
            raise ValueError("MAM assessment cannot be computed because no signed margin values are available.")

        achievedMargins = -np.asarray(signedMarginValues, dtype=float)
        achievedMargins = achievedMargins[np.isfinite(achievedMargins)]
        if achievedMargins.size == 0:
            raise ValueError("MAM assessment cannot be computed because signed margin values are invalid.")

        halfMamMm = float(max(0.0, mamMm) * 0.5)
        redCount = int(np.count_nonzero(achievedMargins < halfMamMm))
        orangeCount = int(np.count_nonzero((achievedMargins >= halfMamMm) & (achievedMargins < float(mamMm))))
        greenCount = int(np.count_nonzero(achievedMargins >= float(mamMm)))
        mamPass = bool(float(np.min(achievedMargins)) >= float(mamMm))
        statusText = (
            f"MAM satisfied: all margins are at least {float(mamMm):.1f} mm."
            if mamPass
            else f"MAM not satisfied: minimum achieved margin is {float(np.min(achievedMargins)):.2f} mm."
        )
        return {
            "MamMm": float(mamMm),
            "MamPass": mamPass,
            "MinAchievedMarginMm": float(np.min(achievedMargins)),
            "MeanAchievedMarginMm": float(np.mean(achievedMargins)),
            "MaxAchievedMarginMm": float(np.max(achievedMargins)),
            "CountRed": redCount,
            "CountOrange": orangeCount,
            "CountGreen": greenCount,
            "StatusText": statusText,
        }

    def applyBeginnerMamColoring(self, marginModelNode: vtkMRMLModelNode | None, mamMm: float) -> dict[str, float | int | bool | str]:
        if not marginModelNode:
            raise ValueError("Margin model node is required.")
        if mamMm < 0.0:
            raise ValueError("MAM must be non-negative.")

        signedDistanceArray = self.getSignedDistanceArray(marginModelNode)
        backupArray = self.getSignedDistanceBackupArray(marginModelNode)
        if backupArray is None:
            self.backupSignedDistanceArray(marginModelNode, signedDistanceArray)
            backupArray = self.getSignedDistanceBackupArray(marginModelNode)
        if backupArray is None:
            raise RuntimeError("Signed-distance backup array is unavailable for MAM coloring.")

        backupValueCount = _data_array_value_count(backupArray)
        signedDistanceValueCount = _data_array_value_count(signedDistanceArray)
        if backupValueCount != signedDistanceValueCount:
            raise RuntimeError(
                "Signed-distance array length mismatch during MAM coloring "
                f"(signed={signedDistanceValueCount}, backup={backupValueCount})."
            )

        backupValues = np.asarray(
            numpy_support.vtk_to_numpy(backupArray),
            dtype=np.float64,
        )
        achievedMargins = -backupValues
        validMask = np.isfinite(achievedMargins)
        if ENABLE_MAM_DEBUG_LOGGING:
            logging.info(
                "MAM debug: coloring arrays | valueCount=%d | finiteCount=%d | nonFiniteCount=%d",
                int(backupValues.size),
                int(np.count_nonzero(validMask)),
                int(np.count_nonzero(~validMask)),
            )

        targetValues = numpy_support.vtk_to_numpy(signedDistanceArray)
        bucketValues = np.zeros(backupValues.shape[0], dtype=np.float64)
        halfMamMm = float(mamMm) * 0.5
        bucketValues[validMask & (achievedMargins >= halfMamMm)] = 1.0
        bucketValues[validMask & (achievedMargins >= float(mamMm))] = 2.0
        targetValues[:] = bucketValues.astype(targetValues.dtype, copy=False)
        signedDistanceArray.Modified()

        colorNode = self.ensureBeginnerMamColorNode()
        self.configureMarginDisplayNode(marginModelNode, autoRange=False, scalarRange=(0.0, 2.0))
        displayNode = marginModelNode.GetDisplayNode()
        if displayNode:
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())
        self.refreshNodeDisplay(marginModelNode)
        return self.computeMamAssessmentSummary(backupValues, mamMm)

    @staticmethod
    def recolorSignedDistanceArray(signedDistanceArray: vtk.vtkDataArray, thresholds: Sequence[float]) -> int:
        if signedDistanceArray is None:
            raise ValueError("Signed distance array is required.")

        sortedThresholds = sorted(float(value) for value in thresholds)
        if len(sortedThresholds) == 0:
            raise ValueError("At least one threshold value is required.")

        thresholdArray = np.asarray(sortedThresholds, dtype=np.float64)
        targetValues = numpy_support.vtk_to_numpy(signedDistanceArray)
        sourceValues = np.asarray(targetValues, dtype=np.float64)
        bucketIndices = np.searchsorted(thresholdArray, sourceValues, side="right")
        targetValues[:] = bucketIndices.astype(targetValues.dtype, copy=False)
        signedDistanceArray.Modified()

        return len(sortedThresholds) + 1

    @staticmethod
    def restoreSignedDistanceArray(targetArray: vtk.vtkDataArray, sourceArray: vtk.vtkDataArray) -> None:
        if targetArray is None or sourceArray is None:
            raise ValueError("Both target and source arrays are required.")

        targetValueCount = _data_array_value_count(targetArray)
        sourceValueCount = _data_array_value_count(sourceArray)
        if targetValueCount != sourceValueCount:
            raise RuntimeError(
                f"Signed-distance arrays must have same length. target={targetValueCount}, source={sourceValueCount}"
            )

        targetValues = numpy_support.vtk_to_numpy(targetArray)
        sourceValues = np.asarray(numpy_support.vtk_to_numpy(sourceArray), dtype=targetValues.dtype)
        np.copyto(targetValues, sourceValues, casting="unsafe")
        targetArray.Modified()

    def populateResultTableFromMarginModel(self, marginModelNode: vtkMRMLModelNode, tableNode: vtkMRMLTableNode) -> None:
        if not marginModelNode or not tableNode:
            raise ValueError("Margin model and output table nodes are required.")

        tableNode.RemoveAllColumns()
        fieldData = self.getModelFieldData(marginModelNode)

        for arrayIndex in range(fieldData.GetNumberOfArrays()):
            sourceArray = fieldData.GetArray(arrayIndex)
            if sourceArray is None:
                continue
            copiedArray = sourceArray.NewInstance()
            copiedArray.DeepCopy(sourceArray)
            copiedArray.SetName(sourceArray.GetName())
            tableNode.AddColumn(copiedArray)

    def getSignedMarginValuesArray(self, marginModelNode: vtkMRMLModelNode | None) -> np.ndarray:
        if not marginModelNode:
            raise ValueError("Margin model node is required.")

        signedDistanceArray = self.getSignedDistanceBackupArray(marginModelNode)
        if signedDistanceArray is None:
            signedDistanceArray = self.getSignedDistanceArray(marginModelNode)

        signedMarginValuesArray = np.asarray(
            numpy_support.vtk_to_numpy(signedDistanceArray),
            dtype=np.float64,
        )
        finiteMask = np.isfinite(signedMarginValuesArray)
        invalidValueCount = int(np.count_nonzero(~finiteMask))
        if invalidValueCount > 0:
            logging.warning("Ignored %d non-finite signed-margin values during summary computation.", invalidValueCount)

        finiteValues = signedMarginValuesArray[finiteMask]
        if finiteValues.size <= 0:
            raise RuntimeError("Signed margin model contains no finite scalar values for summary computation.")
        return finiteValues

    def getSignedMarginValues(self, marginModelNode: vtkMRMLModelNode | None) -> list[float]:
        return self.getSignedMarginValuesArray(marginModelNode).astype(float).tolist()

    def getWorkingSegmentInfo(self, segmentationNode: vtkMRMLSegmentationNode | None, operationName: str) -> tuple[str, str]:
        segmentID = self.getWorkingSegmentID(segmentationNode, operationName)
        segment = segmentationNode.GetSegmentation().GetSegment(segmentID) if segmentationNode else None
        segmentName = segment.GetName() if segment and segment.GetName() else segmentID
        return segmentID, segmentName

    def populateTrajectorySummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        trajectoryMetrics: Sequence[dict[str, float | int | str]],
    ) -> None:
        if not tableNode:
            raise ValueError("Trajectory summary table node is required.")

        tableNode.RemoveAllColumns()
        self._addNumericColumn(
            tableNode,
            "Trajectory Index",
            [int(metric.get("TrajectoryIndex", 0)) for metric in trajectoryMetrics],
            integer=True,
        )
        self._addStringColumn(tableNode, "Role", [str(metric.get("Role", "")) for metric in trajectoryMetrics])
        self._addStringColumn(tableNode, "Source", [str(metric.get("Source", "")) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Entry R (mm)", [float(metric.get("EntryR", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Entry A (mm)", [float(metric.get("EntryA", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Entry S (mm)", [float(metric.get("EntryS", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Target R (mm)", [float(metric.get("TargetR", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Target A (mm)", [float(metric.get("TargetA", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Target S (mm)", [float(metric.get("TargetS", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Direction R", [float(metric.get("DirR", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Direction A", [float(metric.get("DirA", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Direction S", [float(metric.get("DirS", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Length (mm)", [float(metric.get("LengthMm", 0.0)) for metric in trajectoryMetrics])
        self._addNumericColumn(tableNode, "Angle (deg)", [float(metric.get("AngleDeg", float("nan"))) for metric in trajectoryMetrics])
        self._addNumericColumn(
            tableNode,
            "Radial Offset (mm)",
            [float(metric.get("RadialOffsetMm", 0.0)) for metric in trajectoryMetrics],
        )

    def populateMasterTrajectoryValidationTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        validationRows: Sequence[dict[str, float | int | bool | str]],
    ) -> None:
        if not tableNode:
            raise ValueError("Master trajectory validation table node is required.")

        orderedRows = list(validationRows)
        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "Structure Segment ID", [str(row.get("StructureSegmentID", "")) for row in orderedRows])
        self._addStringColumn(tableNode, "Structure Name", [str(row.get("StructureName", "")) for row in orderedRows])
        self._addStringColumn(
            tableNode,
            "Trajectory Intersects",
            ["Yes" if bool(row.get("TrajectoryIntersects", False)) else "No" for row in orderedRows],
        )
        self._addNumericColumn(tableNode, "Min Distance (mm)", [float(row.get("MinDistanceMm", float("nan"))) for row in orderedRows])

    def populateDerivedTrajectoryBundleSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        bundleRows: Sequence[dict[str, Any]],
    ) -> None:
        if not tableNode:
            raise ValueError("Derived trajectory bundle summary table node is required.")

        orderedRows = list(bundleRows)
        tableNode.RemoveAllColumns()
        self._addNumericColumn(
            tableNode,
            "Trajectory Index",
            [int(row.get("TrajectoryIndex", 0)) for row in orderedRows],
            integer=True,
        )
        self._addStringColumn(tableNode, "Role", [str(row.get("Role", "")) for row in orderedRows])
        self._addStringColumn(tableNode, "Source", [str(row.get("Source", "")) for row in orderedRows])
        self._addNumericColumn(tableNode, "Angle (deg)", [float(row.get("AngleDeg", float("nan"))) for row in orderedRows])
        self._addNumericColumn(tableNode, "Radial Offset (mm)", [float(row.get("RadialOffsetMm", 0.0)) for row in orderedRows])
        self._addStringColumn(tableNode, "Entry Point RAS", [str(row.get("EntryPointRAS", "")) for row in orderedRows])
        self._addStringColumn(tableNode, "Target Point RAS", [str(row.get("TargetPointRAS", "")) for row in orderedRows])
        self._addStringColumn(
            tableNode,
            "Intersects Critical Structures",
            ["Yes" if bool(row.get("IntersectsCriticalStructures", False)) else "No" for row in orderedRows],
        )
        self._addNumericColumn(
            tableNode,
            "Min Distance To Critical Structures (mm)",
            [float(row.get("MinDistanceToCriticalStructuresMm", float("nan"))) for row in orderedRows],
        )

    def populateCoaxialPlanTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        coaxialRows: Sequence[dict[str, Any]],
    ) -> None:
        if not tableNode:
            raise ValueError("Coaxial plan table node is required.")

        orderedRows = list(coaxialRows)
        tableNode.RemoveAllColumns()
        self._addNumericColumn(
            tableNode,
            "Trajectory Index",
            [int(row.get("TrajectoryIndex", 0)) for row in orderedRows],
            integer=True,
        )
        self._addStringColumn(tableNode, "Role", [str(row.get("Role", "")) for row in orderedRows])
        self._addStringColumn(tableNode, "Source", [str(row.get("Source", "")) for row in orderedRows])
        self._addStringColumn(tableNode, "Technique", [str(row.get("Technique", "")) for row in orderedRows])
        self._addNumericColumn(
            tableNode,
            "Active Element (mm)",
            [float(row.get("ActiveElementLengthMm", 0.0)) for row in orderedRows],
        )
        self._addNumericColumn(
            tableNode,
            "Spare (mm)",
            [float(row.get("SpareMm", 0.0)) for row in orderedRows],
        )
        self._addNumericColumn(
            tableNode,
            "Push-Through Offset (mm)",
            [float(row.get("PushThroughOffsetMm", 0.0)) for row in orderedRows],
        )
        self._addStringColumn(tableNode, "Entry Point RAS", [str(row.get("EntryPointRAS", "")) for row in orderedRows])
        self._addStringColumn(tableNode, "Target Point RAS", [str(row.get("TargetPointRAS", "")) for row in orderedRows])
        self._addStringColumn(
            tableNode,
            "Navigation Target RAS",
            [str(row.get("NavigationTargetRAS", "")) for row in orderedRows],
        )

    def populateKeyValueTable(self, tableNode: vtkMRMLTableNode | None, valuesByKey: dict[str, Any]) -> None:
        if not tableNode:
            raise ValueError("Key/value table node is required.")

        orderedKeys = [str(key) for key in valuesByKey.keys()]
        orderedValues: list[str] = []
        for key in orderedKeys:
            value = valuesByKey.get(key)
            if isinstance(value, (dict, list, tuple)):
                orderedValues.append(json.dumps(value, sort_keys=True))
            else:
                orderedValues.append(str(value))
        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "Field", orderedKeys)
        self._addStringColumn(tableNode, "Value", orderedValues)

    def populatePlanSummaryTable(self, tableNode: vtkMRMLTableNode | None, planSummary: dict[str, float | int | str]) -> None:
        if not tableNode:
            raise ValueError("Plan summary table node is required.")

        tableNode.RemoveAllColumns()
        self._addNumericColumn(
            tableNode,
            "Trajectory Count",
            [int(planSummary.get("TrajectoryCount", 0))],
            integer=True,
        )
        self._addStringColumn(tableNode, "Tumor Segment ID", [planSummary.get("TumorSegmentID", "")])
        self._addStringColumn(tableNode, "Tumor Segment Name", [planSummary.get("TumorSegmentName", "")])
        self._addNumericColumn(
            tableNode,
            "Minimum Signed Margin (mm)",
            [float(planSummary.get("MinSignedMarginMm", float("nan")))],
        )
        self._addNumericColumn(
            tableNode,
            "Mean Signed Margin (mm)",
            [float(planSummary.get("MeanSignedMarginMm", float("nan")))],
        )
        self._addNumericColumn(
            tableNode,
            "Median Signed Margin (mm)",
            [float(planSummary.get("MedianSignedMarginMm", float("nan")))],
        )
        self._addNumericColumn(
            tableNode,
            "P20 Signed Margin (mm)",
            [float(planSummary.get("P20SignedMarginMm", float("nan")))],
        )
        self._addNumericColumn(
            tableNode,
            "P80 Signed Margin (mm)",
            [float(planSummary.get("P80SignedMarginMm", float("nan")))],
        )

    def populateMarginThresholdSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        thresholdSummaryRows: Sequence[dict[str, float | int | str]],
    ) -> None:
        if not tableNode:
            raise ValueError("Margin threshold summary table node is required.")

        tableNode.RemoveAllColumns()
        self._addStringColumn(
            tableNode,
            "Margin Bucket",
            [row.get("Bucket", "") for row in thresholdSummaryRows],
        )
        self._addNumericColumn(
            tableNode,
            "Count",
            [int(row.get("Count", 0)) for row in thresholdSummaryRows],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Percent (%)",
            [float(row.get("Percent", 0.0)) for row in thresholdSummaryRows],
        )

    def populateStructureSafetySummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        structureSafetyRows: Sequence[dict[str, float | int | str]],
    ) -> None:
        if not tableNode:
            raise ValueError("Structure safety summary table node is required.")

        tableNode.RemoveAllColumns()
        self._addStringColumn(
            tableNode,
            "Structure Segment ID",
            [row.get("StructureSegmentID", "") for row in structureSafetyRows],
        )
        self._addStringColumn(
            tableNode,
            "Structure Name",
            [row.get("StructureName", "") for row in structureSafetyRows],
        )
        self._addNumericColumn(
            tableNode,
            "Minimum Distance (mm)",
            [float(row.get("MinDistanceMm", float("nan"))) for row in structureSafetyRows],
        )
        self._addNumericColumn(
            tableNode,
            "Mean Distance (mm)",
            [float(row.get("MeanDistanceMm", float("nan"))) for row in structureSafetyRows],
        )
        self._addNumericColumn(
            tableNode,
            "Median Distance (mm)",
            [float(row.get("MedianDistanceMm", float("nan"))) for row in structureSafetyRows],
        )
        self._addNumericColumn(
            tableNode,
            "P20 Distance (mm)",
            [float(row.get("P20DistanceMm", float("nan"))) for row in structureSafetyRows],
        )
        self._addNumericColumn(
            tableNode,
            "P80 Distance (mm)",
            [float(row.get("P80DistanceMm", float("nan"))) for row in structureSafetyRows],
        )

    def populateStructureSafetyThresholdSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        thresholdSummaryRows: Sequence[dict[str, float | int | str]],
    ) -> None:
        if not tableNode:
            raise ValueError("Structure safety threshold summary table node is required.")

        tableNode.RemoveAllColumns()
        self._addStringColumn(
            tableNode,
            "Structure Segment ID",
            [row.get("StructureSegmentID", "") for row in thresholdSummaryRows],
        )
        self._addStringColumn(
            tableNode,
            "Structure Name",
            [row.get("StructureName", "") for row in thresholdSummaryRows],
        )
        self._addNumericColumn(
            tableNode,
            "Count < 0 mm",
            [int(row.get("CountBelow0Mm", 0)) for row in thresholdSummaryRows],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Percent < 0 mm",
            [float(row.get("PercentBelow0Mm", 0.0)) for row in thresholdSummaryRows],
        )
        self._addNumericColumn(
            tableNode,
            "Count < 2 mm",
            [int(row.get("CountBelow2Mm", 0)) for row in thresholdSummaryRows],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Percent < 2 mm",
            [float(row.get("PercentBelow2Mm", 0.0)) for row in thresholdSummaryRows],
        )
        self._addNumericColumn(
            tableNode,
            "Count < 5 mm",
            [int(row.get("CountBelow5Mm", 0)) for row in thresholdSummaryRows],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Percent < 5 mm",
            [float(row.get("PercentBelow5Mm", 0.0)) for row in thresholdSummaryRows],
        )
        self._addNumericColumn(
            tableNode,
            "Count >= 5 mm",
            [int(row.get("CountAtLeast5Mm", 0)) for row in thresholdSummaryRows],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Percent >= 5 mm",
            [float(row.get("PercentAtLeast5Mm", 0.0)) for row in thresholdSummaryRows],
        )

    def populateProbeCoordinationConstraintSettingsTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        settings: ProbeCoordinationConstraintSettings,
    ) -> None:
        if not tableNode:
            raise ValueError("Probe coordination constraint settings table node is required.")

        tableNode.RemoveAllColumns()
        self._addStringColumn(
            tableNode,
            "Setting",
            [
                "MinInterProbeDistanceMm",
                "MaxInterProbeDistanceMm",
                "MinEntryPointSpacingMm",
                "MinTargetPointSpacingMm",
                "MaxParallelAngleDeg",
                "MaxAllowedOverlapPercentBetweenPerProbeVolumes",
                "EnableNoTouchCheck",
                "RequireAllProbePairsFeasible",
                "EnableInterProbeDistanceRule",
                "EnableEntrySpacingRule",
                "EnableTargetSpacingRule",
                "EnableAngleRule",
                "EnableOverlapRule",
            ],
        )
        self._addStringColumn(
            tableNode,
            "Value",
            [
                f"{float(settings.minInterProbeDistanceMm):.3f}",
                f"{float(settings.maxInterProbeDistanceMm):.3f}",
                f"{float(settings.minEntryPointSpacingMm):.3f}",
                f"{float(settings.minTargetPointSpacingMm):.3f}",
                f"{float(settings.maxParallelAngleDeg):.3f}",
                f"{float(settings.maxAllowedOverlapPercentBetweenPerProbeVolumes):.3f}",
                str(bool(settings.enableNoTouchCheck)),
                str(bool(settings.requireAllProbePairsFeasible)),
                str(bool(settings.enableInterProbeDistanceRule)),
                str(bool(settings.enableEntrySpacingRule)),
                str(bool(settings.enableTargetSpacingRule)),
                str(bool(settings.enableAngleRule)),
                str(bool(settings.enableOverlapRule)),
            ],
        )

    def populateProbePairCoordinationSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        pairRows: Sequence[dict[str, float | int | bool | str]],
    ) -> None:
        if not tableNode:
            raise ValueError("Probe pair coordination summary table node is required.")

        orderedRows = sorted(
            pairRows,
            key=lambda row: (int(row.get("ProbeAIndex", 0)), int(row.get("ProbeBIndex", 0))),
        )
        tableNode.RemoveAllColumns()
        self._addNumericColumn(
            tableNode,
            "Probe A Index",
            [int(row.get("ProbeAIndex", 0)) for row in orderedRows],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Probe B Index",
            [int(row.get("ProbeBIndex", 0)) for row in orderedRows],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Is Feasible",
            [1 if bool(row.get("IsFeasible", False)) else 0 for row in orderedRows],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Failed Constraint Count",
            [int(row.get("FailedConstraintCount", 0)) for row in orderedRows],
            integer=True,
        )
        self._addStringColumn(
            tableNode,
            "Failed Constraint Names",
            [str(row.get("FailedConstraintNames", "")) for row in orderedRows],
        )
        self._addNumericColumn(
            tableNode,
            "Inter-Probe Distance (mm)",
            [float(row.get("InterProbeDistanceMm", float("nan"))) for row in orderedRows],
        )
        self._addNumericColumn(
            tableNode,
            "Entry Point Spacing (mm)",
            [float(row.get("EntryPointSpacingMm", float("nan"))) for row in orderedRows],
        )
        self._addNumericColumn(
            tableNode,
            "Target Point Spacing (mm)",
            [float(row.get("TargetPointSpacingMm", float("nan"))) for row in orderedRows],
        )
        self._addNumericColumn(
            tableNode,
            "Probe Axis Angle (deg)",
            [float(row.get("ProbeAxisAngleDeg", float("nan"))) for row in orderedRows],
        )
        self._addNumericColumn(
            tableNode,
            "Overlap Redundancy (%)",
            [float(row.get("OverlapRedundancyPercent", float("nan"))) for row in orderedRows],
        )

    def populateProbeCoordinationSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        summary: dict[str, float | int | bool | str],
    ) -> None:
        if not tableNode:
            raise ValueError("Probe coordination summary table node is required.")

        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "Scenario Or Plan", [summary.get("ScenarioOrPlanName", "CurrentPlan")])
        self._addNumericColumn(tableNode, "Probe Count", [int(summary.get("ProbeCount", 0))], integer=True)
        self._addNumericColumn(tableNode, "Pair Count", [int(summary.get("PairCount", 0))], integer=True)
        self._addNumericColumn(tableNode, "Feasible Pair Count", [int(summary.get("FeasiblePairCount", 0))], integer=True)
        self._addNumericColumn(tableNode, "Infeasible Pair Count", [int(summary.get("InfeasiblePairCount", 0))], integer=True)
        self._addNumericColumn(
            tableNode,
            "All Pairs Feasible",
            [1 if bool(summary.get("AllPairsFeasible", False)) else 0],
            integer=True,
        )
        self._addStringColumn(
            tableNode,
            "Aggregated Failed Constraint Names",
            [summary.get("AggregatedFailedConstraintNames", "")],
        )
        self._addNumericColumn(
            tableNode,
            "No-Touch Pass",
            [1 if bool(summary.get("NoTouchPass", False)) else 0],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Coordination Gate Pass",
            [1 if bool(summary.get("CoordinationGatePass", False)) else 0],
            integer=True,
        )
        self._addStringColumn(
            tableNode,
            "Coordination Failure Summary",
            [str(summary.get("CoordinationFailureSummary", ""))],
        )

    def populateNoTouchSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        noTouchSummary: dict[str, float | int | bool | str],
    ) -> None:
        if not tableNode:
            raise ValueError("No-touch summary table node is required.")

        tableNode.RemoveAllColumns()
        self._addNumericColumn(
            tableNode,
            "No-Touch Checked",
            [1 if bool(noTouchSummary.get("NoTouchChecked", False)) else 0],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "No-Touch Pass",
            [1 if bool(noTouchSummary.get("NoTouchPass", False)) else 0],
            integer=True,
        )
        self._addNumericColumn(
            tableNode,
            "Entry Points Inside Tumor Count",
            [int(noTouchSummary.get("EntryPointsInsideTumorCount", 0))],
            integer=True,
        )
        self._addStringColumn(
            tableNode,
            "Failed Trajectory Indices",
            [str(noTouchSummary.get("FailedTrajectoryIndices", ""))],
        )
        self._addStringColumn(tableNode, "Reason", [str(noTouchSummary.get("Reason", ""))])

    def populateCohortExecutionSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        executionSummary: dict[str, Any],
    ) -> None:
        if not tableNode:
            raise ValueError("Cohort execution summary table node is required.")

        tableNode.RemoveAllColumns()
        orderedFields = [
            "StudyID",
            "StudyDisplayName",
            "ExecutionMode",
            "CaseCount",
            "SuccessCount",
            "FailureCount",
            "SuccessRatePercent",
            "StudyDescription",
        ]
        self._addStringColumn(tableNode, "Field", orderedFields)
        self._addStringColumn(tableNode, "Value", [str(executionSummary.get(field, "")) for field in orderedFields])

    def populateCohortCaseSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        caseResults: Sequence[CohortCaseResult],
    ) -> None:
        if not tableNode:
            raise ValueError("Cohort case summary table node is required.")

        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "CaseID", [result.caseId for result in caseResults])
        self._addStringColumn(tableNode, "DisplayName", [result.displayName for result in caseResults])
        self._addStringColumn(tableNode, "InputReference", [result.inputReference for result in caseResults])
        self._addStringColumn(tableNode, "ScenarioID", [result.scenarioId for result in caseResults])
        self._addStringColumn(tableNode, "ExecutionStatus", [result.executionStatus for result in caseResults])
        self._addStringColumn(tableNode, "StatusMessage", [result.statusMessage for result in caseResults])
        self._addStringColumn(tableNode, "PresetID", [str(result.metricValues.get("PresetID", result.presetId)) for result in caseResults])
        self._addNumericColumn(
            tableNode,
            "TrajectoryCount",
            [
                float(result.metricValues.get("TrajectoryCount", float("nan")))
                if result.executionStatus == "Success"
                else float("nan")
                for result in caseResults
            ],
        )
        self._addNumericColumn(
            tableNode,
            "CoveragePercent",
            [
                float(result.metricValues.get("CoveragePercent", float("nan")))
                if result.executionStatus == "Success"
                else float("nan")
                for result in caseResults
            ],
        )
        self._addNumericColumn(
            tableNode,
            "MinSignedMarginMm",
            [
                float(result.metricValues.get("MinSignedMarginMm", float("nan")))
                if result.executionStatus == "Success"
                else float("nan")
                for result in caseResults
            ],
        )
        self._addNumericColumn(
            tableNode,
            "MedianSignedMarginMm",
            [
                float(result.metricValues.get("MedianSignedMarginMm", float("nan")))
                if result.executionStatus == "Success"
                else float("nan")
                for result in caseResults
            ],
        )
        self._addNumericColumn(
            tableNode,
            "WorstStructureMinDistanceMm",
            [
                float(result.metricValues.get("WorstStructureMinDistanceMm", float("nan")))
                if result.executionStatus == "Success"
                else float("nan")
                for result in caseResults
            ],
        )
        self._addNumericColumn(
            tableNode,
            "CompositeScore",
            [
                float(result.metricValues.get("CompositeScore", float("nan")))
                if result.executionStatus == "Success"
                else float("nan")
                for result in caseResults
            ],
        )
        self._addNumericColumn(
            tableNode,
            "IsFeasible",
            [
                1 if self._coerceBoolean(result.metricValues.get("IsFeasible"), defaultValue=False) else 0
                for result in caseResults
            ],
            integer=True,
        )
        self._addStringColumn(
            tableNode,
            "RecommendationTag",
            [str(result.metricValues.get("RecommendationTag", "")) for result in caseResults],
        )

    def populateCohortAggregateMetricsTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        aggregateMetrics: dict[str, Any],
    ) -> None:
        if not tableNode:
            raise ValueError("Cohort aggregate metrics table node is required.")

        tableNode.RemoveAllColumns()
        orderedFields = sorted(str(fieldName) for fieldName in aggregateMetrics.keys())
        self._addStringColumn(tableNode, "Metric", orderedFields)
        self._addStringColumn(tableNode, "Value", [str(aggregateMetrics.get(fieldName, "")) for fieldName in orderedFields])

    def populateCohortComparisonSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        comparisonRows: Sequence[dict[str, float | int | str]],
    ) -> None:
        if not tableNode:
            raise ValueError("Cohort comparison summary table node is required.")

        orderedRows = sorted(comparisonRows, key=lambda row: str(row.get("PresetID", "")))
        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "PresetID", [str(row.get("PresetID", "")) for row in orderedRows])
        self._addNumericColumn(tableNode, "CaseCount", [int(row.get("CaseCount", 0)) for row in orderedRows], integer=True)
        self._addNumericColumn(tableNode, "SuccessCount", [int(row.get("SuccessCount", 0)) for row in orderedRows], integer=True)
        self._addNumericColumn(
            tableNode,
            "MeanCoveragePercent",
            [float(row.get("MeanCoveragePercent", float("nan"))) for row in orderedRows],
        )
        self._addNumericColumn(
            tableNode,
            "MeanCompositeScore",
            [float(row.get("MeanCompositeScore", float("nan"))) for row in orderedRows],
        )

    def populateExportSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        summaryValues: dict[str, Any],
    ) -> None:
        if not tableNode:
            raise ValueError("Export summary table node is required.")

        orderedFields = [
            "ExportMode",
            "ExportBaseName",
            "SelectedScenarioID",
            "SelectedScenarioName",
            "FileCount",
            "WarningCount",
            "LastExportStatus",
            "LastExportDirectory",
            "LastExportSequence",
        ]
        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "Field", orderedFields)
        self._addStringColumn(tableNode, "Value", [str(summaryValues.get(field, "")) for field in orderedFields])

    def populateExportManifestPreviewTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        manifestValues: dict[str, Any],
    ) -> None:
        if not tableNode:
            raise ValueError("Export manifest preview table node is required.")

        previewFields = [
            "exportId",
            "exportTimestampISO",
            "exportSequence",
            "exportMode",
            "exportBaseName",
            "selectedScenarioID",
            "selectedScenarioName",
            "profileSourceMode",
            "presetID",
            "presetName",
            "targetSegmentID",
            "targetSegmentName",
            "notes",
        ]
        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "Field", previewFields)
        self._addStringColumn(tableNode, "Value", [str(manifestValues.get(field, "")) for field in previewFields])

    def populateReproducibilityPackageSummaryTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        summaryValues: dict[str, Any],
    ) -> None:
        if not tableNode:
            raise ValueError("Reproducibility package summary table node is required.")

        orderedFields = [
            "PackageMode",
            "PackageBaseName",
            "PackagePath",
            "PackageDirectory",
            "ArtifactCount",
            "WarningCount",
            "LastPackageStatus",
            "LastPackageSequence",
        ]
        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "Field", orderedFields)
        self._addStringColumn(tableNode, "Value", [str(summaryValues.get(field, "")) for field in orderedFields])

    def populateReproducibilityManifestPreviewTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        manifestValues: dict[str, Any],
    ) -> None:
        if not tableNode:
            raise ValueError("Reproducibility manifest preview table node is required.")

        previewFields = [
            "packageId",
            "packageTimestampISO",
            "packageSequence",
            "packageMode",
            "packageBaseName",
            "createdByModule",
            "benchmarkCaseIds",
            "studyIds",
            "scenarioIds",
            "reportIds",
            "warnings",
            "notes",
        ]
        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "Field", previewFields)
        previewValues: list[str] = []
        for fieldName in previewFields:
            value = manifestValues.get(fieldName, "")
            if isinstance(value, list):
                previewValues.append(";".join(str(item) for item in value))
            elif isinstance(value, dict):
                previewValues.append(json.dumps(value, sort_keys=True))
            else:
                previewValues.append(str(value))
        self._addStringColumn(tableNode, "Value", previewValues)

    def populateReproducibilityArtifactIndexTable(
        self,
        tableNode: vtkMRMLTableNode | None,
        artifactEntries: Sequence[ReproducibilityArtifactEntry],
    ) -> None:
        if not tableNode:
            raise ValueError("Reproducibility artifact index table node is required.")

        orderedEntries = sorted(artifactEntries, key=lambda entry: entry.relativePath)
        tableNode.RemoveAllColumns()
        self._addStringColumn(tableNode, "ArtifactKey", [str(entry.artifactKey) for entry in orderedEntries])
        self._addStringColumn(tableNode, "Category", [str(entry.category) for entry in orderedEntries])
        self._addStringColumn(tableNode, "RelativePath", [str(entry.relativePath) for entry in orderedEntries])
        self._addStringColumn(tableNode, "Status", [str(entry.status) for entry in orderedEntries])
        self._addStringColumn(tableNode, "SourcePath", [str(entry.sourcePath) for entry in orderedEntries])
        self._addNumericColumn(tableNode, "SizeBytes", [int(entry.sizeBytes) for entry in orderedEntries], integer=True)
        self._addStringColumn(tableNode, "SHA256", [str(entry.sha256) for entry in orderedEntries])
        self._addStringColumn(tableNode, "Warning", [str(entry.warning) for entry in orderedEntries])

    @staticmethod
    def _addStringColumn(tableNode: vtkMRMLTableNode, columnName: str, values: Sequence[float | int | str]) -> None:
        column = vtk.vtkStringArray()
        column.SetName(columnName)
        for value in values:
            column.InsertNextValue("" if value is None else str(value))
        tableNode.AddColumn(column)

    @staticmethod
    def _addNumericColumn(
        tableNode: vtkMRMLTableNode,
        columnName: str,
        values: Sequence[float | int],
        integer: bool = False,
    ) -> None:
        column = vtk.vtkIntArray() if integer else vtk.vtkDoubleArray()
        column.SetName(columnName)
        for value in values:
            if integer:
                column.InsertNextValue(int(value))
            else:
                column.InsertNextValue(float(value))
        tableNode.AddColumn(column)

    def getValidSegmentationSegments(
        self,
        segmentationNode: vtkMRMLSegmentationNode | None,
        operationName: str,
    ) -> list[dict[str, str]]:
        if not segmentationNode:
            raise ValueError(f"{operationName}: segmentation node is required.")

        self._ensureSegmentationHasClosedSurface(segmentationNode)
        segmentation = segmentationNode.GetSegmentation()
        validSegments: list[dict[str, str]] = []
        for segmentIndex in range(segmentation.GetNumberOfSegments()):
            segmentID = segmentation.GetNthSegmentID(segmentIndex)
            if not segmentID:
                continue
            closedSurface = vtk.vtkPolyData()
            segmentationNode.GetClosedSurfaceRepresentation(segmentID, closedSurface)
            if closedSurface.GetNumberOfPoints() <= 0:
                logging.warning("%s: segment '%s' has no closed-surface points and will be skipped.", operationName, segmentID)
                continue
            segment = segmentation.GetSegment(segmentID)
            segmentName = segment.GetName() if segment and segment.GetName() else segmentID
            validSegments.append(
                {
                    "segmentID": segmentID,
                    "segmentName": segmentName,
                }
            )

        if len(validSegments) == 0:
            raise RuntimeError(
                f"{operationName}: segmentation '{segmentationNode.GetName()}' has no valid closed-surface segments."
            )
        return validSegments

    def segmentationSegmentToModel(
        self,
        segmentationNode: vtkMRMLSegmentationNode,
        segmentID: str,
        modelName: str,
        outputModelNode: vtkMRMLModelNode | None = None,
    ) -> vtkMRMLModelNode:
        self._ensureSegmentationHasClosedSurface(segmentationNode)

        segmentation = segmentationNode.GetSegmentation()
        segment = segmentation.GetSegment(segmentID) if segmentation else None
        if segment is None:
            raise RuntimeError(
                f"model conversion for '{modelName}': segment '{segmentID}' was not found in '{segmentationNode.GetName()}'."
            )

        closedSurface = vtk.vtkPolyData()
        segmentationNode.GetClosedSurfaceRepresentation(segmentID, closedSurface)
        if closedSurface.GetNumberOfPoints() <= 0:
            raise RuntimeError(f"Segment '{segmentID}' has no closed-surface representation.")

        modelNode = outputModelNode
        if not modelNode or not slicer.mrmlScene.IsNodePresent(modelNode):
            modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", modelName)

        modelNode.CreateDefaultDisplayNodes()
        polyDataCopy = vtk.vtkPolyData()
        polyDataCopy.DeepCopy(closedSurface)
        modelNode.SetAndObservePolyData(polyDataCopy)

        modelDisplayNode = modelNode.GetDisplayNode()
        if modelDisplayNode:
            modelDisplayNode.SetVisibility(False)
        return modelNode

    def segmentationFirstSegmentToModel(
        self,
        segmentationNode: vtkMRMLSegmentationNode,
        modelName: str,
        outputModelNode: vtkMRMLModelNode | None = None,
    ) -> vtkMRMLModelNode:
        segmentID = self.getWorkingSegmentID(segmentationNode, f"model conversion for '{modelName}'")
        return self.segmentationSegmentToModel(
            segmentationNode,
            segmentID,
            modelName,
            outputModelNode=outputModelNode,
        )

    @staticmethod
    def _requireModelPolyData(modelNode: vtkMRMLModelNode, modelRole: str) -> vtk.vtkPolyData:
        mesh = modelNode.GetMesh()
        polyData = vtk.vtkPolyData.SafeDownCast(mesh)
        if (
            polyData is None
            or polyData.GetNumberOfPoints() <= 0
            or polyData.GetNumberOfCells() <= 0
        ):
            raise RuntimeError(f"{modelRole} model '{modelNode.GetName()}' has no valid polydata surface.")
        if not SurgicalVision3D_PlannerLogic._polyDataHasFinitePoints(polyData):
            raise RuntimeError(f"{modelRole} model '{modelNode.GetName()}' contains non-finite coordinates.")
        return polyData

    @classmethod
    def _preparePolyDataForDistanceComputation(
        cls,
        inputPolyData: vtk.vtkPolyData | None,
        modelRole: str,
        orientNormals: bool = False,
    ) -> vtk.vtkPolyData:
        if (
            inputPolyData is None
            or inputPolyData.GetNumberOfPoints() <= 0
            or inputPolyData.GetNumberOfCells() <= 0
        ):
            raise RuntimeError(f"{modelRole} polydata is empty and cannot be used for distance computation.")

        surfaceCopy = vtk.vtkPolyData()
        surfaceCopy.DeepCopy(inputPolyData)

        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(surfaceCopy)
        if hasattr(triangleFilter, "PassLinesOff"):
            triangleFilter.PassLinesOff()
        if hasattr(triangleFilter, "PassVertsOff"):
            triangleFilter.PassVertsOff()
        triangleFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(triangleFilter.GetOutputPort())
        cleanFilter.Update()

        preparedPolyData = vtk.vtkPolyData()
        if orientNormals:
            normalsFilter = vtk.vtkPolyDataNormals()
            normalsFilter.SetInputConnection(cleanFilter.GetOutputPort())
            normalsFilter.AutoOrientNormalsOn()
            normalsFilter.ConsistencyOn()
            normalsFilter.SplittingOff()
            if hasattr(normalsFilter, "ComputeCellNormalsOff"):
                normalsFilter.ComputeCellNormalsOff()
            normalsFilter.Update()
            preparedPolyData.DeepCopy(normalsFilter.GetOutput())
        if preparedPolyData.GetNumberOfPoints() <= 0 or preparedPolyData.GetNumberOfCells() <= 0:
            preparedPolyData.DeepCopy(cleanFilter.GetOutput())
        if preparedPolyData.GetNumberOfPoints() <= 2 or preparedPolyData.GetNumberOfCells() <= 0:
            raise RuntimeError(f"{modelRole} polydata has insufficient surface geometry after cleanup.")
        if not cls._polyDataHasFinitePoints(preparedPolyData):
            raise RuntimeError(f"{modelRole} polydata has non-finite coordinates after cleanup.")
        return preparedPolyData

    def computeSignedDistanceModel(
        self,
        sourceModelNode: vtkMRMLModelNode,
        targetModelNode: vtkMRMLModelNode,
        outputModelNode: vtkMRMLModelNode,
        preparedTargetPolyData: vtk.vtkPolyData | None = None,
    ) -> vtkMRMLModelNode:
        sourcePolyData = self._preparePolyDataForDistanceComputation(
            self._requireModelPolyData(sourceModelNode, "Source"),
            "Source",
            orientNormals=False,
        )
        targetPolyData = (
            preparedTargetPolyData
            if preparedTargetPolyData is not None
            else self._preparePolyDataForDistanceComputation(
                self._requireModelPolyData(targetModelNode, "Target"),
                "Target",
                orientNormals=True,
            )
        )
        if (
            targetPolyData is None
            or targetPolyData.GetNumberOfPoints() <= 2
            or targetPolyData.GetNumberOfCells() <= 0
            or not self._polyDataHasFinitePoints(targetPolyData)
        ):
            raise RuntimeError("Target polydata is invalid for signed-distance computation.")

        if ENABLE_MAM_DEBUG_LOGGING:
            logging.info(
                "MAM debug: computeSignedDistanceModel start | source=%s | target=%s | fastPathEnabled=%s",
                self._polyDataSummaryText(sourcePolyData),
                self._polyDataSummaryText(targetPolyData),
                bool(ENABLE_VTK_DISTANCE_FAST_PATH),
            )

        outputPolyData = None
        signedDistances = None

        # Fast path: C++ distance + enclosed-point sign mask.
        if ENABLE_VTK_DISTANCE_FAST_PATH:
            try:
                distanceFilter = vtk.vtkDistancePolyDataFilter()
                distanceFilter.SetInputData(0, sourcePolyData)
                distanceFilter.SetInputData(1, targetPolyData)
                if hasattr(distanceFilter, "SignedDistanceOff"):
                    distanceFilter.SignedDistanceOff()
                elif hasattr(distanceFilter, "SetSignedDistance"):
                    distanceFilter.SetSignedDistance(False)
                if hasattr(distanceFilter, "ComputeSecondDistanceOff"):
                    distanceFilter.ComputeSecondDistanceOff()
                distanceFilter.Update()

                distanceOutput = vtk.vtkPolyData()
                distanceOutput.DeepCopy(distanceFilter.GetOutput())
                pointData = distanceOutput.GetPointData()
                distanceArray = pointData.GetArray("Distance") if pointData else None
                if distanceArray is None and pointData:
                    distanceArray = pointData.GetScalars()

                enclosedFilter = vtk.vtkSelectEnclosedPoints()
                enclosedFilter.SetInputData(distanceOutput)
                enclosedFilter.SetSurfaceData(targetPolyData)
                enclosedFilter.SetTolerance(1e-6)
                if hasattr(enclosedFilter, "CheckSurfaceOff"):
                    enclosedFilter.CheckSurfaceOff()
                enclosedFilter.Update()
                insideMaskArray = enclosedFilter.GetOutput().GetPointData().GetArray("SelectedPoints")

                pointCount = int(distanceOutput.GetNumberOfPoints())
                if (
                    distanceArray is not None
                    and insideMaskArray is not None
                    and _data_array_value_count(distanceArray) == pointCount
                    and _data_array_value_count(insideMaskArray) == pointCount
                ):
                    unsignedDistances = np.asarray(
                        numpy_support.vtk_to_numpy(distanceArray),
                        dtype=np.float64,
                    )
                    insideMask = np.asarray(
                        numpy_support.vtk_to_numpy(insideMaskArray),
                        dtype=np.uint8,
                    ) > 0
                    signedDistances = np.abs(unsignedDistances)
                    signedDistances[insideMask] *= -1.0
                    outputPolyData = distanceOutput
            except Exception:
                logging.debug("Fast signed-distance path failed; falling back to implicit distance.", exc_info=True)

        if signedDistances is not None and (not np.any(np.isfinite(signedDistances))):
            signedDistances = None
            outputPolyData = None

        # Fallback path: robust implicit distance evaluation.
        if signedDistances is None or outputPolyData is None:
            if ENABLE_MAM_DEBUG_LOGGING:
                logging.info("MAM debug: using implicit signed-distance fallback path.")
            outputPolyData = vtk.vtkPolyData()
            outputPolyData.DeepCopy(sourcePolyData)
            points = outputPolyData.GetPoints()
            if points is None or points.GetData() is None:
                raise RuntimeError("Source polydata points are unavailable for signed-distance computation.")

            sourcePointsNumpy = np.asarray(numpy_support.vtk_to_numpy(points.GetData()), dtype=np.float64)
            if sourcePointsNumpy.ndim != 2 or sourcePointsNumpy.shape[0] <= 0 or sourcePointsNumpy.shape[1] < 3:
                raise RuntimeError("Source polydata point coordinates are invalid for signed-distance computation.")

            implicitDistance = vtk.vtkImplicitPolyDataDistance()
            implicitDistance.SetInput(targetPolyData)

            enclosedFilter = vtk.vtkSelectEnclosedPoints()
            enclosedFilter.SetInputData(outputPolyData)
            enclosedFilter.SetSurfaceData(targetPolyData)
            enclosedFilter.SetTolerance(1e-6)
            if hasattr(enclosedFilter, "CheckSurfaceOff"):
                enclosedFilter.CheckSurfaceOff()
            enclosedFilter.Update()

            pointCount = int(sourcePointsNumpy.shape[0])
            insideMask = np.zeros(pointCount, dtype=bool)
            insideMaskArray = enclosedFilter.GetOutput().GetPointData().GetArray("SelectedPoints")
            if (
                insideMaskArray is not None
                and _data_array_value_count(insideMaskArray) == pointCount
            ):
                insideMask = np.asarray(
                    numpy_support.vtk_to_numpy(insideMaskArray),
                    dtype=np.uint8,
                ) > 0
            else:
                # Fall back to API query if the array was not emitted by this VTK build.
                for pointIndex in range(pointCount):
                    insideMask[pointIndex] = bool(enclosedFilter.IsInside(pointIndex))

            signedDistances = np.empty(pointCount, dtype=np.float64)
            for pointIndex in range(pointCount):
                point = sourcePointsNumpy[pointIndex]
                unsignedDistance = abs(float(
                    implicitDistance.EvaluateFunction((float(point[0]), float(point[1]), float(point[2])))
                ))
                signedDistances[pointIndex] = -unsignedDistance if insideMask[pointIndex] else unsignedDistance

        finiteMask = np.isfinite(signedDistances)
        if not np.any(finiteMask):
            raise RuntimeError("Signed-distance computation produced no finite values.")
        invalidValueCount = int(np.count_nonzero(~finiteMask))
        if invalidValueCount > 0:
            logging.warning(
                "Signed-distance computation replaced %d non-finite values with 0.0.",
                invalidValueCount,
            )
            signedDistances[~finiteMask] = 0.0

        signedDistanceArray = numpy_support.numpy_to_vtk(
            signedDistances,
            deep=True,
            array_type=vtk.VTK_DOUBLE,
        )
        signedDistanceArray.SetName(SIGNED_DISTANCE_ARRAY_NAME)

        pointData = outputPolyData.GetPointData()
        if pointData is None:
            raise RuntimeError("Output polydata point data is unavailable for signed-distance results.")
        if SIGNED_DISTANCE_ARRAY_NAME != "Distance" and pointData.GetArray("Distance"):
            pointData.RemoveArray("Distance")
        if pointData.GetArray(SIGNED_DISTANCE_ARRAY_NAME):
            pointData.RemoveArray(SIGNED_DISTANCE_ARRAY_NAME)
        pointData.AddArray(signedDistanceArray)
        pointData.SetActiveScalars(SIGNED_DISTANCE_ARRAY_NAME)

        outputModelNode.SetAndObservePolyData(outputPolyData)
        outputModelNode.CreateDefaultDisplayNodes()
        return outputModelNode

    def getModelFieldData(self, modelNode: vtkMRMLModelNode) -> vtk.vtkFieldData:
        mesh = modelNode.GetMesh()
        if mesh is None:
            raise RuntimeError("Margin model does not contain mesh data.")
        fieldData = mesh.GetAttributesAsFieldData(0)
        if fieldData is None:
            raise RuntimeError("Margin model field data is unavailable.")
        return fieldData

    def getSignedDistanceArray(self, marginModelNode: vtkMRMLModelNode) -> vtk.vtkDataArray:
        fieldData = self.getModelFieldData(marginModelNode)
        signedDistanceArray = fieldData.GetArray(SIGNED_DISTANCE_ARRAY_NAME)
        if signedDistanceArray is None:
            raise RuntimeError(f"Signed distance array '{SIGNED_DISTANCE_ARRAY_NAME}' was not found.")
        return signedDistanceArray

    def getSignedDistanceBackupArray(self, marginModelNode: vtkMRMLModelNode) -> vtk.vtkDataArray | None:
        fieldData = self.getModelFieldData(marginModelNode)
        return fieldData.GetArray(SIGNED_DISTANCE_BACKUP_ARRAY_NAME)

    def backupSignedDistanceArray(self, marginModelNode: vtkMRMLModelNode, signedDistanceArray: vtk.vtkDataArray) -> None:
        fieldData = self.getModelFieldData(marginModelNode)
        if fieldData.GetArray(SIGNED_DISTANCE_BACKUP_ARRAY_NAME):
            fieldData.RemoveArray(SIGNED_DISTANCE_BACKUP_ARRAY_NAME)

        backupArray = signedDistanceArray.NewInstance()
        backupArray.DeepCopy(signedDistanceArray)
        backupArray.SetName(SIGNED_DISTANCE_BACKUP_ARRAY_NAME)
        fieldData.AddArray(backupArray)

    def configureMarginDisplayNode(
        self,
        marginModelNode: vtkMRMLModelNode,
        autoRange: bool = True,
        scalarRange: tuple[float, float] | None = None,
    ) -> None:
        displayNode = marginModelNode.GetDisplayNode()
        if displayNode is None:
            return

        displayNode.SetVisibility(True)
        if hasattr(displayNode, "SetVisibility2D"):
            displayNode.SetVisibility2D(True)
        elif hasattr(displayNode, "SetSliceIntersectionVisibility"):
            displayNode.SetSliceIntersectionVisibility(True)
        displayNode.SetSliceDisplayModeToIntersection()
        displayNode.SetSliceIntersectionThickness(2)
        displayNode.SetScalarVisibility(True)
        displayNode.SetActiveScalarName(SIGNED_DISTANCE_ARRAY_NAME)
        displayNode.SetAndObserveColorNodeID(DEFAULT_MARGIN_COLOR_NODE_ID)

        if autoRange:
            displayNode.AutoScalarRangeOn()
        else:
            displayNode.AutoScalarRangeOff()
            if scalarRange is not None:
                displayNode.SetScalarRange(scalarRange[0], scalarRange[1])

    @staticmethod
    def signedDistanceSummary(signedDistanceArray: vtk.vtkDataArray) -> dict[str, float]:
        values = [signedDistanceArray.GetValue(index) for index in range(_data_array_value_count(signedDistanceArray))]
        if len(values) == 0:
            return {"min": float("nan"), "mean": float("nan"), "median": float("nan"), "max": float("nan")}
        return {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }

    @staticmethod
    def refreshNodeDisplay(node: vtk.vtkObject) -> None:
        displayNode = node.GetDisplayNode() if hasattr(node, "GetDisplayNode") else None
        if displayNode:
            displayNode.Modified()
        node.Modified()

    def getWorkingSegmentID(self, segmentationNode: vtkMRMLSegmentationNode | None, operationName: str) -> str:
        if not segmentationNode:
            raise ValueError(f"{operationName}: segmentation node is required.")
        segmentation = segmentationNode.GetSegmentation()
        if segmentation.GetNumberOfSegments() <= 0:
            raise RuntimeError(
                f"{operationName}: segmentation '{segmentationNode.GetName()}' has no segments."
            )

        # Phase 1 uses first-segment policy for probe/tumor workflow.
        segmentID = segmentation.GetNthSegmentID(0)
        if not segmentID:
            raise RuntimeError(
                f"{operationName}: failed to resolve the first segment in '{segmentationNode.GetName()}'."
            )
        return segmentID

    def _ensureSegmentationHasClosedSurface(self, segmentationNode: vtkMRMLSegmentationNode | None) -> None:
        if not segmentationNode:
            raise ValueError("Segmentation node is required.")
        segmentation = segmentationNode.GetSegmentation()
        if segmentation.GetNumberOfSegments() <= 0:
            raise RuntimeError(f"Segmentation '{segmentationNode.GetName()}' has no segments.")
        self._ensureSegmentationReferenceImageGeometry(segmentationNode)
        segmentation.CreateRepresentation("Closed surface")

    def _cloneReferenceProbe(
        self,
        referenceProbeSegmentation: vtkMRMLSegmentationNode,
        trajectoryIndex: int,
        sourceSurface: vtk.vtkPolyData | None = None,
    ) -> vtkMRMLSegmentationNode:
        resolvedSourceSurface = sourceSurface
        if resolvedSourceSurface is None:
            sourceSegmentID = self.getWorkingSegmentID(referenceProbeSegmentation, "reference probe placement")
            resolvedSourceSurface = vtk.vtkPolyData()
            referenceProbeSegmentation.GetClosedSurfaceRepresentation(sourceSegmentID, resolvedSourceSurface)
        if (
            resolvedSourceSurface.GetNumberOfPoints() <= 0
            or resolvedSourceSurface.GetNumberOfCells() <= 0
            or not self._polyDataHasFinitePoints(resolvedSourceSurface)
        ):
            raise RuntimeError("Reference probe segmentation has no usable closed-surface geometry.")

        sourceSurfaceCopy = vtk.vtkPolyData()
        sourceSurfaceCopy.DeepCopy(resolvedSourceSurface)

        clonedProbeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            f"SV3D Placed Probe {trajectoryIndex + 1:02d}",
        )
        clonedProbeNode.CreateDefaultDisplayNodes()
        self._preferClosedSurfaceSourceRepresentation(clonedProbeNode)
        self._ensureSegmentationReferenceImageGeometry(clonedProbeNode)
        clonedProbeNode.AddSegmentFromClosedSurfaceRepresentation(sourceSurfaceCopy, f"Probe_{trajectoryIndex + 1:02d}", [0.2, 0.9, 0.3])
        clonedProbeNode.GetSegmentation().CreateRepresentation("Closed surface")
        clonedProbeNode.SetAttribute(GENERATED_PROBE_ATTRIBUTE, "1")

        clonedDisplayNode = clonedProbeNode.GetDisplayNode()
        if clonedDisplayNode:
            clonedDisplayNode.SetOpacity(0.35)
            clonedDisplayNode.SetVisibility(True)
        return clonedProbeNode

    def _placeProbeNodeAlongTrajectory(
        self,
        probeNode: vtkMRMLSegmentationNode,
        trajectory: ProbeTrajectory,
        axialPlacementOffsetMm: float = 0.0,
    ) -> None:
        rotationMatrix = rotation_matrix_from_vectors(REFERENCE_PROBE_DIRECTION_RAS, trajectory.directionVector)
        placementOrigin = np.asarray(trajectory.entryPointRAS, dtype=float)
        if abs(float(axialPlacementOffsetMm)) > 1e-6:
            trajectoryDirection = _normalize_vector(trajectory.directionVector)
            placementOrigin = placementOrigin + (trajectoryDirection * float(axialPlacementOffsetMm))
        transformMatrix = _build_rigid_transform(rotationMatrix, placementOrigin.tolist())

        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", f"ProbePlacement_{trajectory.trajectoryIndex + 1:02d}")
        transformNode.SetMatrixTransformToParent(slicer.util.vtkMatrixFromArray(transformMatrix))

        probeNode.SetAndObserveTransformNodeID(transformNode.GetID())
        probeNode.HardenTransform()
        slicer.mrmlScene.RemoveNode(transformNode)

    def _clearSegmentationSegments(self, segmentationNode: vtkMRMLSegmentationNode) -> None:
        segmentation = segmentationNode.GetSegmentation()
        while segmentation.GetNumberOfSegments() > 0:
            segmentation.RemoveSegment(segmentation.GetNthSegmentID(0))

    def _segmentIDs(self, segmentationNode: vtkMRMLSegmentationNode) -> list[str]:
        segmentation = segmentationNode.GetSegmentation()
        return [segmentation.GetNthSegmentID(segmentIndex) for segmentIndex in range(segmentation.GetNumberOfSegments())]

    def _unionSegmentsWithLogicalOperators(self, segmentationNode: vtkMRMLSegmentationNode) -> None:
        segmentIDs = self._segmentIDs(segmentationNode)
        if len(segmentIDs) <= 1:
            return

        self.prepareSegmentationForEditing(segmentationNode)
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")

        try:
            segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
            segmentEditorWidget.setSegmentationNode(segmentationNode)
            segmentEditorWidget.setCurrentSegmentID(segmentIDs[0])
            segmentEditorWidget.setActiveEffectByName("Logical operators")
            effect = segmentEditorWidget.activeEffect()
            if effect is None:
                raise RuntimeError("Logical operators effect is unavailable.")

            appliedModifierSegmentIDs: list[str] = []
            for modifierSegmentID in segmentIDs[1:]:
                if not segmentationNode.GetSegmentation().GetSegment(modifierSegmentID):
                    continue
                effect.self().scriptedEffect.setParameter("Operation", "UNION")
                effect.self().scriptedEffect.setParameter("ModifierSegmentID", modifierSegmentID)
                effect.self().onApply()
                appliedModifierSegmentIDs.append(modifierSegmentID)

            # Clear editor references before mutating/removing modifier segments.
            try:
                segmentEditorWidget.setActiveEffectByName("")
            except Exception:
                pass
            try:
                segmentEditorWidget.setSegmentationNode(None)
                segmentEditorWidget.setMRMLSegmentEditorNode(None)
            except Exception:
                pass

            for modifierSegmentID in appliedModifierSegmentIDs:
                if segmentationNode.GetSegmentation().GetSegment(modifierSegmentID):
                    segmentationNode.GetSegmentation().RemoveSegment(modifierSegmentID)
        finally:
            if segmentEditorWidget and hasattr(segmentEditorWidget, "deleteLater"):
                try:
                    segmentEditorWidget.deleteLater()
                except Exception:
                    pass
            segmentEditorWidget = None
            if segmentEditorNode and slicer.mrmlScene.IsNodePresent(segmentEditorNode):
                slicer.mrmlScene.RemoveNode(segmentEditorNode)

    def _mergeSegmentsByAppendingSurfaces(self, segmentationNode: vtkMRMLSegmentationNode) -> None:
        self._ensureSegmentationHasClosedSurface(segmentationNode)
        segmentIDs = self._segmentIDs(segmentationNode)
        if len(segmentIDs) <= 1:
            return

        appendFilter = vtk.vtkAppendPolyData()
        for segmentID in segmentIDs:
            closedSurface = vtk.vtkPolyData()
            segmentationNode.GetClosedSurfaceRepresentation(segmentID, closedSurface)
            if closedSurface.GetNumberOfPoints() <= 0:
                continue
            surfaceCopy = vtk.vtkPolyData()
            surfaceCopy.DeepCopy(closedSurface)
            appendFilter.AddInputData(surfaceCopy)

        appendFilter.Update()
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.Update()
        if cleanFilter.GetOutput().GetNumberOfPoints() <= 0:
            raise RuntimeError("Probe merge fallback produced an empty closed surface.")

        self._clearSegmentationSegments(segmentationNode)
        segmentationNode.AddSegmentFromClosedSurfaceRepresentation(cleanFilter.GetOutput(), "CombinedAblationZone", [1.0, 0.3, 0.1])


#
# SurgicalVision3D_PlannerTest
#


class SurgicalVision3D_PlannerTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_rotation_matrix_standard_parallel_antiparallel()
        self.test_extract_trajectories_from_paired_points()
        self.test_extract_trajectories_odd_count_handling()
        self.test_extract_single_master_trajectory_from_markups()
        self.test_orthogonal_array_basis_is_deterministic_and_orthonormal()
        self.test_generate_derived_trajectory_array_spacing_and_parallelism()
        self.test_generate_derived_trajectory_array_include_master_toggle()
        self.test_geometry_catalog_loading()
        self.test_mam_assessment_summary()
        self.test_compute_coaxial_plan_pushthrough_offset()
        self.test_recolor_and_restore_include_last_array_value()
        self.test_parameter_node_initialization_and_restore()

    def test_rotation_matrix_standard_parallel_antiparallel(self):
        source = np.array([0.0, 0.0, -1.0], dtype=float)
        target = np.array([0.0, 1.0, 0.0], dtype=float)
        rotation = rotation_matrix_from_vectors(source, target)
        transformed = rotation.dot(_normalize_vector(source))
        self.assertTrue(np.allclose(transformed, _normalize_vector(target), atol=1e-6))

        parallelRotation = rotation_matrix_from_vectors(source, source)
        parallelTransformed = parallelRotation.dot(_normalize_vector(source))
        self.assertTrue(np.allclose(parallelTransformed, _normalize_vector(source), atol=1e-6))

        antiParallelRotation = rotation_matrix_from_vectors(source, -source)
        antiParallelTransformed = antiParallelRotation.dot(_normalize_vector(source))
        self.assertTrue(np.allclose(antiParallelTransformed, _normalize_vector(-source), atol=1e-6))

    def test_extract_trajectories_from_paired_points(self):
        points = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, -10.0),
            (5.0, 1.0, 0.0),
            (5.0, 1.0, -4.0),
        ]
        trajectories = SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(points, strictEven=True)

        self.assertEqual(len(trajectories), 2)
        self.assertEqual(trajectories[0].sourceControlPointIndices, (0, 1))
        self.assertTrue(np.allclose(np.array(trajectories[0].entryPointRAS), np.array([0.0, 0.0, 0.0]), atol=1e-6))
        self.assertTrue(np.allclose(np.array(trajectories[0].targetPointRAS), np.array([0.0, 0.0, -10.0]), atol=1e-6))
        self.assertTrue(np.allclose(np.array(trajectories[0].directionVector), np.array([0.0, 0.0, -1.0]), atol=1e-6))
        self.assertAlmostEqual(trajectories[0].lengthMm, 10.0, places=6)
        self.assertAlmostEqual(trajectories[1].lengthMm, 4.0, places=6)

    def test_extract_trajectories_odd_count_handling(self):
        oddPoints = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, -10.0),
            (2.0, 2.0, 2.0),
        ]
        with self.assertRaises(ValueError):
            SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(oddPoints, strictEven=True)

        trajectories = SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(oddPoints, strictEven=False)
        self.assertEqual(len(trajectories), 1)

    def test_extract_single_master_trajectory_from_markups(self):
        logic = SurgicalVision3D_PlannerLogic()
        endpointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "endpoints")
        slicer.util.updateMarkupsControlPointsFromArray(
            endpointsNode,
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -15.0)], dtype=float),
        )

        trajectory = logic.extractSingleTrajectoryFromMarkups(endpointsNode)
        self.assertAlmostEqual(float(trajectory.lengthMm), 15.0, places=6)
        self.assertTrue(np.allclose(np.array(trajectory.entryPointRAS), np.array([0.0, 0.0, 0.0]), atol=1e-6))
        self.assertTrue(np.allclose(np.array(trajectory.targetPointRAS), np.array([0.0, 0.0, -15.0]), atol=1e-6))

        slicer.util.updateMarkupsControlPointsFromArray(
            endpointsNode,
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -15.0), (5.0, 0.0, 0.0), (5.0, 0.0, -5.0)], dtype=float),
        )
        with self.assertRaises(ValueError):
            logic.extractSingleTrajectoryFromMarkups(endpointsNode)

    def test_orthogonal_array_basis_is_deterministic_and_orthonormal(self):
        logic = SurgicalVision3D_PlannerLogic()
        direction = np.array([0.2, -0.3, -0.93], dtype=float)
        direction = direction / np.linalg.norm(direction)
        basisU1, basisV1 = logic.createOrthogonalArrayBasis(direction)
        basisU2, basisV2 = logic.createOrthogonalArrayBasis(direction)

        self.assertTrue(np.allclose(basisU1, basisU2, atol=1e-6))
        self.assertTrue(np.allclose(basisV1, basisV2, atol=1e-6))
        self.assertAlmostEqual(float(np.dot(basisU1, direction)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(basisV1, direction)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(basisU1, basisV1)), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(basisU1)), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(basisV1)), 1.0, places=6)

    def test_generate_derived_trajectory_array_spacing_and_parallelism(self):
        logic = SurgicalVision3D_PlannerLogic()
        master = ProbeTrajectory(
            entryPointRAS=(10.0, 20.0, 30.0),
            targetPointRAS=(12.0, 23.0, 10.0),
            directionVector=tuple((_normalize_vector((2.0, 3.0, -20.0))).tolist()),
            lengthMm=float(np.linalg.norm(np.array([2.0, 3.0, -20.0], dtype=float))),
            trajectoryIndex=0,
            role="Master",
        )
        derivedCount = 6
        radiusMm = 10.0
        bundle = logic.generateDerivedParallelTrajectories(
            master,
            derivedCount=derivedCount,
            radiusMm=radiusMm,
            angleOffsetDeg=0.0,
            includeMaster=True,
        )

        self.assertEqual(len(bundle), 7)
        derivedTrajectories = [trajectory for trajectory in bundle if trajectory.derivedFromMaster]
        self.assertEqual(len(derivedTrajectories), 6)

        angleValues = sorted(float(trajectory.angleDeg or 0.0) for trajectory in derivedTrajectories)
        spacingValues = []
        for index in range(len(angleValues)):
            currentAngle = angleValues[index]
            nextAngle = angleValues[(index + 1) % len(angleValues)]
            delta = (nextAngle - currentAngle) % 360.0
            spacingValues.append(delta)
        for spacing in spacingValues:
            self.assertAlmostEqual(float(spacing), 60.0, places=4)

        masterDirection = _normalize_vector(master.directionVector)
        masterEntry = np.asarray(master.entryPointRAS, dtype=float)
        masterTarget = np.asarray(master.targetPointRAS, dtype=float)
        for derivedTrajectory in derivedTrajectories:
            derivedDirection = _normalize_vector(derivedTrajectory.directionVector)
            self.assertTrue(np.allclose(derivedDirection, masterDirection, atol=1e-6))
            self.assertAlmostEqual(float(derivedTrajectory.lengthMm), float(master.lengthMm), places=6)
            offsetEntry = np.asarray(derivedTrajectory.entryPointRAS, dtype=float) - masterEntry
            offsetTarget = np.asarray(derivedTrajectory.targetPointRAS, dtype=float) - masterTarget
            self.assertTrue(np.allclose(offsetEntry, offsetTarget, atol=1e-6))
            self.assertAlmostEqual(float(np.linalg.norm(offsetEntry)), radiusMm, places=4)

        bundle4 = logic.generateDerivedParallelTrajectories(
            master,
            derivedCount=4,
            radiusMm=radiusMm,
            angleOffsetDeg=0.0,
            includeMaster=True,
        )
        cardinalRoles = [trajectory.role for trajectory in bundle4 if trajectory.derivedFromMaster]
        self.assertEqual(cardinalRoles, ["North", "East", "South", "West"])

    def test_generate_derived_trajectory_array_include_master_toggle(self):
        logic = SurgicalVision3D_PlannerLogic()
        master = ProbeTrajectory(
            entryPointRAS=(0.0, 0.0, 0.0),
            targetPointRAS=(0.0, 0.0, -20.0),
            directionVector=(0.0, 0.0, -1.0),
            lengthMm=20.0,
            trajectoryIndex=0,
            role="Master",
        )
        withMaster = logic.generateDerivedParallelTrajectories(
            master,
            derivedCount=4,
            radiusMm=8.0,
            includeMaster=True,
        )
        withoutMaster = logic.generateDerivedParallelTrajectories(
            master,
            derivedCount=4,
            radiusMm=8.0,
            includeMaster=False,
        )

        self.assertEqual(len(withMaster), 5)
        self.assertEqual(len(withoutMaster), 4)
        self.assertEqual(withMaster[0].role, "Master")
        self.assertTrue(all(trajectory.derivedFromMaster for trajectory in withoutMaster))

    def test_geometry_catalog_loading(self):
        geometryEntries = SurgicalVision3D_PlannerLogic().loadGeometryCatalog()
        self.assertGreater(len(geometryEntries), 0)
        self.assertEqual(geometryEntries[0].geometryId, "emprint_75w_5m")
        self.assertTrue(all(float(entry.activeElementLengthMm) > 0.0 for entry in geometryEntries))

    def test_mam_assessment_summary(self):
        summary = SurgicalVision3D_PlannerLogic.computeMamAssessmentSummary((-12.0, -7.0, -4.0), 10.0)
        self.assertFalse(bool(summary["MamPass"]))
        self.assertEqual(int(summary["CountRed"]), 1)
        self.assertEqual(int(summary["CountOrange"]), 1)
        self.assertEqual(int(summary["CountGreen"]), 1)
        self.assertAlmostEqual(float(summary["MinAchievedMarginMm"]), 4.0, places=6)

    def test_compute_coaxial_plan_pushthrough_offset(self):
        trajectory = ProbeTrajectory(
            entryPointRAS=(0.0, 0.0, 0.0),
            targetPointRAS=(0.0, 0.0, -20.0),
            # Deliberately opposite direction vector: coaxial planning must use entry/endpoint points.
            directionVector=(0.0, 0.0, 1.0),
            lengthMm=20.0,
            trajectoryIndex=0,
        )
        coaxialSummary = SurgicalVision3D_PlannerLogic().computeCoaxialPlanFromTrajectory(
            trajectory,
            "PushThrough",
            10.0,
            5.0,
        )
        self.assertEqual(coaxialSummary.technique, "PushThrough")
        self.assertTrue(np.allclose(np.array(coaxialSummary.navigationTargetRAS), np.array([0.0, 0.0, -5.0]), atol=1e-6))

    def test_recolor_and_restore_include_last_array_value(self):
        signedDistances = vtk.vtkDoubleArray()
        signedDistances.SetName(SIGNED_DISTANCE_ARRAY_NAME)
        for value in (-9.0, -6.0, -3.0, 2.0):
            signedDistances.InsertNextValue(value)

        backup = vtk.vtkDoubleArray()
        backup.DeepCopy(signedDistances)

        bucketCount = SurgicalVision3D_PlannerLogic.recolorSignedDistanceArray(signedDistances, (-10.0, -5.0, -2.0))
        self.assertEqual(bucketCount, 4)
        self.assertEqual(signedDistances.GetValue(0), 1.0)
        self.assertEqual(signedDistances.GetValue(1), 1.0)
        self.assertEqual(signedDistances.GetValue(2), 2.0)
        self.assertEqual(signedDistances.GetValue(3), 3.0)

        SurgicalVision3D_PlannerLogic.restoreSignedDistanceArray(signedDistances, backup)
        self.assertEqual(signedDistances.GetValue(0), -9.0)
        self.assertEqual(signedDistances.GetValue(1), -6.0)
        self.assertEqual(signedDistances.GetValue(2), -3.0)
        self.assertEqual(signedDistances.GetValue(3), 2.0)

    def test_parameter_node_initialization_and_restore(self):
        logic = SurgicalVision3D_PlannerLogic()
        parameterNode = logic.getParameterNode()

        self.assertTrue(parameterNode.createTrajectoryLinesOnPlacement)
        self.assertTrue(parameterNode.placeMultipleControlPoints)
        self.assertEqual(str(parameterNode.trajectoryPlanningMode), "Single")
        self.assertEqual(int(parameterNode.derivedTrajectoryCount), 4)
        self.assertAlmostEqual(float(parameterNode.derivedTrajectoryRadiusMm), 10.0, places=6)
        self.assertAlmostEqual(float(parameterNode.derivedTrajectoryAngleOffsetDeg), 0.0, places=6)
        self.assertTrue(bool(parameterNode.includeMasterTrajectoryInArray))
        self.assertAlmostEqual(float(parameterNode.coaxialSpareMm), 5.0, places=6)
        self.assertEqual(SurgicalVision3D_PlannerLogic.deserializeNodeIDs(parameterNode.generatedProbeNodeIDs), [])

        probeSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        parameterNode.referenceProbeSegmentation = probeSegmentationNode

        restoredParameterNode = logic.getParameterNode()
        self.assertIsNotNone(restoredParameterNode.referenceProbeSegmentation)
        self.assertEqual(restoredParameterNode.referenceProbeSegmentation.GetID(), probeSegmentationNode.GetID())
