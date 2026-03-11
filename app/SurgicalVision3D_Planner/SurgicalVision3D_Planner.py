from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import subprocess
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Sequence
from urllib.parse import unquote
from xml.etree import ElementTree as ET

import numpy as np
import vtk

import ctk
import qt
import slicer
from slicer import (
    vtkMRMLMarkupsFiducialNode,
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
BEGINNER_WORKFLOW_BANNER_TEXT = (
    "Beginner mode: import a case, define one master trajectory, validate it, assess MAM, then lock and derive coaxial guidance."
)
BEGINNER_MAM_COLOR_NODE_NAME = "SV3D Beginner MAM Colors"
BEGINNER_TRAJECTORY_DISTANCE_SAMPLE_COUNT = 121
BEGINNER_AUTO_ADAPT_PLACEHOLDER_TEXT = "Auto-adjust endpoint (later milestone)"


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    normalized = np.array(vector, dtype=float)
    length = float(np.linalg.norm(normalized))
    if length <= 1e-8:
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


@dataclass
class GeometryCatalogEntry:
    geometryId: str
    displayName: str
    templateRelativePath: str
    activeElementLengthMm: float


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


@dataclass
class CoaxialPlanSummary:
    technique: str
    activeElementLengthMm: float
    navigationTargetRAS: tuple[float, float, float]
    masterEntryPointRAS: tuple[float, float, float]
    masterTargetPointRAS: tuple[float, float, float]
    notes: str = ""


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
Phase 1 ablation planning prototype:
1. Generate trajectories from endpoint control-point pairs.
2. Place translated probe segmentations along trajectories.
3. Merge translated probes into one ablation zone.
4. Evaluate tumor-vs-ablation signed margins using signed closest-point distances.
""")
        self.parent.acknowledgementText = _("""
Legacy AblationPlanner workflow was refactored into this module while preserving scripted-module patterns.
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

    createTrajectoryLinesOnPlacement: bool = True
    clearPreviousGeneratedProbes: bool = True
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

    generatedProbeNodeIDs: str = "[]"
    generatedTrajectoryLineIDs: str = "[]"
    trajectoryValidationSummaryJson: str = "{}"
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
        self._importCaseFolderButton = None
        self._openSegmentEditorButton = None
        self._geometryComboBox = None
        self._mamSpinBox = None
        self._validateTrajectoryButton = None
        self._trajectoryValidationStatusLabel = None
        self._segmentEditorStatusLabel = None
        self._marginAssessmentStatusLabel = None
        self._lockMasterPlanButton = None
        self._resetMasterPlanButton = None
        self._coaxialTechniqueComboBox = None
        self._computeCoaxialPlanButton = None
        self._coaxialStatusLabel = None
        self._autoAdaptPlaceholderButton = None

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

        self.logic = SurgicalVision3D_PlannerLogic()
        self.logic.ensureReferenceProbeTemplatesLoaded()
        self._populateSampleCaseComboBox()
        self._populateBeginnerGeometryComboBox()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

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
        if self._mamSpinBox:
            self._mamSpinBox.connect("valueChanged(double)", self.onMamValueChanged)
        if self._validateTrajectoryButton:
            self._validateTrajectoryButton.connect("clicked(bool)", self.onValidateTrajectoryButton)
        if self._lockMasterPlanButton:
            self._lockMasterPlanButton.connect("clicked(bool)", self.onLockMasterPlanButton)
        if self._resetMasterPlanButton:
            self._resetMasterPlanButton.connect("clicked(bool)", self.onResetMasterPlanButton)
        if self._coaxialTechniqueComboBox:
            self._coaxialTechniqueComboBox.connect("currentIndexChanged(int)", self.onCoaxialTechniqueChanged)
        if self._computeCoaxialPlanButton:
            self._computeCoaxialPlanButton.connect("clicked(bool)", self.onComputeCoaxialPlanButton)

        self.initializeParameterNode()

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

        buttonRowLayout = qt.QHBoxLayout()
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
        gitFormLayout.addRow(buttonRowLayout)

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
        trajectoryLayout.addRow("Endpoint markups:", self.ui.endpointsMarkupsSelector)
        trajectoryLayout.addRow("Instruction:", trajectoryInstructionLabel)
        trajectoryLayout.addRow("Preview:", self._buildButtonRowWidget(self.ui.createTrajectoryLinesButton))

        validationSection = ctk.ctkCollapsibleButton()
        validationSection.text = "Step 4 - Validate Trajectory"
        validationLayout = qt.QFormLayout(validationSection)
        self._validateTrajectoryButton = qt.QPushButton("Validate Against Critical Structures")
        self._trajectoryValidationStatusLabel = self._createStatusLabel("Trajectory not validated.")
        self._autoAdaptPlaceholderButton = qt.QPushButton(BEGINNER_AUTO_ADAPT_PLACEHOLDER_TEXT)
        self._autoAdaptPlaceholderButton.enabled = False
        validationLayout.addRow("Validation:", self._validateTrajectoryButton)
        validationLayout.addRow("Status:", self._trajectoryValidationStatusLabel)
        validationLayout.addRow("Later:", self._autoAdaptPlaceholderButton)

        applicatorSection = ctk.ctkCollapsibleButton()
        applicatorSection.text = "Step 5 - Single Applicator Plan"
        applicatorLayout = qt.QFormLayout(applicatorSection)
        self._geometryComboBox = qt.QComboBox()
        self._mamSpinBox = ctk.ctkDoubleSpinBox()
        self._mamSpinBox.setMinimum(0.0)
        self._mamSpinBox.setMaximum(50.0)
        self._mamSpinBox.setDecimals(1)
        self._mamSpinBox.setSingleStep(0.5)
        self._mamSpinBox.setValue(10.0)
        self._marginAssessmentStatusLabel = self._createStatusLabel("MAM assessment not run.")
        applicatorLayout.addRow("Ablation geometry:", self._geometryComboBox)
        applicatorLayout.addRow("MAM (mm):", self._mamSpinBox)
        applicatorLayout.addRow(
            "Planning:",
            self._buildButtonRowWidget(self.ui.placeProbesButton, self.ui.mergeTranslatedProbesButton, self.ui.evaluateMarginsButton),
        )
        applicatorLayout.addRow("Status:", self._marginAssessmentStatusLabel)

        lockSection = ctk.ctkCollapsibleButton()
        lockSection.text = "Step 6 - Lock + Coaxial Plan"
        lockLayout = qt.QFormLayout(lockSection)
        self._lockMasterPlanButton = qt.QPushButton("Lock Validated Master Plan")
        self._resetMasterPlanButton = qt.QPushButton("Reset Lock")
        self._coaxialTechniqueComboBox = qt.QComboBox()
        self._coaxialTechniqueComboBox.addItem("PullBack")
        self._coaxialTechniqueComboBox.addItem("PushThrough")
        self._computeCoaxialPlanButton = qt.QPushButton("Compute Coaxial Plan")
        self._coaxialStatusLabel = self._createStatusLabel("Coaxial plan not computed.")
        lockLayout.addRow("Locking:", self._buildButtonRowWidget(self._lockMasterPlanButton, self._resetMasterPlanButton))
        lockLayout.addRow("Technique:", self._coaxialTechniqueComboBox)
        lockLayout.addRow("Coaxial plan:", self._computeCoaxialPlanButton)
        lockLayout.addRow("Status:", self._coaxialStatusLabel)

        self._beginnerWorkflowSections = [
            importSection,
            segmentationSection,
            trajectorySection,
            validationSection,
            applicatorSection,
            lockSection,
        ]
        for section in reversed(self._beginnerWorkflowSections):
            uiWidget.layout().insertWidget(0, section)

    def _populateBeginnerGeometryComboBox(self) -> None:
        if not self.logic or not self._geometryComboBox:
            return
        currentGeometryId = str(self._geometryComboBox.currentData() or "")
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
            code, _, stderrText = self._runGitCommand("add", "-A")
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
        self._renameGenericFiducialsNode(node, "endpoints")

    def onNativeFiducialsChanged(self, node) -> None:
        self._renameGenericFiducialsNode(node, "NativeFiducials")

    def onRegisteredFiducialsChanged(self, node) -> None:
        self._renameGenericFiducialsNode(node, "RegisteredFiducials")

    @staticmethod
    def _configureNodeSelectorShowHidden(selector, showHidden: bool) -> None:
        if not selector:
            return
        if hasattr(selector, "setShowHidden"):
            selector.setShowHidden(showHidden)
            return
        if hasattr(selector, "showHidden"):
            selector.showHidden = bool(showHidden)

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

        if self._coaxialTechniqueComboBox:
            targetTechnique = str(self._parameterNode.coaxialTechnique or "PullBack")
            targetIndex = self._coaxialTechniqueComboBox.findText(targetTechnique)
            if targetIndex >= 0 and targetIndex != self._coaxialTechniqueComboBox.currentIndex:
                self._coaxialTechniqueComboBox.blockSignals(True)
                self._coaxialTechniqueComboBox.currentIndex = targetIndex
                self._coaxialTechniqueComboBox.blockSignals(False)

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

        trajectorySummary = self._jsonSummary(self._parameterNode.trajectoryValidationSummaryJson)
        if self._trajectoryValidationStatusLabel:
            self._trajectoryValidationStatusLabel.text = str(
                trajectorySummary.get("StatusText", "Trajectory not validated.")
            )
        mamSummary = self._jsonSummary(self._parameterNode.mamAssessmentSummaryJson)
        if self._marginAssessmentStatusLabel:
            self._marginAssessmentStatusLabel.text = str(
                mamSummary.get("StatusText", "MAM assessment not run.")
            )
        coaxialSummary = self._jsonSummary(self._parameterNode.coaxialPlanSummaryJson)
        if self._coaxialStatusLabel:
            self._coaxialStatusLabel.text = str(
                coaxialSummary.get("notes", "Coaxial plan not computed.")
            )

    @staticmethod
    def _setWidgetVisible(widget, visible: bool) -> None:
        if widget and hasattr(widget, "setVisible"):
            widget.setVisible(bool(visible))

    def _setUiWidgetsVisible(self, widgetNames: Sequence[str], visible: bool) -> None:
        for widgetName in widgetNames:
            self._setWidgetVisible(getattr(self.ui, widgetName, None), visible)

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
                "placed on each trajectory during 'Place Probes'. STL templates in Resources/geometries are auto-loaded "
                "into this selector. The template is expected to be oriented along local -Z."
            ),
            "endpointsMarkupsSelector": (
                "Select trajectory endpoints as ordered entry/target pairs with an even number of control points: "
                "entry1,target1,entry2,target2,..."
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
            "placeProbesButton": "Place probe instances along each endpoint-pair trajectory.",
            "createTrajectoryLinesButton": "Create or refresh line markups from endpoint pairs.",
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
            "exportBundleButton": "Write a deterministic export bundle (JSON + CSV + manifest) without mutating the plan.",
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
            "endpointsMarkupsSelector": "Choose exactly two points: entry first, applicator endpoint second.",
            "tumorSegmentationSelector": "Choose the target tumor segmentation for margin evaluation.",
            "riskStructuresSegmentationSelector": "Choose the segmentation that contains the critical structures to avoid.",
            "placeProbesButton": "Create probe instances from endpoint pairs.",
            "mergeTranslatedProbesButton": "Union the placed probes into one combined ablation zone.",
            "evaluateMarginsButton": "Compute signed margins between the target tumor and the combined ablation zone.",
            "createTrajectoryLinesButton": "Preview the current single master trajectory as a line.",
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
        if self._mamSpinBox:
            self._mamSpinBox.setToolTip("Minimal Ablative Margin in mm. Default is 10 mm.")
        if self._validateTrajectoryButton:
            self._validateTrajectoryButton.setToolTip("Check whether the master trajectory intersects any critical-structure segment.")
        if self._lockMasterPlanButton:
            self._lockMasterPlanButton.setToolTip("Lock the current validated master trajectory and save its snapshot.")
        if self._resetMasterPlanButton:
            self._resetMasterPlanButton.setToolTip("Unlock the master plan and clear the saved lock/coaxial outputs.")
        if self._coaxialTechniqueComboBox:
            self._coaxialTechniqueComboBox.setToolTip("Choose how the coaxial sheath and applicator are coordinated.")
        if self._computeCoaxialPlanButton:
            self._computeCoaxialPlanButton.setToolTip("Derive the coaxial navigation target for the selected technique.")

    def cleanup(self) -> None:
        self.removeObservers()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            if self._parameterNodeGuiTag is not None:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateButtonStates)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        if not self.logic:
            return
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

    def _preloadNamedSceneInputs(self, parameterNode: SurgicalVision3D_PlannerParameterNode) -> None:
        if not self.logic:
            return

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

        nativeFiducials = self.logic.findFirstNodeByClassAndPreferredNames(
            "vtkMRMLMarkupsFiducialNode",
            DEFAULT_NATIVE_FIDUCIAL_NAMES,
        )
        if nativeFiducials and (
            not parameterNode.nativeFiducials
            or self.logic.isGenericDefaultFiducialsNodeName(parameterNode.nativeFiducials.GetName())
        ):
            parameterNode.nativeFiducials = nativeFiducials

        registeredFiducials = self.logic.findFirstNodeByClassAndPreferredNames(
            "vtkMRMLMarkupsFiducialNode",
            DEFAULT_REGISTERED_FIDUCIAL_NAMES,
        )
        if registeredFiducials and (
            not parameterNode.registeredFiducials
            or self.logic.isGenericDefaultFiducialsNodeName(parameterNode.registeredFiducials.GetName())
        ):
            parameterNode.registeredFiducials = registeredFiducials

    def setParameterNode(self, inputParameterNode: SurgicalVision3D_PlannerParameterNode | None) -> None:
        if self._parameterNode:
            if self._parameterNodeGuiTag is not None:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateButtonStates)

        self._parameterNode = inputParameterNode

        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateButtonStates)
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
        if self._parameterNode.criticalStructuresSegmentation and self._parameterNode.riskStructuresSegmentation is not self._parameterNode.criticalStructuresSegmentation:
            self._parameterNode.riskStructuresSegmentation = self._parameterNode.criticalStructuresSegmentation
        elif self._parameterNode.riskStructuresSegmentation and not self._parameterNode.criticalStructuresSegmentation:
            self._parameterNode.criticalStructuresSegmentation = self._parameterNode.riskStructuresSegmentation

        for nodeFieldName in (
            "criticalStructuresSegmentation",
            "combinedProbeSegmentation",
            "outputMarginModel",
            "resultTable",
            "tumorTransform",
            "coaxialNavigationTarget",
            "riskStructuresSegmentation",
            "trajectorySummaryTable",
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
        ):
            node = getattr(self._parameterNode, nodeFieldName)
            if node and not slicer.mrmlScene.IsNodePresent(node):
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

    def _clearOwnedBeginnerOutputs(self, clearReferences: bool = False, unlockPlan: bool = False) -> None:
        if not self.logic or not self._parameterNode:
            return

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

        self._parameterNode.trajectoryValidationSummaryJson = "{}"
        self._parameterNode.mamAssessmentSummaryJson = "{}"
        self._parameterNode.masterPlanSnapshotJson = "{}"
        self._parameterNode.coaxialPlanSummaryJson = "{}"
        if unlockPlan:
            self._parameterNode.masterTrajectoryLocked = False
            self._setMasterTrajectoryLocked(False)

        if clearReferences:
            if self._trajectoryValidationStatusLabel:
                self._trajectoryValidationStatusLabel.text = "Trajectory not validated."
            if self._marginAssessmentStatusLabel:
                self._marginAssessmentStatusLabel.text = "MAM assessment not run."
            if self._coaxialStatusLabel:
                self._coaxialStatusLabel.text = "Coaxial plan not computed."

    def _clearOwnedDerivedOutputs(self, clearReferences: bool = False) -> None:
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
        self._clearOwnedSafetyOutputs(clearReferences=clearReferences)
        self._clearOwnedCoordinationOutputs(clearReferences=clearReferences)
        self._clearOwnedCohortOutputs(clearReferences=clearReferences)
        self._clearOwnedBeginnerOutputs(clearReferences=clearReferences)

    def _updateButtonStates(self, caller=None, event=None) -> None:
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
                self._importCaseFolderButton,
                self._openSegmentEditorButton,
                self._validateTrajectoryButton,
                self._lockMasterPlanButton,
                self._resetMasterPlanButton,
                self._computeCoaxialPlanButton,
            ):
                if widget:
                    widget.enabled = False
            self._updateGitDashboardButtonStates()
            return

        self._reconcileParameterNodeState()
        self._syncBeginnerWidgetsFromParameterNode()

        hasProbeAndEndpoints = bool(self._parameterNode.referenceProbeSegmentation and self._parameterNode.endpointsMarkups)
        generatedProbeNodeIDs = self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs) if self.logic else []
        hasGeneratedProbes = len(generatedProbeNodeIDs) > 0
        hasSingleGeneratedProbe = len(generatedProbeNodeIDs) == 1
        hasCombinedProbe = self._parameterNode.combinedProbeSegmentation is not None
        hasTumor = self._parameterNode.tumorSegmentation is not None
        hasCriticalStructures = self._parameterNode.criticalStructuresSegmentation is not None
        hasRegistrationInputs = bool(self._parameterNode.tumorSegmentation and self._parameterNode.nativeFiducials and self._parameterNode.registeredFiducials)
        hasMarginModel = self._parameterNode.outputMarginModel is not None
        hasTumorTransform = bool(self._parameterNode.tumorSegmentation and self._parameterNode.tumorSegmentation.GetTransformNodeID())
        endpointControlPointCount = int(self._parameterNode.endpointsMarkups.GetNumberOfControlPoints()) if self._parameterNode.endpointsMarkups else 0
        hasEvenEndpointPairs = endpointControlPointCount >= 2 and endpointControlPointCount % 2 == 0
        hasSingleMasterTrajectory = endpointControlPointCount == 2
        isLocked = bool(self._parameterNode.masterTrajectoryLocked)
        trajectoryValidationSummary = self._jsonSummary(self._parameterNode.trajectoryValidationSummaryJson)
        mamAssessmentSummary = self._jsonSummary(self._parameterNode.mamAssessmentSummaryJson)
        trajectoryValidationPass = bool(trajectoryValidationSummary.get("TrajectoryPass", False))
        mamValidationPass = bool(mamAssessmentSummary.get("MamPass", False))

        self.ui.placeProbesButton.enabled = hasProbeAndEndpoints and hasSingleMasterTrajectory and not isLocked
        self.ui.createTrajectoryLinesButton.enabled = hasSingleMasterTrajectory and not isLocked
        self.ui.mergeTranslatedProbesButton.enabled = hasGeneratedProbes and (hasSingleGeneratedProbe or not BEGINNER_WORKFLOW_MODE) and not isLocked
        self.ui.registerTumorButton.enabled = hasRegistrationInputs
        self.ui.hardenTumorTransformButton.enabled = hasTumorTransform
        self.ui.evaluateMarginsButton.enabled = hasTumor and hasSingleMasterTrajectory and (hasCombinedProbe or hasGeneratedProbes) and not isLocked
        self.ui.recolorMarginsButton.enabled = hasMarginModel
        self.ui.resetMarginColorsButton.enabled = hasMarginModel
        self.ui.evaluateProbeCoordinationButton.enabled = hasEvenEndpointPairs
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
        self.ui.exportBundleButton.enabled = bool(exportBaseName.strip()) and (not scenarioRequired or bool(selectedScenarioID.strip()))
        self.ui.runCohortEvaluationButton.enabled = bool(cohortStudyDefinitionPath.strip())
        if hasattr(self.ui, "loadSampleCaseButton"):
            self.ui.loadSampleCaseButton.enabled = bool(self._selectedSampleCaseScenePath())
        if hasattr(self.ui, "generateReproducibilityPackageButton"):
            self.ui.generateReproducibilityPackageButton.enabled = bool(packageBaseName.strip())
        if self._browseCaseFolderButton:
            self._browseCaseFolderButton.enabled = True
        if self._importCaseFolderButton:
            self._importCaseFolderButton.enabled = bool(str(self._caseFolderLineEdit.text).strip()) if self._caseFolderLineEdit else False
        if self._openSegmentEditorButton:
            self._openSegmentEditorButton.enabled = True
        if self._geometryComboBox:
            self._geometryComboBox.enabled = not isLocked
        if self._mamSpinBox:
            self._mamSpinBox.enabled = not isLocked
        if hasattr(self.ui, "endpointsMarkupsSelector"):
            self.ui.endpointsMarkupsSelector.enabled = not isLocked
        if self._validateTrajectoryButton:
            self._validateTrajectoryButton.enabled = hasSingleMasterTrajectory and hasCriticalStructures and not isLocked
        if self._lockMasterPlanButton:
            self._lockMasterPlanButton.enabled = (trajectoryValidationPass and mamValidationPass and not isLocked)
        if self._resetMasterPlanButton:
            self._resetMasterPlanButton.enabled = isLocked
        if self._computeCoaxialPlanButton:
            self._computeCoaxialPlanButton.enabled = isLocked and bool(str(self._parameterNode.masterPlanSnapshotJson or "{}").strip())
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
        return self.logic.extractSingleTrajectoryFromMarkups(self._parameterNode.endpointsMarkups)

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

    def onImportCaseFolderButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to import case folder."), waitCursor=True):
            if not self.logic:
                raise RuntimeError("Module logic is not initialized.")
            caseFolderPath = str(self._caseFolderLineEdit.text).strip() if self._caseFolderLineEdit else ""
            if not caseFolderPath:
                raise ValueError("Select a case folder before importing.")
            self.logic.loadCaseFolder(caseFolderPath)
            self.initializeParameterNode()
            if self._parameterNode:
                self._parameterNode.caseFolderPath = caseFolderPath
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)
            if self._segmentEditorStatusLabel:
                self._segmentEditorStatusLabel.text = "Case imported. Review or refine segments in Segment Editor if needed."
            self._updateButtonStates()

    def onOpenSegmentEditorButton(self) -> None:
        slicer.util.selectModule("SegmentEditor")
        if self._segmentEditorStatusLabel:
            self._segmentEditorStatusLabel.text = "Segment Editor opened. Return here after updating tumor and critical structures."

    def onSelectedGeometryChanged(self, _index=None) -> None:
        if not self.logic or not self._parameterNode or not self._geometryComboBox:
            return
        if self._parameterNode.masterTrajectoryLocked:
            return
        selectedGeometryId = str(self._geometryComboBox.currentData() or "")
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
        self._clearOwnedDerivedOutputs(clearReferences=True)
        self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)
        self._updateButtonStates()

    def onMamValueChanged(self, newValue: float) -> None:
        if not self._parameterNode:
            return
        self._parameterNode.mamMm = float(newValue)
        if self._parameterNode.outputMarginModel:
            self._parameterNode.mamAssessmentSummaryJson = "{}"
            if self._marginAssessmentStatusLabel:
                self._marginAssessmentStatusLabel.text = "MAM changed. Re-run MAM assessment to refresh colors and pass/fail."
        self._updateButtonStates()

    def onCoaxialTechniqueChanged(self, _index=None) -> None:
        if not self._parameterNode or not self._coaxialTechniqueComboBox:
            return
        self._parameterNode.coaxialTechnique = str(self._coaxialTechniqueComboBox.currentText or "PullBack")
        self._updateButtonStates()

    def onValidateTrajectoryButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to validate the master trajectory."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            trajectory = self._singleMasterTrajectory()
            validationRows, validationSummary = self.logic.evaluateMasterTrajectoryAgainstCriticalStructures(
                trajectory,
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
                self._trajectoryValidationStatusLabel.text = str(validationSummary.get("StatusText", "Trajectory validation completed."))
            self._updateButtonStates()

    def onLockMasterPlanButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to lock the master plan."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            trajectoryValidationSummary = self._jsonSummary(self._parameterNode.trajectoryValidationSummaryJson)
            mamAssessmentSummary = self._jsonSummary(self._parameterNode.mamAssessmentSummaryJson)
            if not bool(trajectoryValidationSummary.get("TrajectoryPass", False)):
                raise ValueError("Validate the master trajectory successfully before locking the plan.")
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
            snapshot = LockedMasterPlanSnapshot(
                geometryId=geometryEntry.geometryId,
                geometryDisplayName=geometryEntry.displayName,
                mamMm=float(self._parameterNode.mamMm),
                entryPointRAS=tuple(float(value) for value in trajectory.entryPointRAS),
                targetPointRAS=tuple(float(value) for value in trajectory.targetPointRAS),
                directionVector=tuple(float(value) for value in trajectory.directionVector),
                trajectoryLengthMm=float(trajectory.lengthMm),
                activeElementLengthMm=float(geometryEntry.activeElementLengthMm),
                trajectoryValidationPass=True,
                marginValidationPass=True,
                lockedAtISO=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                tumorSegmentationID=self._parameterNode.tumorSegmentation.GetID() if self._parameterNode.tumorSegmentation else "",
                tumorSegmentID=tumorSegmentID,
                tumorSegmentName=tumorSegmentName,
                endpointsMarkupsID=self._parameterNode.endpointsMarkups.GetID() if self._parameterNode.endpointsMarkups else "",
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
                self._coaxialStatusLabel.text = "Master plan locked. Choose a technique and compute the coaxial plan."
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
            geometryEntry = self._selectedGeometryEntry()
            activeElementLengthMm = float(snapshotValues.get("activeElementLengthMm", geometryEntry.activeElementLengthMm if geometryEntry else 0.0))
            trajectory = ProbeTrajectory(
                entryPointRAS=tuple(float(value) for value in snapshotValues.get("entryPointRAS", (0.0, 0.0, 0.0))),
                targetPointRAS=tuple(float(value) for value in snapshotValues.get("targetPointRAS", (0.0, 0.0, 0.0))),
                directionVector=tuple(float(value) for value in snapshotValues.get("directionVector", (0.0, 0.0, -1.0))),
                lengthMm=float(snapshotValues.get("trajectoryLengthMm", 0.0)),
                trajectoryIndex=0,
            )
            coaxialSummary = self.logic.computeCoaxialPlanFromTrajectory(
                trajectory,
                str(self._parameterNode.coaxialTechnique or "PullBack"),
                activeElementLengthMm,
            )
            coaxialTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                COAXIAL_PLAN_TABLE_NODE_NAME,
                GENERATED_COAXIAL_PLAN_TABLE_ATTRIBUTE,
                self._parameterNode.coaxialPlanTable,
            )
            self.logic.populateKeyValueTable(coaxialTable, asdict(coaxialSummary))
            self._parameterNode.coaxialPlanTable = coaxialTable
            self._parameterNode.coaxialPlanSummaryJson = json.dumps(asdict(coaxialSummary), sort_keys=True)
            coaxialTargetNode = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLMarkupsFiducialNode",
                COAXIAL_NAVIGATION_TARGET_NODE_NAME,
                GENERATED_COAXIAL_TARGET_ATTRIBUTE,
                self._parameterNode.coaxialNavigationTarget,
            )
            slicer.util.updateMarkupsControlPointsFromArray(
                coaxialTargetNode,
                np.array([coaxialSummary.navigationTargetRAS], dtype=float),
            )
            coaxialTargetNode.SetLocked(True)
            self._parameterNode.coaxialNavigationTarget = coaxialTargetNode
            self.logic.createOrUpdateCoaxialLine(
                trajectory.entryPointRAS,
                coaxialSummary.navigationTargetRAS,
            )
            if self._coaxialStatusLabel:
                self._coaxialStatusLabel.text = str(coaxialSummary.notes)
            self._updateButtonStates()

    def onLoadSampleCaseButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to load sample case."), waitCursor=True):
            if not self.logic:
                raise RuntimeError("Module logic is not initialized.")
            selectedScenePath = self._selectedSampleCaseScenePath().strip()
            if not selectedScenePath:
                raise ValueError("Select a sample case scene before loading.")
            self.logic.loadBundledSampleScene(selectedScenePath)
            # Scene load triggers module-scene events, but run one more initialization pass to refresh selectors.
            self.initializeParameterNode()
            if self._parameterNode:
                self._parameterNode.caseFolderPath = ""
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)
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
                    "Place one entry point and one applicator endpoint."
                )
            if controlPointCount % 2 != 0:
                raise ValueError(
                    f"Endpoint markups has {controlPointCount} control points. Add one more point to complete entry/target pairs."
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
            trajectories = self.logic.extractTrajectoriesFromMarkups(self._parameterNode.endpointsMarkups, strictEven=True)
            if self._parameterNode.clearPreviousGeneratedProbes:
                self.logic.removeGeneratedProbeNodes()
                self.logic.removeGeneratedTrajectoryLines()
                self._clearOwnedDerivedOutputs(clearReferences=True)
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
            self._clearOwnedDerivedOutputs(clearReferences=True)
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)

            if self._parameterNode.createTrajectoryLinesOnPlacement:
                generatedLineNodeIDs = self.logic.createTrajectoryLines(trajectories, clearExisting=True)
                self._parameterNode.generatedTrajectoryLineIDs = self.logic.serializeNodeIDs(generatedLineNodeIDs)

            trajectorySummaryTable = self.logic.createOrReuseOwnedOutputNode(
                "vtkMRMLTableNode",
                TRAJECTORY_SUMMARY_TABLE_NODE_NAME,
                GENERATED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
                self._parameterNode.trajectorySummaryTable,
            )
            trajectoryMetrics = self.logic.computeTrajectoryMetrics(trajectories)
            self.logic.populateTrajectorySummaryTable(trajectorySummaryTable, trajectoryMetrics)
            self._parameterNode.trajectorySummaryTable = trajectorySummaryTable

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
                    "Place one entry point and one applicator endpoint."
                )
            if controlPointCount % 2 != 0:
                raise ValueError(
                    f"Endpoint markups has {controlPointCount} control points. Add one more point to complete entry/target pairs."
                )
            trajectories = self.logic.extractTrajectoriesFromMarkups(self._parameterNode.endpointsMarkups, strictEven=True)
            generatedLineNodeIDs = self.logic.createTrajectoryLines(trajectories, clearExisting=True)
            self._parameterNode.generatedTrajectoryLineIDs = self.logic.serializeNodeIDs(generatedLineNodeIDs)
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)
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
            self._clearOwnedBeginnerOutputs(clearReferences=True, unlockPlan=True)
            self._updateButtonStates()

    def onRegisterTumorButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to register tumor fiducials."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            transformNode = self.logic.registerTumorToFiducials(
                self._parameterNode.tumorSegmentation,
                self._parameterNode.nativeFiducials,
                self._parameterNode.registeredFiducials,
                self._parameterNode.tumorTransform,
            )
            self._parameterNode.tumorTransform = transformNode
            if self._segmentEditorStatusLabel:
                self._segmentEditorStatusLabel.text = "Fiducial registration applied to the tumor segmentation."
            self._updateButtonStates()

    def onHardenTumorTransformButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to harden tumor transform."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            self.logic.hardenTumorTransform(self._parameterNode.tumorSegmentation)
            if self._segmentEditorStatusLabel:
                self._segmentEditorStatusLabel.text = "Tumor registration hardened into the segmentation geometry."
            self._updateButtonStates()

    def onRiskStructuresSegmentationChanged(self, _node=None) -> None:
        if not self.logic or not self._parameterNode:
            return

        self._parameterNode.criticalStructuresSegmentation = self._parameterNode.riskStructuresSegmentation
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

            trajectories = self.logic.extractTrajectoriesFromMarkups(self._parameterNode.endpointsMarkups, strictEven=True)
            if len(trajectories) == 0:
                raise ValueError("No valid trajectories are available for probe coordination evaluation.")
            self._evaluateAndPublishProbeCoordination(trajectories, updateStatusLabel=True)
            self._updateButtonStates()

    def onEvaluateMarginsButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to evaluate margins."), waitCursor=True):
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

            outputMarginModel, resultTable, summary = self.logic.evaluateMargins(
                self._parameterNode.tumorSegmentation,
                probeSegmentation,
                self._parameterNode.outputMarginModel,
                self._parameterNode.resultTable,
            )
            self._parameterNode.outputMarginModel = outputMarginModel
            self._parameterNode.resultTable = resultTable
            logging.info("Margin summary: %s", summary)

            signedMarginValues = self.logic.getSignedMarginValues(outputMarginModel)
            mamAssessmentSummary = self.logic.applyBeginnerMamColoring(outputMarginModel, float(self._parameterNode.mamMm))
            self._parameterNode.mamAssessmentSummaryJson = json.dumps(mamAssessmentSummary, sort_keys=True)
            if self._marginAssessmentStatusLabel:
                self._marginAssessmentStatusLabel.text = str(mamAssessmentSummary.get("StatusText", "MAM assessment completed."))
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

            structureSafetySummaryRows, structureSafetyThresholdRows = self.logic.evaluateStructureSafety(
                self._parameterNode.criticalStructuresSegmentation,
                probeSegmentation,
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

        exportMode = str(self.ui.exportModeComboBox.currentText) if hasattr(self.ui, "exportModeComboBox") else str(self._parameterNode.exportMode)
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
                self.ui.exportStatusLabel.text = (
                    f"Exported {int(exportResult['fileCount'])} files to {str(exportResult['bundlePath'])}"
                )
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
        tablesDirectory.mkdir(parents=True, exist_ok=True)
        provenanceDirectory.mkdir(parents=True, exist_ok=True)

        currentPlanSummary, tableExports = self.collectCurrentPlanExportData(parameterNode, exportConfig)
        exportedFiles: list[str] = []

        planSummaryPath = bundlePath / "plan_summary.json"
        self.exportStructuredSummaryToJson(planSummaryPath, currentPlanSummary)
        exportedFiles.append(str(planSummaryPath.relative_to(bundlePath)))

        selectedScenarioSummary: dict[str, Any] = {}
        shouldIncludeSelectedScenario = bool(exportConfig.includeSelectedScenario or exportConfig.exportMode == "SelectedScenario")
        if shouldIncludeSelectedScenario:
            selectedScenarioSummary = self.collectScenarioExportData(exportConfig.selectedExportScenarioID)
            scenarioSummaryPath = bundlePath / "scenario_summary.json"
            self.exportStructuredSummaryToJson(scenarioSummaryPath, selectedScenarioSummary)
            exportedFiles.append(str(scenarioSummaryPath.relative_to(bundlePath)))

        for tableFileName, tableNode in tableExports:
            outputCsvPath = tablesDirectory / tableFileName
            self.exportTableNodeToCsv(tableNode, outputCsvPath)
            exportedFiles.append(str(outputCsvPath.relative_to(bundlePath)))

        scenarioRegistryTable = self._findFirstTableNodeByName("SV3D Scenario Registry")
        if scenarioRegistryTable:
            scenarioRegistryJsonPath = provenanceDirectory / "scenario_registry.json"
            self.exportStructuredSummaryToJson(scenarioRegistryJsonPath, self._tableNodeToDictionaries(scenarioRegistryTable))
            exportedFiles.append(str(scenarioRegistryJsonPath.relative_to(bundlePath)))

        recommendationSummaryTable = self._findFirstTableNodeByName("SV3D Feasible Candidate Recommendation")
        if recommendationSummaryTable:
            recommendationJsonPath = provenanceDirectory / "recommendation_summary.json"
            self.exportStructuredSummaryToJson(recommendationJsonPath, self._tableNodeToDictionaries(recommendationSummaryTable))
            exportedFiles.append(str(recommendationJsonPath.relative_to(bundlePath)))

        manifest = self.buildPlanExportManifest(
            parameterNode=parameterNode,
            exportConfig=exportConfig,
            exportSequence=exportSequence,
            filesExported=exportedFiles,
            selectedScenarioSummary=selectedScenarioSummary,
        )
        manifestPath = bundlePath / "manifest.json"
        exportedFiles.insert(0, str(manifestPath.relative_to(bundlePath)))
        manifest.filesExported = list(exportedFiles)
        self.exportStructuredSummaryToJson(manifestPath, asdict(manifest))

        return {
            "manifest": manifest,
            "bundlePath": str(bundlePath),
            "exportDirectory": str(exportRoot),
            "exportSequence": int(exportSequence),
            "fileCount": int(len(exportedFiles)),
            "status": "Success",
            "selectedScenarioName": str(selectedScenarioSummary.get("ScenarioName", "")),
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

        slicer.mrmlScene.Clear(0)
        loadedAny = False

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

        SurgicalVision3D_PlannerLogic().ensureReferenceProbeTemplatesLoaded()
        return loadedAny

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
        geometriesDirectory = self._resolveResourcePath("Resources/geometries")
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

    def ensureReferenceProbeTemplatesLoaded(self) -> list[vtkMRMLSegmentationNode]:
        templatePaths = self.discoverReferenceProbeTemplateFiles()
        if len(templatePaths) == 0:
            return []

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
                loadedTemplates.append(existingTemplate)
                continue

            closedSurface = self._loadClosedSurfaceFromStl(templatePath)
            if not closedSurface or closedSurface.GetNumberOfPoints() <= 0:
                logging.warning("Failed to load reference probe STL template: %s", templatePath)
                continue

            templateName = self._referenceProbeTemplateDisplayName(templatePath)
            templateNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", templateName)
            templateNode.CreateDefaultDisplayNodes()
            segmentID = templateNode.AddSegmentFromClosedSurfaceRepresentation(closedSurface, templateName, [0.2, 0.8, 1.0])
            if not segmentID:
                logging.warning("Failed to import reference probe template geometry into segmentation node: %s", templatePath)
                slicer.mrmlScene.RemoveNode(templateNode)
                continue

            templateNode.SetAttribute(REFERENCE_PROBE_TEMPLATE_ATTRIBUTE, "1")
            templateNode.SetAttribute(REFERENCE_PROBE_TEMPLATE_SOURCE_PATH_ATTRIBUTE, str(templatePath))
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
            metrics.append(
                {
                    "TrajectoryIndex": int(trajectory.trajectoryIndex + 1),
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
            entry = np.array(pointsRAS[pointIndex], dtype=float)
            target = np.array(pointsRAS[pointIndex + 1], dtype=float)
            direction = target - entry
            length = float(np.linalg.norm(direction))
            if length <= 1e-8:
                raise ValueError(f"Control-point pair {pointIndex}-{pointIndex + 1} has zero length.")

            trajectory = ProbeTrajectory(
                entryPointRAS=tuple(entry.tolist()),
                targetPointRAS=tuple(target.tolist()),
                directionVector=tuple((direction / length).tolist()),
                lengthMm=length,
                trajectoryIndex=len(trajectories),
                label=f"Trajectory {len(trajectories) + 1}",
                sourceControlPointIndices=(pointIndex, pointIndex + 1),
            )
            trajectories.append(trajectory)
        return trajectories

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
            label = endpointsMarkups.GetNthControlPointLabel(trajectory.sourceControlPointIndices[0])
            if label:
                trajectory.label = label
        return trajectories

    def placeProbeInstances(
        self,
        referenceProbeSegmentation: vtkMRMLSegmentationNode | None,
        trajectories: Sequence[ProbeTrajectory],
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

        generatedProbeNodeIDs: list[str] = []
        for trajectory in trajectories:
            probeNode = self._cloneReferenceProbe(referenceProbeSegmentation, trajectory.trajectoryIndex)
            self._placeProbeNodeAlongTrajectory(probeNode, trajectory)
            trajectory.generatedProbeNodeID = probeNode.GetID()
            trajectory.status = "placed"
            generatedProbeNodeIDs.append(probeNode.GetID())

        return generatedProbeNodeIDs

    def createTrajectoryLines(self, trajectories: Sequence[ProbeTrajectory], clearExisting: bool = True) -> list[str]:
        if clearExisting:
            self.removeGeneratedTrajectoryLines()

        generatedLineNodeIDs: list[str] = []
        for trajectory in trajectories:
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", f"SV3D Trajectory {trajectory.trajectoryIndex + 1:02d}")
            pointArray = np.array([trajectory.entryPointRAS, trajectory.targetPointRAS], dtype=float)
            slicer.util.updateMarkupsControlPointsFromArray(lineNode, pointArray)
            lineNode.SetAttribute(GENERATED_TRAJECTORY_LINE_ATTRIBUTE, "1")

            lineDisplayNode = lineNode.GetDisplayNode()
            if lineDisplayNode:
                lineDisplayNode.SetPropertiesLabelVisibility(False)
                lineDisplayNode.SetPointLabelsVisibility(False)
                lineDisplayNode.SetSelectable(False)

            generatedLineNodeIDs.append(lineNode.GetID())
        return generatedLineNodeIDs

    @staticmethod
    def _lineIntersectsClosedSurface(
        lineStartRAS: Sequence[float],
        lineEndRAS: Sequence[float],
        closedSurface: vtk.vtkPolyData,
    ) -> bool:
        if closedSurface is None or closedSurface.GetNumberOfPoints() <= 0:
            return False
        obbTree = vtk.vtkOBBTree()
        obbTree.SetDataSet(closedSurface)
        obbTree.BuildLocator()
        intersectionPoints = vtk.vtkPoints()
        intersectionCellIds = vtk.vtkIdList()
        intersectionCount = obbTree.IntersectWithLine(lineStartRAS, lineEndRAS, intersectionPoints, intersectionCellIds)
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
        for segmentInfo in self.getValidSegmentationSegments(criticalStructuresSegmentation, "master trajectory validation"):
            segmentID = str(segmentInfo["segmentID"])
            if excludedSegmentID and segmentID == excludedSegmentID:
                continue

            closedSurface = vtk.vtkPolyData()
            criticalStructuresSegmentation.GetClosedSurfaceRepresentation(segmentID, closedSurface)
            intersects = self._lineIntersectsClosedSurface(
                trajectory.entryPointRAS,
                trajectory.targetPointRAS,
                closedSurface,
            )
            minDistanceMm = self._lineToClosedSurfaceMinDistanceMm(
                trajectory.entryPointRAS,
                trajectory.targetPointRAS,
                closedSurface,
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

    def computeCoaxialPlanFromTrajectory(
        self,
        trajectory: ProbeTrajectory,
        technique: str,
        activeElementLengthMm: float,
    ) -> CoaxialPlanSummary:
        normalizedDirection = _normalize_vector(trajectory.directionVector)
        techniqueName = str(technique or "PullBack")
        if techniqueName == "PushThrough":
            navigationTarget = np.array(trajectory.targetPointRAS, dtype=float) - (normalizedDirection * float(activeElementLengthMm))
            noteText = (
                f"Push-through: navigate the coaxial needle to the derived target, then advance the applicator "
                f"{float(activeElementLengthMm):.1f} mm."
            )
        else:
            navigationTarget = np.array(trajectory.targetPointRAS, dtype=float)
            noteText = (
                f"Pull-back: navigate the coaxial needle to the locked endpoint, then retract the sheath to expose "
                f"{float(activeElementLengthMm):.1f} mm of active element."
            )

        return CoaxialPlanSummary(
            technique=techniqueName,
            activeElementLengthMm=float(activeElementLengthMm),
            navigationTargetRAS=tuple(float(value) for value in navigationTarget.tolist()),
            masterEntryPointRAS=tuple(float(value) for value in trajectory.entryPointRAS),
            masterTargetPointRAS=tuple(float(value) for value in trajectory.targetPointRAS),
            notes=noteText,
        )

    def createOrUpdateCoaxialLine(
        self,
        entryPointRAS: Sequence[float],
        navigationTargetRAS: Sequence[float],
    ):
        existingLineNodes = [
            node for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode")
            if node.GetAttribute(GENERATED_COAXIAL_LINE_ATTRIBUTE) == "1"
        ]
        lineNode = existingLineNodes[0] if len(existingLineNodes) > 0 else slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode",
            COAXIAL_NAVIGATION_LINE_NODE_NAME,
        )
        slicer.util.updateMarkupsControlPointsFromArray(
            lineNode,
            np.array([entryPointRAS, navigationTargetRAS], dtype=float),
        )
        lineNode.SetAttribute(GENERATED_COAXIAL_LINE_ATTRIBUTE, "1")
        if lineNode.GetDisplayNode():
            lineNode.GetDisplayNode().SetPropertiesLabelVisibility(False)
            lineNode.GetDisplayNode().SetPointLabelsVisibility(False)
        for redundantLineNode in existingLineNodes[1:]:
            slicer.mrmlScene.RemoveNode(redundantLineNode)
        return lineNode

    def removeNodesByAttribute(self, className: str, attributeName: str, attributeValue: str = "1", keepNodeID: str | None = None) -> None:
        for node in slicer.util.getNodesByClass(className):
            if keepNodeID and node.GetID() == keepNodeID:
                continue
            if node.GetAttribute(attributeName) == attributeValue:
                slicer.mrmlScene.RemoveNode(node)

    def removeGeneratedProbeNodes(self, keepNodeID: str | None = None) -> None:
        self.removeNodesByAttribute("vtkMRMLSegmentationNode", GENERATED_PROBE_ATTRIBUTE, keepNodeID=keepNodeID)

    def removeGeneratedTrajectoryLines(self) -> None:
        self.removeNodesByAttribute("vtkMRMLMarkupsLineNode", GENERATED_TRAJECTORY_LINE_ATTRIBUTE)

    def removeNodesByIDs(self, nodeIDs: Sequence[str]) -> None:
        for nodeID in nodeIDs:
            node = slicer.mrmlScene.GetNodeByID(nodeID)
            if node:
                slicer.mrmlScene.RemoveNode(node)

    def removeNodeIfOwned(self, node: vtk.vtkObject | None, ownershipAttribute: str, ownershipValue: str = "1") -> bool:
        if not node:
            return False
        if not slicer.mrmlScene.IsNodePresent(node):
            return False
        if node.GetAttribute(ownershipAttribute) != ownershipValue:
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
        self.removeNodesByAttribute(className, ownershipAttribute, keepNodeID=outputNode.GetID())
        return outputNode

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
        self._clearSegmentationSegments(combinedSegmentation)

        for nodeIndex, translatedProbeNode in enumerate(validProbeNodes):
            self._ensureSegmentationHasClosedSurface(translatedProbeNode)
            sourceSegmentID = self.getWorkingSegmentID(
                translatedProbeNode,
                "probe merge input",
            )
            closedSurface = vtk.vtkPolyData()
            translatedProbeNode.GetClosedSurfaceRepresentation(sourceSegmentID, closedSurface)
            if closedSurface.GetNumberOfPoints() <= 0:
                continue
            combinedSegmentation.AddSegmentFromClosedSurfaceRepresentation(
                closedSurface,
                f"Probe_{nodeIndex + 1:02d}",
                [1.0, 0.3, 0.1],
            )

        if combinedSegmentation.GetSegmentation().GetNumberOfSegments() == 0:
            raise RuntimeError("Unable to build combined ablation segmentation from generated probes.")

        try:
            self._unionSegmentsWithLogicalOperators(combinedSegmentation)
        except Exception as exc:
            logging.warning("Logical operators union failed (%s). Falling back to appended surfaces.", exc)
            self._mergeSegmentsByAppendingSurfaces(combinedSegmentation)

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

        structureSafetySummaryRows: list[dict[str, float | int | str]] = []
        structureSafetyThresholdRows: list[dict[str, float | int | str]] = []
        try:
            for segmentInfo in riskSegments:
                segmentID = str(segmentInfo["segmentID"])
                segmentName = str(segmentInfo["segmentName"])
                self.segmentationSegmentToModel(
                    riskStructuresSegmentation,
                    segmentID,
                    TEMP_STRUCTURE_SAFETY_MODEL_NODE_NAME,
                    outputModelNode=tempStructureModel,
                )

                # Negative values indicate structure points inside/overlapping ablation geometry.
                self.computeSignedDistanceModel(tempStructureModel, probeModel, tempDistanceModel)

                signedDistanceValues = self.getSignedMarginValues(tempDistanceModel)
                distanceSummary = self.computeDistanceSummary(signedDistanceValues)
                thresholdSummary = self.computeDistanceThresholdSummary(signedDistanceValues)

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
        if hasattr(colorNode, "SetNamesInitialised"):
            colorNode.SetNamesInitialised(True)
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

        halfMamMm = float(mamMm) * 0.5
        signedMarginsForSummary: list[float] = []
        for pointIndex in range(_data_array_value_count(backupArray)):
            signedDistanceValue = float(backupArray.GetValue(pointIndex))
            achievedMargin = -signedDistanceValue
            signedMarginsForSummary.append(signedDistanceValue)
            if achievedMargin >= float(mamMm):
                bucketValue = 2.0
            elif achievedMargin >= halfMamMm:
                bucketValue = 1.0
            else:
                bucketValue = 0.0
            signedDistanceArray.SetValue(pointIndex, bucketValue)

        colorNode = self.ensureBeginnerMamColorNode()
        self.configureMarginDisplayNode(marginModelNode, autoRange=False, scalarRange=(0.0, 2.0))
        displayNode = marginModelNode.GetDisplayNode()
        if displayNode:
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())
        self.refreshNodeDisplay(marginModelNode)
        return self.computeMamAssessmentSummary(signedMarginsForSummary, mamMm)

    @staticmethod
    def recolorSignedDistanceArray(signedDistanceArray: vtk.vtkDataArray, thresholds: Sequence[float]) -> int:
        if signedDistanceArray is None:
            raise ValueError("Signed distance array is required.")

        sortedThresholds = sorted(float(value) for value in thresholds)
        if len(sortedThresholds) == 0:
            raise ValueError("At least one threshold value is required.")

        for pointIndex in range(_data_array_value_count(signedDistanceArray)):
            signedDistanceValue = signedDistanceArray.GetValue(pointIndex)
            bucketIndex = len(sortedThresholds)
            for thresholdIndex, threshold in enumerate(sortedThresholds):
                if signedDistanceValue < threshold:
                    bucketIndex = thresholdIndex
                    break
            signedDistanceArray.SetValue(pointIndex, float(bucketIndex))

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

        for pointIndex in range(targetValueCount):
            targetArray.SetValue(pointIndex, sourceArray.GetValue(pointIndex))

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

    def getSignedMarginValues(self, marginModelNode: vtkMRMLModelNode | None) -> list[float]:
        if not marginModelNode:
            raise ValueError("Margin model node is required.")

        signedDistanceArray = self.getSignedDistanceBackupArray(marginModelNode)
        if signedDistanceArray is None:
            signedDistanceArray = self.getSignedDistanceArray(marginModelNode)

        signedMarginValues: list[float] = []
        invalidValueCount = 0
        for pointIndex in range(_data_array_value_count(signedDistanceArray)):
            signedValue = float(signedDistanceArray.GetValue(pointIndex))
            if math.isfinite(signedValue):
                signedMarginValues.append(signedValue)
            else:
                invalidValueCount += 1

        if invalidValueCount > 0:
            logging.warning("Ignored %d non-finite signed-margin values during summary computation.", invalidValueCount)

        if len(signedMarginValues) == 0:
            raise RuntimeError("Signed margin model contains no finite scalar values for summary computation.")
        return signedMarginValues

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
        if polyData is None or polyData.GetNumberOfPoints() <= 0:
            raise RuntimeError(f"{modelRole} model '{modelNode.GetName()}' has no valid polydata points.")
        return polyData

    def computeSignedDistanceModel(
        self,
        sourceModelNode: vtkMRMLModelNode,
        targetModelNode: vtkMRMLModelNode,
        outputModelNode: vtkMRMLModelNode,
    ) -> vtkMRMLModelNode:
        sourcePolyData = self._requireModelPolyData(sourceModelNode, "Source")
        targetPolyData = self._requireModelPolyData(targetModelNode, "Target")

        outputPolyData = vtk.vtkPolyData()
        outputPolyData.DeepCopy(sourcePolyData)

        implicitDistance = vtk.vtkImplicitPolyDataDistance()
        implicitDistance.SetInput(targetPolyData)

        enclosedFilter = vtk.vtkSelectEnclosedPoints()
        enclosedFilter.SetInputData(outputPolyData)
        enclosedFilter.SetSurfaceData(targetPolyData)
        enclosedFilter.SetTolerance(1e-6)
        enclosedFilter.CheckSurfaceOff()
        enclosedFilter.Update()

        pointCount = outputPolyData.GetNumberOfPoints()
        signedDistanceArray = vtk.vtkDoubleArray()
        signedDistanceArray.SetName(SIGNED_DISTANCE_ARRAY_NAME)
        signedDistanceArray.SetNumberOfComponents(1)
        signedDistanceArray.SetNumberOfTuples(pointCount)

        pointCoordinates = [0.0, 0.0, 0.0]
        for pointIndex in range(pointCount):
            outputPolyData.GetPoint(pointIndex, pointCoordinates)
            unsignedDistance = abs(float(implicitDistance.EvaluateFunction(pointCoordinates)))
            signedDistance = -unsignedDistance if bool(enclosedFilter.IsInside(pointIndex)) else unsignedDistance
            signedDistanceArray.SetValue(pointIndex, signedDistance)

        pointData = outputPolyData.GetPointData()
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
        segmentation.CreateRepresentation("Closed surface")

    def _cloneReferenceProbe(self, referenceProbeSegmentation: vtkMRMLSegmentationNode, trajectoryIndex: int) -> vtkMRMLSegmentationNode:
        sourceSegmentID = self.getWorkingSegmentID(referenceProbeSegmentation, "reference probe placement")
        sourceSurface = vtk.vtkPolyData()
        referenceProbeSegmentation.GetClosedSurfaceRepresentation(sourceSegmentID, sourceSurface)
        if sourceSurface.GetNumberOfPoints() <= 0:
            raise RuntimeError("Reference probe segmentation has no usable closed-surface geometry.")

        clonedProbeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            f"SV3D Placed Probe {trajectoryIndex + 1:02d}",
        )
        clonedProbeNode.CreateDefaultDisplayNodes()
        clonedProbeNode.AddSegmentFromClosedSurfaceRepresentation(sourceSurface, f"Probe_{trajectoryIndex + 1:02d}", [0.2, 0.9, 0.3])
        clonedProbeNode.GetSegmentation().CreateRepresentation("Closed surface")
        clonedProbeNode.SetAttribute(GENERATED_PROBE_ATTRIBUTE, "1")

        clonedDisplayNode = clonedProbeNode.GetDisplayNode()
        if clonedDisplayNode:
            clonedDisplayNode.SetOpacity(0.35)
            clonedDisplayNode.SetVisibility(True)
        return clonedProbeNode

    def _placeProbeNodeAlongTrajectory(self, probeNode: vtkMRMLSegmentationNode, trajectory: ProbeTrajectory) -> None:
        rotationMatrix = rotation_matrix_from_vectors(REFERENCE_PROBE_DIRECTION_RAS, trajectory.directionVector)
        transformMatrix = _build_rigid_transform(rotationMatrix, trajectory.entryPointRAS)

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

            for modifierSegmentID in segmentIDs[1:]:
                effect.self().scriptedEffect.setParameter("Operation", "UNION")
                effect.self().scriptedEffect.setParameter("ModifierSegmentID", modifierSegmentID)
                effect.self().onApply()
                segmentationNode.GetSegmentation().RemoveSegment(modifierSegmentID)
        finally:
            segmentEditorWidget = None
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

        slicer.util.updateMarkupsControlPointsFromArray(
            endpointsNode,
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -15.0), (5.0, 0.0, 0.0), (5.0, 0.0, -5.0)], dtype=float),
        )
        with self.assertRaises(ValueError):
            logic.extractSingleTrajectoryFromMarkups(endpointsNode)

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
            directionVector=(0.0, 0.0, -1.0),
            lengthMm=20.0,
            trajectoryIndex=0,
        )
        coaxialSummary = SurgicalVision3D_PlannerLogic().computeCoaxialPlanFromTrajectory(
            trajectory,
            "PushThrough",
            10.0,
        )
        self.assertEqual(coaxialSummary.technique, "PushThrough")
        self.assertTrue(np.allclose(np.array(coaxialSummary.navigationTargetRAS), np.array([0.0, 0.0, -10.0]), atol=1e-6))

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
        self.assertEqual(SurgicalVision3D_PlannerLogic.deserializeNodeIDs(parameterNode.generatedProbeNodeIDs), [])

        probeSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        parameterNode.referenceProbeSegmentation = probeSegmentationNode

        restoredParameterNode = logic.getParameterNode()
        self.assertIsNotNone(restoredParameterNode.referenceProbeSegmentation)
        self.assertEqual(restoredParameterNode.referenceProbeSegmentation.GetID(), probeSegmentationNode.GetID())
