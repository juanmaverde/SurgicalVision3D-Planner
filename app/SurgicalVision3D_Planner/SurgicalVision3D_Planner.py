from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import vtk

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
from slicer.parameterNodeWrapper import parameterNodeWrapper
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
TEMP_PROBE_SAFETY_MODEL_NODE_NAME = "SV3D Temp Probe Safety Input"
TEMP_STRUCTURE_SAFETY_MODEL_NODE_NAME = "SV3D Temp Structure Safety Input"
TEMP_STRUCTURE_SAFETY_DISTANCE_MODEL_NODE_NAME = "SV3D Temp Structure Safety Distance"


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
        self.parent.dependencies = ["ModelToModelDistance", "SegmentEditor", "FiducialRegistration"]
        self.parent.contributors = ["Juan Verde (Surgeon Scientist)"]
        self.parent.helpText = _("""
Phase 1 ablation planning prototype:
1. Generate trajectories from endpoint control-point pairs.
2. Place translated probe segmentations along trajectories.
3. Merge translated probes into one ablation zone.
4. Evaluate tumor-vs-ablation signed margins using ModelToModelDistance.
""")
        self.parent.acknowledgementText = _("""
Legacy AblationPlanner workflow was refactored into this module while preserving scripted-module patterns.
""")


#
# SurgicalVision3D_PlannerParameterNode
#


@parameterNodeWrapper
class SurgicalVision3D_PlannerParameterNode:
    referenceProbeSegmentation: vtkMRMLSegmentationNode | None = None
    endpointsMarkups: vtkMRMLMarkupsFiducialNode | None = None
    tumorSegmentation: vtkMRMLSegmentationNode | None = None
    riskStructuresSegmentation: vtkMRMLSegmentationNode | None = None
    nativeFiducials: vtkMRMLMarkupsFiducialNode | None = None
    registeredFiducials: vtkMRMLMarkupsFiducialNode | None = None
    combinedProbeSegmentation: vtkMRMLSegmentationNode | None = None
    outputMarginModel: vtkMRMLModelNode | None = None
    resultTable: vtkMRMLTableNode | None = None
    tumorTransform: vtkMRMLTransformNode | None = None
    trajectorySummaryTable: vtkMRMLTableNode | None = None
    planSummaryTable: vtkMRMLTableNode | None = None
    marginThresholdSummaryTable: vtkMRMLTableNode | None = None
    structureSafetySummaryTable: vtkMRMLTableNode | None = None
    structureSafetyThresholdSummaryTable: vtkMRMLTableNode | None = None

    createTrajectoryLinesOnPlacement: bool = True
    clearPreviousGeneratedProbes: bool = True
    recolorThresholdLow: float = -10.0
    recolorThresholdMid: float = -5.0
    recolorThresholdHigh: float = -2.0

    generatedProbeNodeIDs: str = "[]"
    generatedTrajectoryLineIDs: str = "[]"


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

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SurgicalVision3D_Planner.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

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

        self.logic = SurgicalVision3D_PlannerLogic()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.placeProbesButton.connect("clicked(bool)", self.onPlaceProbesButton)
        self.ui.createTrajectoryLinesButton.connect("clicked(bool)", self.onCreateTrajectoryLinesButton)
        self.ui.mergeTranslatedProbesButton.connect("clicked(bool)", self.onMergeTranslatedProbesButton)
        self.ui.registerTumorButton.connect("clicked(bool)", self.onRegisterTumorButton)
        self.ui.hardenTumorTransformButton.connect("clicked(bool)", self.onHardenTumorTransformButton)
        self.ui.evaluateMarginsButton.connect("clicked(bool)", self.onEvaluateMarginsButton)
        self.ui.recolorMarginsButton.connect("clicked(bool)", self.onRecolorMarginsButton)
        self.ui.resetMarginColorsButton.connect("clicked(bool)", self.onResetMarginColorsButton)
        self.ui.riskStructuresSegmentationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.onRiskStructuresSegmentationChanged,
        )

        self.initializeParameterNode()

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
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: SurgicalVision3D_PlannerParameterNode | None) -> None:
        if self._parameterNode:
            if self._parameterNodeGuiTag is not None:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateButtonStates)

        self._parameterNode = inputParameterNode

        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateButtonStates)
            self._updateButtonStates()

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

        for nodeFieldName in (
            "combinedProbeSegmentation",
            "outputMarginModel",
            "resultTable",
            "tumorTransform",
            "riskStructuresSegmentation",
            "trajectorySummaryTable",
            "planSummaryTable",
            "marginThresholdSummaryTable",
            "structureSafetySummaryTable",
            "structureSafetyThresholdSummaryTable",
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

    def _updateButtonStates(self, caller=None, event=None) -> None:
        if not self._parameterNode:
            for buttonName in (
                "placeProbesButton",
                "createTrajectoryLinesButton",
                "mergeTranslatedProbesButton",
                "registerTumorButton",
                "hardenTumorTransformButton",
                "evaluateMarginsButton",
                "recolorMarginsButton",
                "resetMarginColorsButton",
            ):
                getattr(self.ui, buttonName).enabled = False
            return

        self._reconcileParameterNodeState()

        hasProbeAndEndpoints = bool(self._parameterNode.referenceProbeSegmentation and self._parameterNode.endpointsMarkups)
        generatedProbeNodeIDs = self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs) if self.logic else []
        hasGeneratedProbes = len(generatedProbeNodeIDs) > 0
        hasCombinedProbe = self._parameterNode.combinedProbeSegmentation is not None
        hasTumor = self._parameterNode.tumorSegmentation is not None
        hasRegistrationInputs = bool(self._parameterNode.tumorSegmentation and self._parameterNode.nativeFiducials and self._parameterNode.registeredFiducials)
        hasMarginModel = self._parameterNode.outputMarginModel is not None
        hasTumorTransform = bool(self._parameterNode.tumorSegmentation and self._parameterNode.tumorSegmentation.GetTransformNodeID())

        self.ui.placeProbesButton.enabled = hasProbeAndEndpoints
        self.ui.createTrajectoryLinesButton.enabled = self._parameterNode.endpointsMarkups is not None
        self.ui.mergeTranslatedProbesButton.enabled = hasGeneratedProbes
        self.ui.registerTumorButton.enabled = hasRegistrationInputs
        self.ui.hardenTumorTransformButton.enabled = hasTumorTransform
        self.ui.evaluateMarginsButton.enabled = hasTumor and (hasCombinedProbe or hasGeneratedProbes)
        self.ui.recolorMarginsButton.enabled = hasMarginModel
        self.ui.resetMarginColorsButton.enabled = hasMarginModel

    def onPlaceProbesButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to place probes."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")

            if not self._parameterNode.referenceProbeSegmentation:
                raise ValueError("Select a reference probe segmentation before placing probes.")
            if not self._parameterNode.endpointsMarkups:
                raise ValueError("Select an endpoint markups node before placing probes.")
            controlPointCount = int(self._parameterNode.endpointsMarkups.GetNumberOfControlPoints())
            if controlPointCount == 0:
                raise ValueError("Endpoint markups node has no control points.")
            if controlPointCount % 2 != 0:
                raise ValueError(
                    f"Endpoint markups has {controlPointCount} control points. Add one more point to complete entry/target pairs."
                )

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

            generatedProbeNodeIDs = self.logic.placeProbeInstances(self._parameterNode.referenceProbeSegmentation, trajectories)
            if self._parameterNode.clearPreviousGeneratedProbes:
                trackedProbeNodeIDs = generatedProbeNodeIDs
            else:
                trackedProbeNodeIDs = self.logic.mergeNodeIDLists(existingProbeNodeIDs, generatedProbeNodeIDs)
            self._parameterNode.generatedProbeNodeIDs = self.logic.serializeNodeIDs(trackedProbeNodeIDs)

            # Probe placement invalidates previously merged/evaluated module-owned outputs.
            self._clearOwnedDerivedOutputs(clearReferences=True)

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
            if not self._parameterNode.endpointsMarkups:
                raise ValueError("Select an endpoint markups node before creating trajectory lines.")
            controlPointCount = int(self._parameterNode.endpointsMarkups.GetNumberOfControlPoints())
            if controlPointCount == 0:
                raise ValueError("Endpoint markups node has no control points.")
            if controlPointCount % 2 != 0:
                raise ValueError(
                    f"Endpoint markups has {controlPointCount} control points. Add one more point to complete entry/target pairs."
                )
            trajectories = self.logic.extractTrajectoriesFromMarkups(self._parameterNode.endpointsMarkups, strictEven=True)
            generatedLineNodeIDs = self.logic.createTrajectoryLines(trajectories, clearExisting=True)
            self._parameterNode.generatedTrajectoryLineIDs = self.logic.serializeNodeIDs(generatedLineNodeIDs)
            self._updateButtonStates()

    def onMergeTranslatedProbesButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to merge translated probes."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
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
            self._parameterNode.outputMarginModel = None
            self._parameterNode.resultTable = None
            self._parameterNode.planSummaryTable = None
            self._parameterNode.marginThresholdSummaryTable = None
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
            self._updateButtonStates()

    def onHardenTumorTransformButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to harden tumor transform."), waitCursor=True):
            if not self.logic or not self._parameterNode:
                raise RuntimeError("Module logic is not initialized.")
            self.logic.hardenTumorTransform(self._parameterNode.tumorSegmentation)
            self._updateButtonStates()

    def onRiskStructuresSegmentationChanged(self, _node=None) -> None:
        if not self.logic or not self._parameterNode:
            return

        self._clearOwnedSafetyOutputs(clearReferences=True)
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
            trajectoryCount = len(
                self.logic.resolveExistingNodeIDs(
                    self.logic.deserializeNodeIDs(self._parameterNode.generatedProbeNodeIDs)
                )
            )
            if trajectoryCount <= 0:
                trajectoryCount = self.logic.tableNodeRowCount(self._parameterNode.trajectorySummaryTable)

            tumorSegmentID, tumorSegmentName = self.logic.getWorkingSegmentInfo(
                self._parameterNode.tumorSegmentation,
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
                self._parameterNode.riskStructuresSegmentation,
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
    def tableNodeRowCount(tableNode: vtkMRMLTableNode | None) -> int:
        if not tableNode or not slicer.mrmlScene.IsNodePresent(tableNode):
            return 0
        table = tableNode.GetTable()
        if table is None:
            return 0
        return int(table.GetNumberOfRows())

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
        if not referenceProbeSegmentation:
            raise ValueError("Reference probe segmentation is required.")
        if len(trajectories) == 0:
            raise ValueError("No trajectories were provided.")

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
        if not hasattr(slicer.modules, "modeltomodeldistance"):
            raise RuntimeError("ModelToModelDistance module is not available.")

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
        tumorModel = self.segmentationFirstSegmentToModel(
            tumorSegmentation,
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
            distanceParameters = {
                "vtkFile1": tumorModel.GetID(),
                "vtkFile2": probeModel.GetID(),
                "distanceType": "signed_closest_point",
                "vtkOutput": marginModel.GetID(),
            }
            cliNode = slicer.cli.runSync(slicer.modules.modeltomodeldistance, None, distanceParameters)
            if cliNode:
                slicer.mrmlScene.RemoveNode(cliNode)

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
        if not hasattr(slicer.modules, "modeltomodeldistance"):
            raise RuntimeError("ModelToModelDistance module is not available.")

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

                # Signed convention from ModelToModelDistance:
                # negative values indicate structure points inside/overlapping ablation geometry.
                distanceParameters = {
                    "vtkFile1": tempStructureModel.GetID(),
                    "vtkFile2": probeModel.GetID(),
                    "distanceType": "signed_closest_point",
                    "vtkOutput": tempDistanceModel.GetID(),
                }
                cliNode = slicer.cli.runSync(slicer.modules.modeltomodeldistance, None, distanceParameters)
                if cliNode:
                    slicer.mrmlScene.RemoveNode(cliNode)

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
