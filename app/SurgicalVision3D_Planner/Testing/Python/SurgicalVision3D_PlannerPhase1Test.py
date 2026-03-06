import json
import tempfile
from pathlib import Path

import numpy as np
import vtk

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

import SurgicalVision3D_Planner as planner


class SurgicalVision3D_PlannerPhase1Test(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_rotation_matrix_covers_parallel_cases()
        self.test_trajectory_extraction_and_odd_point_handling()
        self.test_trajectory_metrics_from_known_point_pairs()
        self.test_signed_margin_summary_metrics_from_synthetic_values()
        self.test_margin_threshold_summary_metrics()
        self.test_phase2a_metrics_handle_empty_or_invalid_values()
        self.test_structure_segment_enumeration_handles_multiple_segments()
        self.test_structure_safety_summary_metrics_from_synthetic_values()
        self.test_structure_safety_threshold_metrics_from_synthetic_values()
        self.test_structure_safety_optional_input_returns_empty_outputs()
        self.test_structure_safety_empty_segmentation_guard()
        self.test_empty_segmentation_guard()
        self.test_repeated_trajectory_lines_are_replaced()
        self.test_owned_output_node_reuse_policy()
        self.test_repeated_merge_reuses_combined_output()
        self.test_summary_table_outputs_are_reused_deterministically()
        self.test_structure_safety_tables_are_reused_deterministically()
        self.test_probe_coordination_pairwise_metrics()
        self.test_probe_coordination_plan_aggregation_and_gate_flags()
        self.test_no_touch_entry_outside_tumor_rule()
        self.test_probe_coordination_tables_are_reused_deterministically()
        self.test_export_manifest_creation_from_synthetic_state()
        self.test_export_bundle_path_generation()
        self.test_table_node_to_dict_serialization()
        self.test_csv_and_json_export_helpers()
        self.test_selected_scenario_export_mode()
        self.test_export_bundle_handles_missing_optional_outputs()
        self.test_repeated_export_sequence_behavior()
        self.test_recolor_restore_uses_full_array_length()
        self.test_parameter_node_restore_round_trip()

    def _createSpherePolyData(self, center=(0.0, 0.0, 0.0), radius=3.0) -> vtk.vtkPolyData:
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetRadius(radius)
        sphereSource.SetCenter(center[0], center[1], center[2])
        sphereSource.SetThetaResolution(16)
        sphereSource.SetPhiResolution(16)
        sphereSource.Update()
        return sphereSource.GetOutput()

    def _createSphereSegmentation(self, nodeName: str, center=(0.0, 0.0, 0.0)) -> slicer.vtkMRMLSegmentationNode:
        spherePolyData = self._createSpherePolyData(center=center)

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", nodeName)
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.AddSegmentFromClosedSurfaceRepresentation(spherePolyData, "Segment_1", [1.0, 0.3, 0.2])
        segmentationNode.GetSegmentation().CreateRepresentation("Closed surface")
        return segmentationNode

    def test_rotation_matrix_covers_parallel_cases(self):
        source = np.array([0.0, 0.0, -1.0], dtype=float)

        aligned = planner.rotation_matrix_from_vectors(source, source).dot(source)
        self.assertTrue(np.allclose(aligned, source, atol=1e-6))

        opposite = planner.rotation_matrix_from_vectors(source, -source).dot(source)
        self.assertTrue(np.allclose(opposite, -source, atol=1e-6))

        orthogonal = planner.rotation_matrix_from_vectors(source, [0.0, 1.0, 0.0]).dot(source)
        self.assertTrue(np.allclose(orthogonal, [0.0, 1.0, 0.0], atol=1e-6))

    def test_trajectory_extraction_and_odd_point_handling(self):
        trajectories = planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
            [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, -10.0),
                (5.0, 2.0, 1.0),
                (5.0, 2.0, -4.0),
            ],
            strictEven=True,
        )
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(trajectories[0].sourceControlPointIndices, (0, 1))
        self.assertAlmostEqual(trajectories[0].lengthMm, 10.0, places=6)

        with self.assertRaises(ValueError):
            planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
                [(0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (1.0, 1.0, 1.0)],
                strictEven=True,
            )

        safeTrajectories = planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
            [(0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (1.0, 1.0, 1.0)],
            strictEven=False,
        )
        self.assertEqual(len(safeTrajectories), 1)

    def test_trajectory_metrics_from_known_point_pairs(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        trajectories = planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
            [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, -10.0),
                (2.0, 3.0, 4.0),
                (2.0, 3.0, -1.0),
            ],
            strictEven=True,
        )

        metrics = logic.computeTrajectoryMetrics(trajectories)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(int(metrics[0]["TrajectoryIndex"]), 1)
        self.assertAlmostEqual(float(metrics[0]["LengthMm"]), 10.0, places=6)
        self.assertEqual(int(metrics[1]["TrajectoryIndex"]), 2)
        self.assertAlmostEqual(float(metrics[1]["LengthMm"]), 5.0, places=6)
        self.assertAlmostEqual(float(metrics[1]["DirS"]), -1.0, places=6)

    def test_signed_margin_summary_metrics_from_synthetic_values(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        signedMargins = np.array([-2.0, 0.0, 1.0, 4.0, 10.0], dtype=float)

        summary = logic.computeSignedMarginSummary(
            signedMargins.tolist(),
            trajectoryCount=3,
            tumorSegmentID="Segment_1",
            tumorSegmentName="Tumor",
        )
        self.assertEqual(int(summary["TrajectoryCount"]), 3)
        self.assertEqual(summary["TumorSegmentID"], "Segment_1")
        self.assertEqual(summary["TumorSegmentName"], "Tumor")
        self.assertAlmostEqual(float(summary["MinSignedMarginMm"]), float(np.min(signedMargins)), places=6)
        self.assertAlmostEqual(float(summary["MeanSignedMarginMm"]), float(np.mean(signedMargins)), places=6)
        self.assertAlmostEqual(float(summary["MedianSignedMarginMm"]), float(np.median(signedMargins)), places=6)
        self.assertAlmostEqual(float(summary["P20SignedMarginMm"]), float(np.quantile(signedMargins, 0.20)), places=6)
        self.assertAlmostEqual(float(summary["P80SignedMarginMm"]), float(np.quantile(signedMargins, 0.80)), places=6)

    def test_margin_threshold_summary_metrics(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        signedMargins = [-3.0, -0.1, 0.0, 1.9, 2.0, 4.9, 5.0, 10.0]
        summaryRows = logic.computeMarginThresholdSummary(signedMargins)

        self.assertEqual(len(summaryRows), 4)
        self.assertEqual(summaryRows[0]["Bucket"], "< 0 mm")
        self.assertEqual(int(summaryRows[0]["Count"]), 2)
        self.assertAlmostEqual(float(summaryRows[0]["Percent"]), 25.0, places=6)
        self.assertEqual(summaryRows[1]["Bucket"], "< 2 mm")
        self.assertEqual(int(summaryRows[1]["Count"]), 4)
        self.assertAlmostEqual(float(summaryRows[1]["Percent"]), 50.0, places=6)
        self.assertEqual(summaryRows[2]["Bucket"], "< 5 mm")
        self.assertEqual(int(summaryRows[2]["Count"]), 6)
        self.assertAlmostEqual(float(summaryRows[2]["Percent"]), 75.0, places=6)
        self.assertEqual(summaryRows[3]["Bucket"], ">= 5 mm")
        self.assertEqual(int(summaryRows[3]["Count"]), 2)
        self.assertAlmostEqual(float(summaryRows[3]["Percent"]), 25.0, places=6)

    def test_phase2a_metrics_handle_empty_or_invalid_values(self):
        logic = planner.SurgicalVision3D_PlannerLogic()

        with self.assertRaises(ValueError):
            logic.computeSignedMarginSummary([], trajectoryCount=0, tumorSegmentID="A", tumorSegmentName="B")
        with self.assertRaises(ValueError):
            logic.computeMarginThresholdSummary([])
        with self.assertRaises(ValueError):
            logic.computeSignedMarginSummary([np.nan], trajectoryCount=1, tumorSegmentID="A", tumorSegmentName="B")
        with self.assertRaises(ValueError):
            logic.computeMarginThresholdSummary([np.nan, np.inf])

    def test_structure_segment_enumeration_handles_multiple_segments(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        riskSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "RiskStructures")
        riskSegmentation.CreateDefaultDisplayNodes()
        riskSegmentation.AddSegmentFromClosedSurfaceRepresentation(
            self._createSpherePolyData(center=(12.0, 0.0, 0.0), radius=2.0),
            "Vessel",
            [0.8, 0.1, 0.1],
        )
        riskSegmentation.AddSegmentFromClosedSurfaceRepresentation(
            self._createSpherePolyData(center=(0.0, 12.0, 0.0), radius=2.0),
            "BileDuct",
            [0.9, 0.8, 0.1],
        )
        riskSegmentation.GetSegmentation().CreateRepresentation("Closed surface")

        segments = logic.getValidSegmentationSegments(riskSegmentation, "test risk segment enumeration")
        self.assertEqual(len(segments), 2)
        names = {segmentInfo["segmentName"] for segmentInfo in segments}
        self.assertIn("Vessel", names)
        self.assertIn("BileDuct", names)

    def test_structure_safety_summary_metrics_from_synthetic_values(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        distances = np.array([-1.0, 0.0, 2.0, 6.0], dtype=float)
        summary = logic.computeDistanceSummary(distances.tolist())

        self.assertAlmostEqual(float(summary["MinDistanceMm"]), float(np.min(distances)), places=6)
        self.assertAlmostEqual(float(summary["MeanDistanceMm"]), float(np.mean(distances)), places=6)
        self.assertAlmostEqual(float(summary["MedianDistanceMm"]), float(np.median(distances)), places=6)
        self.assertAlmostEqual(float(summary["P20DistanceMm"]), float(np.quantile(distances, 0.20)), places=6)
        self.assertAlmostEqual(float(summary["P80DistanceMm"]), float(np.quantile(distances, 0.80)), places=6)

    def test_structure_safety_threshold_metrics_from_synthetic_values(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        distances = [-1.0, 0.0, 1.0, 4.0, 5.0]
        summary = logic.computeDistanceThresholdSummary(distances)

        self.assertEqual(int(summary["CountBelow0Mm"]), 1)
        self.assertAlmostEqual(float(summary["PercentBelow0Mm"]), 20.0, places=6)
        self.assertEqual(int(summary["CountBelow2Mm"]), 3)
        self.assertAlmostEqual(float(summary["PercentBelow2Mm"]), 60.0, places=6)
        self.assertEqual(int(summary["CountBelow5Mm"]), 4)
        self.assertAlmostEqual(float(summary["PercentBelow5Mm"]), 80.0, places=6)
        self.assertEqual(int(summary["CountAtLeast5Mm"]), 1)
        self.assertAlmostEqual(float(summary["PercentAtLeast5Mm"]), 20.0, places=6)

    def test_structure_safety_optional_input_returns_empty_outputs(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        summaryRows, thresholdRows = logic.evaluateStructureSafety(None, None)
        self.assertEqual(summaryRows, [])
        self.assertEqual(thresholdRows, [])

    def test_structure_safety_empty_segmentation_guard(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        emptyRiskSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "EmptyRiskSegmentation")
        with self.assertRaisesRegex(RuntimeError, "has no segments"):
            logic.getValidSegmentationSegments(emptyRiskSegmentation, "structure safety evaluation")

    def test_empty_segmentation_guard(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        emptySegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "EmptySegmentation")
        with self.assertRaisesRegex(RuntimeError, "has no segments"):
            logic.getWorkingSegmentID(emptySegmentation, "guard test")

    def test_repeated_trajectory_lines_are_replaced(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        trajectories = planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
            [(0.0, 0.0, 0.0), (0.0, 0.0, -10.0)],
            strictEven=True,
        )

        firstLineNodeIDs = logic.createTrajectoryLines(trajectories, clearExisting=True)
        self.assertEqual(len(logic.resolveExistingNodeIDs(firstLineNodeIDs)), 1)

        secondLineNodeIDs = logic.createTrajectoryLines(trajectories, clearExisting=True)
        self.assertEqual(len(logic.resolveExistingNodeIDs(firstLineNodeIDs)), 0)
        self.assertEqual(len(logic.resolveExistingNodeIDs(secondLineNodeIDs)), 1)

        generatedLineNodes = [
            node
            for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode")
            if node.GetAttribute(planner.GENERATED_TRAJECTORY_LINE_ATTRIBUTE) == "1"
        ]
        self.assertEqual(len(generatedLineNodes), 1)

    def test_owned_output_node_reuse_policy(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        userOwnedModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "UserModel")

        generatedModel = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLModelNode",
            planner.MARGIN_MODEL_NODE_NAME,
            planner.GENERATED_MARGIN_MODEL_ATTRIBUTE,
            userOwnedModel,
        )
        self.assertNotEqual(generatedModel.GetID(), userOwnedModel.GetID())
        self.assertEqual(generatedModel.GetAttribute(planner.GENERATED_MARGIN_MODEL_ATTRIBUTE), "1")

        reusedModel = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLModelNode",
            planner.MARGIN_MODEL_NODE_NAME,
            planner.GENERATED_MARGIN_MODEL_ATTRIBUTE,
            generatedModel,
        )
        self.assertEqual(reusedModel.GetID(), generatedModel.GetID())

    def test_repeated_merge_reuses_combined_output(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        referenceProbe = self._createSphereSegmentation("ProbeReference")
        trajectories = planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
            [(0.0, 0.0, 0.0), (0.0, 0.0, -8.0)],
            strictEven=True,
        )
        generatedProbeNodeIDs = logic.placeProbeInstances(referenceProbe, trajectories)

        combined1 = logic.mergeProbeInstances(generatedProbeNodeIDs)
        combined2 = logic.mergeProbeInstances(generatedProbeNodeIDs, combined1)

        self.assertEqual(combined1.GetID(), combined2.GetID())
        self.assertEqual(combined2.GetName(), planner.COMBINED_PROBE_NODE_NAME)
        self.assertEqual(combined2.GetAttribute(planner.GENERATED_COMBINED_PROBE_ATTRIBUTE), "1")
        self.assertEqual(combined2.GetSegmentation().GetNumberOfSegments(), 1)

    def test_summary_table_outputs_are_reused_deterministically(self):
        logic = planner.SurgicalVision3D_PlannerLogic()

        trajectories = planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
            [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, -10.0),
                (1.0, 1.0, 1.0),
                (1.0, 1.0, -4.0),
            ],
            strictEven=True,
        )
        trajectoryMetrics = logic.computeTrajectoryMetrics(trajectories)
        trajectoryTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.TRAJECTORY_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
        )
        logic.populateTrajectorySummaryTable(trajectoryTable, trajectoryMetrics)
        self.assertEqual(logic.tableNodeRowCount(trajectoryTable), 2)

        reusedTrajectoryTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.TRAJECTORY_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE,
            trajectoryTable,
        )
        self.assertEqual(reusedTrajectoryTable.GetID(), trajectoryTable.GetID())
        logic.populateTrajectorySummaryTable(reusedTrajectoryTable, [])
        self.assertEqual(logic.tableNodeRowCount(reusedTrajectoryTable), 0)
        ownedTrajectoryTables = [
            node
            for node in slicer.util.getNodesByClass("vtkMRMLTableNode")
            if node.GetAttribute(planner.GENERATED_TRAJECTORY_SUMMARY_TABLE_ATTRIBUTE) == "1"
        ]
        self.assertEqual(len(ownedTrajectoryTables), 1)

        planSummaryTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.PLAN_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_PLAN_SUMMARY_TABLE_ATTRIBUTE,
        )
        planSummary = logic.computeSignedMarginSummary(
            signedMarginValues=[-1.0, 0.5, 2.0, 7.0],
            trajectoryCount=2,
            tumorSegmentID="Segment_1",
            tumorSegmentName="Tumor",
        )
        logic.populatePlanSummaryTable(planSummaryTable, planSummary)
        self.assertEqual(logic.tableNodeRowCount(planSummaryTable), 1)

        thresholdSummaryTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.MARGIN_THRESHOLD_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_MARGIN_THRESHOLD_TABLE_ATTRIBUTE,
        )
        thresholdSummary = logic.computeMarginThresholdSummary([-1.0, 0.5, 2.0, 7.0])
        logic.populateMarginThresholdSummaryTable(thresholdSummaryTable, thresholdSummary)
        self.assertEqual(logic.tableNodeRowCount(thresholdSummaryTable), 4)

    def test_structure_safety_tables_are_reused_deterministically(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        summaryRows = [
            {
                "StructureSegmentID": "Segment_1",
                "StructureName": "Vessel",
                "MinDistanceMm": -1.0,
                "MeanDistanceMm": 1.0,
                "MedianDistanceMm": 0.5,
                "P20DistanceMm": -0.5,
                "P80DistanceMm": 2.0,
            },
            {
                "StructureSegmentID": "Segment_2",
                "StructureName": "BileDuct",
                "MinDistanceMm": 0.2,
                "MeanDistanceMm": 1.8,
                "MedianDistanceMm": 1.3,
                "P20DistanceMm": 0.5,
                "P80DistanceMm": 2.8,
            },
        ]
        thresholdRows = [
            {
                "StructureSegmentID": "Segment_1",
                "StructureName": "Vessel",
                "CountBelow0Mm": 5,
                "PercentBelow0Mm": 25.0,
                "CountBelow2Mm": 12,
                "PercentBelow2Mm": 60.0,
                "CountBelow5Mm": 18,
                "PercentBelow5Mm": 90.0,
                "CountAtLeast5Mm": 2,
                "PercentAtLeast5Mm": 10.0,
            }
        ]

        summaryTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.STRUCTURE_SAFETY_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_STRUCTURE_SAFETY_SUMMARY_TABLE_ATTRIBUTE,
        )
        thresholdTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.STRUCTURE_SAFETY_THRESHOLD_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_STRUCTURE_SAFETY_THRESHOLD_TABLE_ATTRIBUTE,
        )
        logic.populateStructureSafetySummaryTable(summaryTable, summaryRows)
        logic.populateStructureSafetyThresholdSummaryTable(thresholdTable, thresholdRows)
        self.assertEqual(logic.tableNodeRowCount(summaryTable), 2)
        self.assertEqual(logic.tableNodeRowCount(thresholdTable), 1)

        reusedSummaryTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.STRUCTURE_SAFETY_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_STRUCTURE_SAFETY_SUMMARY_TABLE_ATTRIBUTE,
            summaryTable,
        )
        reusedThresholdTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.STRUCTURE_SAFETY_THRESHOLD_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_STRUCTURE_SAFETY_THRESHOLD_TABLE_ATTRIBUTE,
            thresholdTable,
        )
        self.assertEqual(reusedSummaryTable.GetID(), summaryTable.GetID())
        self.assertEqual(reusedThresholdTable.GetID(), thresholdTable.GetID())
        logic.populateStructureSafetySummaryTable(reusedSummaryTable, [])
        logic.populateStructureSafetyThresholdSummaryTable(reusedThresholdTable, [])
        self.assertEqual(logic.tableNodeRowCount(reusedSummaryTable), 0)
        self.assertEqual(logic.tableNodeRowCount(reusedThresholdTable), 0)

    def test_probe_coordination_pairwise_metrics(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        trajectories = planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
            [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, -10.0),
                (10.0, 0.0, 0.0),
                (10.0, 0.0, -10.0),
                (0.0, 0.0, 0.0),
                (0.0, 10.0, 0.0),
            ],
            strictEven=True,
        )
        settings = planner.ProbeCoordinationConstraintSettings(
            minInterProbeDistanceMm=1.0,
            maxInterProbeDistanceMm=200.0,
            minEntryPointSpacingMm=1.0,
            minTargetPointSpacingMm=1.0,
            enableAngleRule=True,
            maxParallelAngleDeg=5.0,
        )

        pairRowParallel = logic.evaluateProbePairCoordination(trajectories[0], trajectories[1], settings)
        self.assertAlmostEqual(float(pairRowParallel["InterProbeDistanceMm"]), 10.0, places=6)
        self.assertAlmostEqual(float(pairRowParallel["EntryPointSpacingMm"]), 10.0, places=6)
        self.assertAlmostEqual(float(pairRowParallel["TargetPointSpacingMm"]), 10.0, places=6)
        self.assertAlmostEqual(float(pairRowParallel["ProbeAxisAngleDeg"]), 0.0, places=6)
        self.assertIn("ProbeAxesTooParallel", str(pairRowParallel["FailedConstraintNames"]))

        pairRowOrthogonal = logic.evaluateProbePairCoordination(trajectories[0], trajectories[2], settings)
        self.assertAlmostEqual(float(pairRowOrthogonal["ProbeAxisAngleDeg"]), 90.0, places=6)
        self.assertNotIn("ProbeAxesTooParallel", str(pairRowOrthogonal["FailedConstraintNames"]))

    def test_probe_coordination_plan_aggregation_and_gate_flags(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        trajectories = planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
            [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, -10.0),
                (10.0, 0.0, 0.0),
                (10.0, 0.0, -10.0),
                (30.0, 0.0, 0.0),
                (30.0, 0.0, -10.0),
            ],
            strictEven=True,
        )
        settings = planner.ProbeCoordinationConstraintSettings(
            minInterProbeDistanceMm=12.0,
            maxInterProbeDistanceMm=200.0,
            minEntryPointSpacingMm=0.0,
            minTargetPointSpacingMm=0.0,
            enableEntrySpacingRule=False,
            enableTargetSpacingRule=False,
            enableAngleRule=False,
            enableOverlapRule=False,
            requireAllProbePairsFeasible=True,
        )

        pairRows, planSummary, noTouchSummary = logic.evaluatePlanProbeCoordination(trajectories, settings, None)
        self.assertEqual(len(pairRows), 3)
        self.assertEqual([(int(row["ProbeAIndex"]), int(row["ProbeBIndex"])) for row in pairRows], [(1, 2), (1, 3), (2, 3)])
        self.assertEqual(int(planSummary["PairCount"]), 3)
        self.assertEqual(int(planSummary["InfeasiblePairCount"]), 1)
        self.assertFalse(bool(planSummary["AllPairsFeasible"]))
        self.assertFalse(bool(planSummary["CoordinationGatePass"]))
        self.assertIn("ProbePairCoordinationFailed", str(planSummary["CoordinationFailureSummary"]))
        self.assertFalse(bool(noTouchSummary["NoTouchChecked"]))

        settings.requireAllProbePairsFeasible = False
        _, relaxedPlanSummary, _ = logic.evaluatePlanProbeCoordination(trajectories, settings, None)
        self.assertTrue(bool(relaxedPlanSummary["CoordinationGatePass"]))
        self.assertEqual(str(relaxedPlanSummary["CoordinationFailureSummary"]), "")

    def test_no_touch_entry_outside_tumor_rule(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        tumorSegmentation = self._createSphereSegmentation("NoTouchTumor", center=(0.0, 0.0, 0.0))
        trajectories = planner.SurgicalVision3D_PlannerLogic.extractTrajectoriesFromPointPairs(
            [
                (10.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, -10.0),
            ],
            strictEven=True,
        )

        noTouchSummary = logic.evaluateNoTouchArrangement(trajectories, tumorSegmentation)
        self.assertTrue(bool(noTouchSummary["NoTouchChecked"]))
        self.assertFalse(bool(noTouchSummary["NoTouchPass"]))
        self.assertEqual(int(noTouchSummary["EntryPointsInsideTumorCount"]), 1)
        self.assertEqual(str(noTouchSummary["FailedTrajectoryIndices"]), "2")

        settings = planner.ProbeCoordinationConstraintSettings(
            enableNoTouchCheck=True,
            requireAllProbePairsFeasible=False,
            enableInterProbeDistanceRule=False,
            enableEntrySpacingRule=False,
            enableTargetSpacingRule=False,
            enableAngleRule=False,
            enableOverlapRule=False,
        )
        _, planSummary, _ = logic.evaluatePlanProbeCoordination(trajectories, settings, tumorSegmentation)
        self.assertFalse(bool(planSummary["NoTouchPass"]))
        self.assertFalse(bool(planSummary["CoordinationGatePass"]))
        self.assertIn("NoTouchCheckFailed", str(planSummary["CoordinationFailureSummary"]))

    def test_probe_coordination_tables_are_reused_deterministically(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        settings = planner.ProbeCoordinationConstraintSettings(enableNoTouchCheck=True, enableAngleRule=True)
        pairRows = [
            {
                "ProbeAIndex": 1,
                "ProbeBIndex": 2,
                "IsFeasible": False,
                "FailedConstraintCount": 2,
                "FailedConstraintNames": "EntryPointSpacingBelowMin;ProbeAxesTooParallel",
                "InterProbeDistanceMm": 4.0,
                "EntryPointSpacingMm": 2.0,
                "TargetPointSpacingMm": 2.0,
                "ProbeAxisAngleDeg": 1.5,
                "OverlapRedundancyPercent": 82.0,
            }
        ]
        planSummary = {
            "ScenarioOrPlanName": "CurrentPlan",
            "ProbeCount": 2,
            "PairCount": 1,
            "FeasiblePairCount": 0,
            "InfeasiblePairCount": 1,
            "AllPairsFeasible": False,
            "AggregatedFailedConstraintNames": "EntryPointSpacingBelowMin;ProbeAxesTooParallel",
            "NoTouchPass": False,
            "CoordinationGatePass": False,
            "CoordinationFailureSummary": "ProbePairCoordinationFailed;NoTouchCheckFailed",
        }
        noTouchSummary = {
            "NoTouchChecked": True,
            "NoTouchPass": False,
            "Reason": "Entry point is inside tumor for one or more trajectories.",
            "EntryPointsInsideTumorCount": 1,
            "FailedTrajectoryIndices": "2",
        }

        settingsTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.PROBE_COORDINATION_SETTINGS_TABLE_NODE_NAME,
            planner.GENERATED_PROBE_COORDINATION_SETTINGS_TABLE_ATTRIBUTE,
        )
        pairTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.PROBE_PAIR_COORDINATION_TABLE_NODE_NAME,
            planner.GENERATED_PROBE_PAIR_COORDINATION_TABLE_ATTRIBUTE,
        )
        planTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.PROBE_COORDINATION_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_PROBE_COORDINATION_SUMMARY_TABLE_ATTRIBUTE,
        )
        noTouchTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.NO_TOUCH_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_NO_TOUCH_SUMMARY_TABLE_ATTRIBUTE,
        )

        logic.populateProbeCoordinationConstraintSettingsTable(settingsTable, settings)
        logic.populateProbePairCoordinationSummaryTable(pairTable, pairRows)
        logic.populateProbeCoordinationSummaryTable(planTable, planSummary)
        logic.populateNoTouchSummaryTable(noTouchTable, noTouchSummary)
        self.assertEqual(logic.tableNodeRowCount(settingsTable), 13)
        self.assertEqual(logic.tableNodeRowCount(pairTable), 1)
        self.assertEqual(logic.tableNodeRowCount(planTable), 1)
        self.assertEqual(logic.tableNodeRowCount(noTouchTable), 1)

        reusedSettingsTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.PROBE_COORDINATION_SETTINGS_TABLE_NODE_NAME,
            planner.GENERATED_PROBE_COORDINATION_SETTINGS_TABLE_ATTRIBUTE,
            settingsTable,
        )
        reusedPairTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.PROBE_PAIR_COORDINATION_TABLE_NODE_NAME,
            planner.GENERATED_PROBE_PAIR_COORDINATION_TABLE_ATTRIBUTE,
            pairTable,
        )
        reusedPlanTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.PROBE_COORDINATION_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_PROBE_COORDINATION_SUMMARY_TABLE_ATTRIBUTE,
            planTable,
        )
        reusedNoTouchTable = logic.createOrReuseOwnedOutputNode(
            "vtkMRMLTableNode",
            planner.NO_TOUCH_SUMMARY_TABLE_NODE_NAME,
            planner.GENERATED_NO_TOUCH_SUMMARY_TABLE_ATTRIBUTE,
            noTouchTable,
        )
        self.assertEqual(reusedSettingsTable.GetID(), settingsTable.GetID())
        self.assertEqual(reusedPairTable.GetID(), pairTable.GetID())
        self.assertEqual(reusedPlanTable.GetID(), planTable.GetID())
        self.assertEqual(reusedNoTouchTable.GetID(), noTouchTable.GetID())

        logic.populateProbePairCoordinationSummaryTable(reusedPairTable, [])
        self.assertEqual(logic.tableNodeRowCount(reusedPairTable), 0)

        ownedPairTables = [
            node
            for node in slicer.util.getNodesByClass("vtkMRMLTableNode")
            if node.GetAttribute(planner.GENERATED_PROBE_PAIR_COORDINATION_TABLE_ATTRIBUTE) == "1"
        ]
        self.assertEqual(len(ownedPairTables), 1)

    def test_export_manifest_creation_from_synthetic_state(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        parameterNode = logic.getParameterNode()
        exportConfig = planner.PlanExportConfig(
            exportMode="CurrentWorkingPlan",
            exportBaseName="Trial Export",
            includeTrajectoryTables=True,
            includeSafetyTables=True,
            includeCoordinationTables=True,
        )

        manifest = logic.buildPlanExportManifest(
            parameterNode=parameterNode,
            exportConfig=exportConfig,
            exportSequence=7,
            filesExported=["manifest.json", "plan_summary.json", "tables/trajectory_summary.csv"],
            selectedScenarioSummary={"ScenarioName": "Scenario Alpha"},
        )
        self.assertEqual(manifest.exportId, "SV3D-Export-0007")
        self.assertEqual(manifest.exportMode, "CurrentWorkingPlan")
        self.assertEqual(manifest.exportBaseName, "Trial Export")
        self.assertEqual(manifest.selectedScenarioName, "Scenario Alpha")
        self.assertEqual(manifest.filesExported[0], "manifest.json")

    def test_export_bundle_path_generation(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        bundlePath = logic.buildDeterministicBundlePath("C:/tmp/SV3D", "My Export", 12)
        self.assertEqual(bundlePath.name, "My_Export_0012")

    def test_table_node_to_dict_serialization(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "ExportSerializationTable")
        planSummary = {
            "TrajectoryCount": 3,
            "TumorSegmentID": "TumorSegment_01",
            "TumorSegmentName": "Tumor A",
            "MinSignedMarginMm": -1.0,
            "MeanSignedMarginMm": 2.0,
            "MedianSignedMarginMm": 1.0,
            "P20SignedMarginMm": 0.1,
            "P80SignedMarginMm": 4.0,
        }
        logic.populatePlanSummaryTable(tableNode, planSummary)

        rows = logic._tableNodeToDictionaries(tableNode)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Trajectory Count"], "3")
        self.assertEqual(rows[0]["Tumor Segment ID"], "TumorSegment_01")

    def test_csv_and_json_export_helpers(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        outputRoot = Path(tempfile.mkdtemp(prefix="sv3d_export_test_"))
        try:
            tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "ExportCsvTable")
            thresholdSummary = [
                {"Bucket": "< 0 mm", "Count": 2, "Percent": 50.0},
                {"Bucket": ">= 5 mm", "Count": 2, "Percent": 50.0},
            ]
            logic.populateMarginThresholdSummaryTable(tableNode, thresholdSummary)

            csvPath = outputRoot / "tables" / "threshold.csv"
            logic.exportTableNodeToCsv(tableNode, csvPath)
            self.assertTrue(csvPath.exists())
            csvLines = csvPath.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(csvLines), 2)
            self.assertIn("Margin Bucket", csvLines[0])

            jsonPath = outputRoot / "manifest.json"
            jsonPayload = {"ExportMode": "CurrentWorkingPlan", "FileCount": 2}
            logic.exportStructuredSummaryToJson(jsonPath, jsonPayload)
            self.assertTrue(jsonPath.exists())
            loadedPayload = json.loads(jsonPath.read_text(encoding="utf-8"))
            self.assertEqual(loadedPayload["ExportMode"], "CurrentWorkingPlan")
            self.assertEqual(int(loadedPayload["FileCount"]), 2)
        finally:
            for exportedPath in sorted(outputRoot.rglob("*"), reverse=True):
                if exportedPath.is_file():
                    exportedPath.unlink()
                elif exportedPath.is_dir():
                    exportedPath.rmdir()
            if outputRoot.exists():
                outputRoot.rmdir()

    def test_selected_scenario_export_mode(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        scenarioRegistryTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "SV3D Scenario Registry")
        logic._addStringColumn(scenarioRegistryTable, "ScenarioID", ["S001", "S002"])
        logic._addStringColumn(scenarioRegistryTable, "ScenarioName", ["Baseline", "CandidateA"])

        summary = logic.collectScenarioExportData("S002")
        self.assertEqual(summary["SelectedScenarioID"], "S002")
        self.assertEqual(summary["ScenarioName"], "CandidateA")
        self.assertEqual(summary["Source"], "SV3D Scenario Registry")

    def test_export_bundle_handles_missing_optional_outputs(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        parameterNode = logic.getParameterNode()
        exportRoot = Path(tempfile.mkdtemp(prefix="sv3d_bundle_optional_"))
        try:
            exportConfig = planner.PlanExportConfig(
                exportMode="CurrentWorkingPlan",
                exportBaseName="Phase7A",
                exportDirectory=str(exportRoot),
                includeWorkingPlan=True,
                includeSelectedScenario=False,
                includeScenarioComparison=True,
                includeRecommendationOutputs=True,
                includeTrajectoryTables=True,
                includeSafetyTables=True,
                includeCoverageTables=True,
                includeFeasibilityTables=True,
                includeCoordinationTables=True,
                lastExportSequence=0,
            )
            exportResult = logic.exportPlanBundle(parameterNode, exportConfig)
            bundlePath = Path(exportResult["bundlePath"])
            self.assertTrue((bundlePath / "manifest.json").exists())
            self.assertTrue((bundlePath / "plan_summary.json").exists())
            self.assertGreaterEqual(int(exportResult["fileCount"]), 2)
        finally:
            for exportedPath in sorted(exportRoot.rglob("*"), reverse=True):
                if exportedPath.is_file():
                    exportedPath.unlink()
                elif exportedPath.is_dir():
                    exportedPath.rmdir()
            if exportRoot.exists():
                exportRoot.rmdir()

    def test_repeated_export_sequence_behavior(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        parameterNode = logic.getParameterNode()
        exportRoot = Path(tempfile.mkdtemp(prefix="sv3d_bundle_sequence_"))
        try:
            firstConfig = planner.PlanExportConfig(
                exportMode="CurrentWorkingPlan",
                exportBaseName="SequenceCheck",
                exportDirectory=str(exportRoot),
                lastExportSequence=0,
            )
            firstResult = logic.exportPlanBundle(parameterNode, firstConfig)
            secondConfig = planner.PlanExportConfig(
                exportMode="CurrentWorkingPlan",
                exportBaseName="SequenceCheck",
                exportDirectory=str(exportRoot),
                lastExportSequence=int(firstResult["exportSequence"]),
            )
            secondResult = logic.exportPlanBundle(parameterNode, secondConfig)

            firstBundleName = Path(firstResult["bundlePath"]).name
            secondBundleName = Path(secondResult["bundlePath"]).name
            self.assertEqual(firstBundleName, "SequenceCheck_0001")
            self.assertEqual(secondBundleName, "SequenceCheck_0002")
            self.assertNotEqual(firstBundleName, secondBundleName)
        finally:
            for exportedPath in sorted(exportRoot.rglob("*"), reverse=True):
                if exportedPath.is_file():
                    exportedPath.unlink()
                elif exportedPath.is_dir():
                    exportedPath.rmdir()
            if exportRoot.exists():
                exportRoot.rmdir()

    def test_recolor_restore_uses_full_array_length(self):
        signedDistances = vtk.vtkDoubleArray()
        for value in (-8.0, -4.0, -1.0, 3.0):
            signedDistances.InsertNextValue(value)
        backup = vtk.vtkDoubleArray()
        backup.DeepCopy(signedDistances)

        for _ in range(3):
            bucketCount = planner.SurgicalVision3D_PlannerLogic.recolorSignedDistanceArray(signedDistances, (-10.0, -5.0, -2.0))
            self.assertEqual(bucketCount, 4)
            self.assertEqual(signedDistances.GetValue(3), 3.0)

            planner.SurgicalVision3D_PlannerLogic.restoreSignedDistanceArray(signedDistances, backup)
            self.assertEqual(signedDistances.GetValue(3), 3.0)

    def test_parameter_node_restore_round_trip(self):
        logic = planner.SurgicalVision3D_PlannerLogic()
        parameterNode = logic.getParameterNode()
        self.assertEqual(planner.SurgicalVision3D_PlannerLogic.deserializeNodeIDs(parameterNode.generatedProbeNodeIDs), [])

        probeSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        parameterNode.referenceProbeSegmentation = probeSegmentation

        restored = logic.getParameterNode()
        self.assertEqual(restored.referenceProbeSegmentation.GetID(), probeSegmentation.GetID())
