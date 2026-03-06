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
