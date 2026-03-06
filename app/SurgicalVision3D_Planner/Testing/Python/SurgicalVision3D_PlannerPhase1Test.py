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
        self.test_empty_segmentation_guard()
        self.test_repeated_trajectory_lines_are_replaced()
        self.test_owned_output_node_reuse_policy()
        self.test_repeated_merge_reuses_combined_output()
        self.test_recolor_restore_uses_full_array_length()
        self.test_parameter_node_restore_round_trip()

    def _createSphereSegmentation(self, nodeName: str, center=(0.0, 0.0, 0.0)) -> slicer.vtkMRMLSegmentationNode:
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetRadius(3.0)
        sphereSource.SetCenter(center[0], center[1], center[2])
        sphereSource.SetThetaResolution(16)
        sphereSource.SetPhiResolution(16)
        sphereSource.Update()

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", nodeName)
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.AddSegmentFromClosedSurfaceRepresentation(sphereSource.GetOutput(), "Segment_1", [1.0, 0.3, 0.2])
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
