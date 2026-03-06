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
        self.test_recolor_restore_uses_full_array_length()
        self.test_parameter_node_restore_round_trip()

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

    def test_recolor_restore_uses_full_array_length(self):
        signedDistances = vtk.vtkDoubleArray()
        for value in (-8.0, -4.0, -1.0, 3.0):
            signedDistances.InsertNextValue(value)
        backup = vtk.vtkDoubleArray()
        backup.DeepCopy(signedDistances)

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
