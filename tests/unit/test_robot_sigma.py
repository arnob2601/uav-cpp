import unittest
import numpy as np
from uav.robot import UnderwaterRobot
from uav.datatypes import Pose, Action, ActionType
from uav.noise_models import GaussianAccumulatedNoiseModel


class MockPlanner:
    def get_next_action(self, state):
        return Action(ActionType.MOVE, vx=0, vy=0)


class MockEnv:
    def __init__(self, rows=10, cols=10):
        self.rows = rows
        self.cols = cols

    def get_neighbors(self, r, c, radius=5):
        return []


class TestRobotSigma(unittest.TestCase):
    def setUp(self):
        self.planner = MockPlanner()
        self.env = MockEnv()
        self.noise_model = GaussianAccumulatedNoiseModel(drift_variance=0.1)
        self.robot = UnderwaterRobot(Pose(0, 0), self.planner, self.env, self.noise_model)

    def test_sigma_initialization(self):
        self.assertTrue(np.array_equal(self.robot.sigma, np.zeros((2, 2))))

    def test_sigma_accumulation_on_move(self):
        action = Action(ActionType.MOVE, vx=1.0, vy=0.0, dt=1.0)  # 1 meter
        self.robot.update_internal_state(action)

        # Expected: drift_variance * distance * I
        expected_variance = 0.1 * 1.0
        self.assertAlmostEqual(self.robot.sigma[0, 0], expected_variance)
        self.assertAlmostEqual(self.robot.sigma[1, 1], expected_variance)
        self.assertEqual(self.robot.sigma[0, 1], 0.0)

    def test_sigma_reset_on_resurface(self):
        # Accumulate first
        action = Action(ActionType.MOVE, vx=1.0, vy=0.0, dt=1.0)
        self.robot.update_internal_state(action)
        self.assertGreater(self.robot.sigma[0, 0], 0.0)

        # Resurface
        resurface_action = Action(ActionType.RESURFACE)
        observation = {
            'true_pose': Pose(10, 10),
            'true_map_patch': np.zeros((10, 10))
        }
        self.robot.update_internal_state(resurface_action, observation)

        # Should be zeroed
        self.assertTrue(np.array_equal(self.robot.sigma, np.zeros((2, 2))))
        # Belief should be reset
        self.assertEqual(self.robot.belief_pose.x, 10)
        self.assertEqual(self.robot.belief_pose.y, 10)
