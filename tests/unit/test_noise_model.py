import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from uav.datatypes import Pose, Action, ActionType
from uav.noise_models import GaussianAccumulatedNoiseModel

class TestGaussianNoiseModel(unittest.TestCase):
    def setUp(self):
        self.noise_model = GaussianAccumulatedNoiseModel(drift_variance=0.1)

    def test_apply_drift_adds_noise(self):
        start_pose = Pose(0, 0, 0)
        action = Action(ActionType.MOVE, vx=1.0, vy=0.0, dt=1.0) # Moves 1 unit in X
        
        # Run multiple times to ensure randomness (heuristic check)
        poses = [self.noise_model.apply_drift(start_pose, action) for _ in range(100)]
        
        xs = [p.x for p in poses]
        ys = [p.y for p in poses]
        
        # Mean should be around 1.0 for X and 0.0 for Y
        self.assertAlmostEqual(np.mean(xs), 1.0, delta=0.5)
        self.assertAlmostEqual(np.mean(ys), 0.0, delta=0.5)
        
        # Variance should be non-zero
        self.assertGreater(np.var(xs), 0.0)
        self.assertGreater(np.var(ys), 0.0)

    def test_covariance_accumulation(self):
        action = Action(ActionType.MOVE, vx=1.0, vy=0.0, dt=1.0) # 1 meter
        cov_update = self.noise_model.get_drift_covariance(action)
        
        # Expect variance * distance
        # distance = 1.0
        # drift_variance = 0.1
        expected_val = 0.1 * 1.0
        
        self.assertAlmostEqual(cov_update[0, 0], expected_val)
        self.assertAlmostEqual(cov_update[1, 1], expected_val)

if __name__ == '__main__':
    unittest.main()
