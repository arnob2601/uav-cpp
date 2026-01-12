import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from uav.simulator import CoverageSimulator
from uav.robot import UnderwaterRobot
from uav.datatypes import Pose, Action, ActionType
from uav.noise_models import GaussianAccumulatedNoiseModel
from uav.environment import Grid

class MockPlanner:
    def get_next_action(self, state):
        return Action(ActionType.MOVE, vx=1, vy=0)
    
    def clone(self):
        return MockPlanner()

class TestSimBranching(unittest.TestCase):
    def setUp(self):
        self.env = Grid(10, 10)
        self.planner = MockPlanner()
        self.noise_model = GaussianAccumulatedNoiseModel(drift_variance=0.1)
        self.robot = UnderwaterRobot(Pose(5,5), self.planner, self.env, self.noise_model, sensor_radius=5)
        self.sim = CoverageSimulator(self.env, self.robot, self.noise_model)

    def test_clone_independence(self):
        # 1. Step the original sim
        self.sim.step()
        original_pose_x = self.sim.true_pose.x
        
        # 2. Clone
        sim_branch = self.sim.clone()
        
        # 3. Step ONLY the branch
        sim_branch.step()
        
        # 4. Verify Divergence
        # Original sim should stay at original_pose_x (from step 1) 
        # Wait, if I didn't step original again, it should be same.
        # But let's verify branch moved further.
        
        self.assertNotEqual(sim_branch.true_pose.x, self.sim.true_pose.x)
        self.assertGreater(sim_branch.true_pose.x, self.sim.true_pose.x)
        
        # 5. Verify Modifying Branch doesn't affect Original Robot State
        sim_branch.robot.belief_pose.x = 999
        self.assertNotEqual(self.sim.robot.belief_pose.x, 999)

    def test_sigma_independence(self):
        sim_branch = self.sim.clone()
        
        # Modify branch sigma
        sim_branch.robot.sigma[0,0] = 100.0
        
        # Original should be low
        self.assertNotEqual(self.sim.robot.sigma[0,0], 100.0)

if __name__ == '__main__':
    unittest.main()
