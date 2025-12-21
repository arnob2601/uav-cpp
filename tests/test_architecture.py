import pytest
import numpy as np
from uav.datatypes import Pose, Action, ActionType, RobotState
from uav.robot import UnderwaterRobot
from uav.simulator import CoverageSimulator
from uav.planners import BlindPlanner
from uav.noise_models import UniformNoiseModel
from uav.environment import Grid
from uav.interfaces import BasePlanner

class MockEnv:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
    def get_neighbors(self, r, c, radius):
        # Return center only for testing specific logic, or mocked neighbors
        return [(r, c)]
    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

def test_robot_dead_reckoning():
    # Setup
    env = MockEnv(10, 10)
    start_pose = Pose(0, 0, 0)
    planner = BlindPlanner([]) # No plan needed for manual stepping
    robot = UnderwaterRobot(start_pose, planner, env)
    
    # Action: Move 1 unit x
    action = Action(type=ActionType.MOVE, vx=1.0, vy=0.0, dt=1.0)
    
    # Update internal state (no observation)
    robot.update_internal_state(action, observation=None)
    
    assert robot.belief_pose.x == 1.0
    assert robot.belief_pose.y == 0.0
    
    # Check map update (mock env returns only self via get_neighbors override logic? No, mocked to return center)
    # belief is y=0, x=1. -> r=0, c=1.
    assert robot.perceived_map[0, 1] == 1.0

def test_simulator_drift():
    # Setup
    env = MockEnv(10, 10)
    start_pose = Pose(0, 0, 0)
    planner = BlindPlanner([(0,0)])
    robot = UnderwaterRobot(start_pose, planner, env)
    
    # Force drift in noise model
    class DeterministicNoiseModel(UniformNoiseModel):
        def apply_drift(self, true_pose, action):
            # Apply normal motion + 1.0 x drift
            new_x = true_pose.x + action.vx * action.dt + 1.0
            new_y = true_pose.y + action.vy * action.dt
            return Pose(new_x, new_y, true_pose.theta)
            
    noise_model = DeterministicNoiseModel()
    sim = CoverageSimulator(env, robot, noise_model)
    
    # Step
    # Robot wants to move 1 unit X (vx=1)
    # We cheat and call simulator step?
    # Or just manually trigger drift application logic?
    # Simulator.step() calls robot.step(). 
    # Robot step consults planner.
    # BlindPlanner with waypoints [(10, 0)] might produce vx=1.
    
    planner.waypoints = [(10.0, 0.0)] # Target far away
    
    action = robot.step()
    assert action.vx > 0
    
    # Sim step
    sim.step()
    
    # Robot belief should be x=1 (0 + 1)
    # True pose should be x=2 (0 + 1 + 1 drift)
    
    assert robot.belief_pose.x == pytest.approx(1.0)
    assert sim.true_pose.x == pytest.approx(2.0)

def test_boundary_enforcement():
    # Setup
    env = MockEnv(10, 10)
    # Start at right edge (cols=10, valid indices 0..9)
    start_pose = Pose(9.0, 5.0, 0)
    planner = BlindPlanner([])
    robot = UnderwaterRobot(start_pose, planner, env)
    
    # Simple model that just moves as requested (no random drift for this test to be deterministic)
    class NoDriftModel(UniformNoiseModel):
        def apply_drift(self, true_pose, action):
            return Pose(true_pose.x + action.vx * action.dt, true_pose.y + action.vy * action.dt, true_pose.theta)
            
    sim = CoverageSimulator(env, robot, NoDriftModel())
    
    # Action: Move Right (positive X)
    action = Action(type=ActionType.MOVE, vx=1.0, vy=0.0, dt=1.0)
    
    # We need to inject this action or force the robot to take it.
    # Since robot uses planner, let's just bypass robot.step() and call sim logic directly 
    # OR mock the planner to return this action.
    class RightPlanner(BasePlanner):
        def get_next_action(self, belief_state):
            return Action(type=ActionType.MOVE, vx=1.0, vy=0.0, dt=1.0)
            
    robot.planner = RightPlanner()
    
    sim.step()
    
    # Expectation: The robot should NOT be at 10.0 (out of bounds)
    # It should be clamped to something < 10 or remain at 9.0
    
    assert sim.true_pose.x < 10.0
    # Also check Y didn't change
    assert sim.true_pose.y == 5.0
