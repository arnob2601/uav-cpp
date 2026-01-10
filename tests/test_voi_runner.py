import unittest
from uav.runner import ScenarioRunner
from uav.simulator import CoverageSimulator
from uav.robot import UnderwaterRobot
from uav.datatypes import Pose, Action, ActionType
from uav.noise_models import GaussianAccumulatedNoiseModel
from uav.environment import Grid


class MockPlanner:
    def get_next_action(self, state):
        # Move right
        return Action(ActionType.MOVE, vx=1, vy=0)

    def clone(self):
        return MockPlanner()


class TestVoIRunner(unittest.TestCase):
    def setUp(self):
        self.env = Grid(20, 20)
        self.planner = MockPlanner()
        self.noise_model = GaussianAccumulatedNoiseModel(drift_variance=0.0)  # No noise for predictable testing first
        self.robot = UnderwaterRobot(Pose(5, 5), self.planner, self.env, self.noise_model, sensor_radius=5)
        self.sim = CoverageSimulator(self.env, self.robot, self.noise_model)
        self.runner = ScenarioRunner()

    def test_recovery_cost_calculation(self):
        # Manually induce a "hole"
        # Belief says (6,6) is covered
        self.sim.robot.perceived_map[6, 6] = 1.0
        # Truth says (6,6) is NOT covered
        self.sim.true_map_coverage[6, 6] = 0.0

        # Current pose is (5,5)
        # Distance to (6,6) roughly sqrt(2) = 1.414

        cost = self.runner.calculate_recovery_cost(self.sim)
        self.assertAlmostEqual(cost, 1.414, delta=0.1)

    def test_voi_oracle_basic(self):
        # With cost_surface = 20.0 and no noise, surfacing is just a cost penalty.
        # So VoI should be negative (NoSurface is cheaper).

        voi = self.runner.run_branching_oracle(self.sim, time_steps_ahead=5)

        # Q_no_surface cost approx 5 (distance) + 0 (recovery) = 5
        # Q_surface cost = 20 + 5 = 25
        # VoI = 5 - 25 = -20

        self.assertLess(voi, 0)
        self.assertAlmostEqual(voi, -20.0, delta=2.0)
