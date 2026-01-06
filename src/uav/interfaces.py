from abc import ABC, abstractmethod
from .datatypes import Pose, Action, RobotState


class NoiseModel(ABC):
    """Abstract base for drift and error models"""
    @abstractmethod
    def apply_drift(self, true_pose: Pose, action: Action) -> Pose:
        """Calculates where the robot ACTUALLY goes"""
        pass

    @abstractmethod
    def predict_motion(self, belief_pose: Pose, action: Action) -> Pose:
        """Calculates where the robot THINKS it went (Dead Reckoning)"""
        pass

    def clone(self):
        """Returns a deep copy of the noise model state."""
        import copy
        return copy.deepcopy(self)


class BasePlanner(ABC):
    """Abstract base for all planning strategies"""
    def __init__(self, policy=None):
        self.plan_queue = []
        self.policy = policy

    @abstractmethod
    def get_next_action(self, belief_state: RobotState) -> Action:
        """
        Decides the next action based on the robot's belief state.
        """
        pass

    def clone(self):
        """Returns a deep copy of the planner state."""
        import copy
        return copy.deepcopy(self)
