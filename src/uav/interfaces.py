from abc import ABC, abstractmethod
from typing import Optional
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

class BasePlanner(ABC):
    """Abstract base for all planning strategies"""
    
    def __init__(self):
        self.plan_queue = []

    @abstractmethod
    def get_next_action(self, belief_state: RobotState) -> Action:
        """
        Decides the next action based on the robot's belief state.
        """
        pass
