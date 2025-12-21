from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class ActionType(Enum):
    MOVE = auto()
    RESURFACE = auto()


@dataclass
class Pose:
    x: float
    y: float
    theta: float = 0.0


@dataclass
class Action:
    type: ActionType
    # For move: velocity or target waypoint (relative or absolute)
    # The user example showed vx, vy but also discussed "move to target" in existing code.
    # Let's keep it generic for now, aligned with the example:
    vx: float = 0.0
    vy: float = 0.0
    # Duration of action
    dt: float = 1.0

    # Optional: absolute target for higher level planners that don't do velocity control yet
    target_pose: 'Pose' = None


@dataclass
class RobotState:
    """The Robot's internal belief"""
    pose: Pose
    uncertainty_covariance: np.ndarray = None  # Optional for now
    perceived_occupancy_grid: np.ndarray = None
