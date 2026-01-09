from abc import ABC, abstractmethod
import numpy as np
from .datatypes import Pose
from . import planner  # The ported grid planner
from typing import List, Tuple


class PlanningPolicy(ABC):
    """
    Abstract base class for planning policies.
    """
    @abstractmethod
    @abstractmethod
    def plan(self, current_pose: Pose, belief_map: np.ndarray) -> List[Tuple[float, float]]:
        """
        Generates a list of waypoints (x, y) to cover the remaining area.

        Args:
            current_pose: Robot's current belief pose.
            belief_map: 2D array where >0 indicates covered/scanned.

        Returns:
            List of (x, y) tuples.
        """
        pass


class StaticPathPolicy(PlanningPolicy):
    """
    Returns a predefined path once, then empty.
    Useful for testing BlindPlanner with static paths.
    """
    def __init__(self, waypoints: List[Tuple[float, float]]):
        self.waypoints = waypoints
        self.returned = False

    def plan(self, current_pose: Pose, belief_map: np.ndarray) -> List[Tuple[float, float]]:
        if not self.returned:
            self.returned = True
            return self.waypoints
        return []


class TSPRegionPolicy(PlanningPolicy):
    """
    Identifies disjoint unvisited regions, generates paths for each,
    and solves TSP (Greedy) to visit them.
    This is now a wrapper around the core planner logic.
    """
    def __init__(self, radius=3):
        self.radius = radius

    def plan(self, current_pose: Pose, belief_map: np.ndarray) -> List[Tuple[float, float]]:
        return planner.plan_coverage_tsp(belief_map, current_pose, radius=self.radius)
