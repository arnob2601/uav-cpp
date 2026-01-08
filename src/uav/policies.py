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
    def plan(self, current_pose: Pose, belief_map: np.ndarray, grid_config: dict) -> List[Tuple[float, float]]:
        """
        Generates a list of waypoints (x, y) to cover the remaining area.

        Args:
            current_pose: Robot's current belief pose.
            belief_map: 2D array where >0 indicates covered/scanned.
            grid_config: Dict with 'rows' and 'cols'.

        Returns:
            List of (x, y) tuples.
        """
        pass


class LawnmowerPolicy(PlanningPolicy):
    """
    Generates a lawnmower path for the bounding box of the unvisited area.
    """
    def __init__(self, radius=5):
        self.radius = radius

    def plan(self, current_pose: Pose, belief_map: np.ndarray, grid_config: dict) -> List[Tuple[float, float]]:
        rows, cols = grid_config['rows'], grid_config['cols']

        # 1. Identify unvisited areas
        unvisited_mask = (belief_map == 0)

        # If all visited, return empty
        if not np.any(unvisited_mask):
            return []

        # 2. Find bounding box of unvisited area
        r_indices, c_indices = np.where(unvisited_mask)

        min_r, max_r = np.min(r_indices), np.max(r_indices)
        min_c, max_c = np.min(c_indices), np.max(c_indices)

        # Add some margin/padding for the maneuver, but clamp to grid
        margin = self.radius
        p_min_r = max(0, min_r - margin)
        p_max_r = min(rows - 1, max_r + margin)
        p_min_c = max(0, min_c - margin)
        p_max_c = min(cols - 1, max_c + margin)

        ox = [p_min_c, p_max_c, p_max_c, p_min_c, p_min_c]
        oy = [p_min_r, p_min_r, p_max_r, p_max_r, p_min_r]

        resolution = 1.0 * self.radius

        # Call the underlying planner
        rx, ry = planner.planning(ox, oy, resolution)

        if not rx:
            return []

        # 3. Connect current pose to the start of the sweep path?
        path = []

        for x, y in zip(rx, ry):
            # Clamp
            cx = max(0, min(x, cols - 1))
            cy = max(0, min(y, rows - 1))
            path.append((cx, cy))

        return path


class TSPRegionPolicy(PlanningPolicy):
    """
    Identifies disjoint unvisited regions, generates paths for each,
    and solves TSP (Greedy) to visit them.
    This is now a wrapper around the core planner logic.
    """
    def __init__(self, radius=3):
        self.radius = radius

    def plan(self, current_pose: Pose, belief_map: np.ndarray, grid_config: dict) -> List[Tuple[float, float]]:
        return planner.plan_coverage_tsp(belief_map, grid_config, current_pose, self.radius)
