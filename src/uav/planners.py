from .interfaces import BasePlanner
from .datatypes import RobotState, Action, ActionType, Pose
import numpy as np
import uav.planner # For the legacy planning utils

class BlindPlanner(BasePlanner):
    """
    Follows a pre-defined path blindly.
    """
    def __init__(self, waypoints):
        """
        waypoints: list of (x, y) tuples or Pose objects
        """
        super().__init__()
        self.waypoints = list(waypoints) # Queue of waypoints (x, y)
        self.current_waypoint_idx = 0
        self.arrival_threshold = 1.0 # Distance to consider waypoint reached

    def get_next_action(self, belief_state: RobotState) -> Action:
        if self.current_waypoint_idx >= len(self.waypoints):
            # Finished
            return Action(type=ActionType.MOVE, vx=0.0, vy=0.0, dt=1.0)

        target = self.waypoints[self.current_waypoint_idx]

        # Calculate vector to target
        curr_x, curr_y = belief_state.pose.x, belief_state.pose.y
        target_x, target_y = target[0], target[1]

        dx = target_x - curr_x
        dy = target_y - curr_y

        dist = np.hypot(dx, dy)

        if dist < self.arrival_threshold:
            # Reached, move to next
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.waypoints):
                 return Action(type=ActionType.MOVE, vx=0.0, vy=0.0, dt=1.0)
            target = self.waypoints[self.current_waypoint_idx]
            target_x, target_y = target[0], target[1]
            dx = target_x - curr_x
            dy = target_y - curr_y
            dist = np.hypot(dx, dy)

        # Normalize velocity (unit step per dt=1)
        if dist > 0:
            vx = (dx / dist) # * speed
            vy = (dy / dist)
        else:
            vx, vy = 0.0, 0.0

        return Action(type=ActionType.MOVE, vx=vx, vy=vy, dt=1.0)

class IntervalPlanner(BlindPlanner):
    """
    Resurfaces every N steps.
    """
    def __init__(self, waypoints, resurface_interval=50):
        super().__init__(waypoints)
        self.resurface_interval = resurface_interval
        self.step_counter = 0

    def get_next_action(self, belief_state: RobotState) -> Action:
        self.step_counter += 1

        if self.step_counter % self.resurface_interval == 0:
            return Action(type=ActionType.RESURFACE, dt=5.0) # Assume resurfacing takes fixed dt time

        return super().get_next_action(belief_state)

def generate_lawnmower_path_coordinates(grid_rows, grid_cols, radius=5):
    """
    Generates a list of coordinates using the ported planner.
    """
    margin = radius
    min_r, max_r = 1 - margin, grid_rows - 2 + margin
    min_c, max_c = 1 - margin, grid_cols - 2 + margin

    ox = [min_c, max_c, max_c, min_c, min_c]
    oy = [min_r, min_r, max_r, max_r, min_r]

    resolution = 1.0 * radius

    # Plan using the old utils
    rx, ry = uav.planner.planning(ox, oy, resolution)

    dense_path = []

    if not rx:
        return []

    # Start point
    curr_r = int(round(ry[0]))
    curr_c = int(round(rx[0]))

    # Clamp
    curr_r = max(0, min(curr_r, grid_rows - 1))
    curr_c = max(0, min(curr_c, grid_cols - 1))

    dense_path.append((curr_c, curr_r)) # Store as x, y

    for i in range(1, len(rx)):
        target_r = int(round(ry[i]))
        target_c = int(round(rx[i]))

        target_r = max(0, min(target_r, grid_rows - 1))
        target_c = max(0, min(target_c, grid_cols - 1))

        while (curr_r, curr_c) != (target_r, target_c):
            dr = target_r - curr_r
            dc = target_c - curr_c

            # Step size 1
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_c = 0 if dc == 0 else (1 if dc > 0 else -1)

            curr_r += step_r
            curr_c += step_c
            dense_path.append((curr_c, curr_r)) # x, y

    return dense_path
