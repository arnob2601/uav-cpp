import numpy as np
from .datatypes import Pose, Action, ActionType, RobotState
from .interfaces import BasePlanner


class UnderwaterRobot:
    def __init__(self, start_pose: Pose, planner: BasePlanner, env):
        self.planner = planner
        self.env = env
        # Belief State (Where robot thinks it is)
        self.belief_pose = start_pose
        # 0=Unexplored, 1=Free, 2=Obstacle. Initialize with 0.
        self.perceived_map = np.zeros((env.rows, env.cols))
        self.accumulated_error = 0.0

        # Mark initial position as covered/free
        self._mark_coverage(self.belief_pose)

    def step(self) -> Action:
        """Consult planner for next move"""
        return self.planner.get_next_action(
            RobotState(self.belief_pose, None, self.perceived_map)
        )

    def update_internal_state(self, action: Action, observation=None):
        """
        Called after Simulator executes action.
        If Action == MOVE: Update belief using dead reckoning (simple kinematics).
        If Action == RESURFACE: 'observation' contains True Pose and True Map patch.
        """
        if action.type == ActionType.RESURFACE and observation:
            # GROUND TRUTH RESET
            self.belief_pose = observation['true_pose']
            # Merge true map data into perceived map
            if 'true_map_patch' in observation:
                self.perceived_map = self._merge_maps(self.perceived_map, observation['true_map_patch'])
        else:
            # DEAD RECKONING (Drift accumulates here because we don't know it)
            self.belief_pose = self._apply_kinematics(self.belief_pose, action)
            # Update map based on where we THINK we are
            self._mark_coverage(self.belief_pose)

    def _apply_kinematics(self, pose: Pose, action: Action) -> Pose:
        """
        Simple motion model: new_pos = old_pos + velocity * dt
        """
        if action.type == ActionType.MOVE:
            # Simple Euler integration
            # If we had theta/velocity control:
            # dx = action.v * cos(pose.theta) * dt
            # But action has vx, wy directly?
            # User example showed vx, vy.
            # Let's assume global frame velocities for now or robot frame?
            # "vx: float = 0.0" in Action.
            # Let's assume simplistically these are delta x, delta y for the grid if dt=1

            new_x = pose.x + action.vx * action.dt
            new_y = pose.y + action.vy * action.dt
            new_theta = pose.theta  # No rotation yet

            return Pose(new_x, new_y, new_theta)
        return pose

    def _mark_coverage(self, pose: Pose):
        """
        Mark the cell corresponding to the pose as Free/Covered.
        """
        # Map indices
        r = int(round(pose.y))
        c = int(round(pose.x))

        # Simulate sensor radius = 3
        neighbors = self.env.get_neighbors(r, c, radius=3)
        rows, cols = self.perceived_map.shape
        for nr, nc in neighbors:
            if 0 <= nr < rows and 0 <= nc < cols:
                self.perceived_map[nr, nc] = 1.0

    def _merge_maps(self, internal_map, truth_patch):
        """
        Merge truth_patch into internal_map.
        For simplicity, overwrite internal with truth where truth is known/provided.
        If truth_patch is the whole map, just replace.
        """
        # If truth_patch is same shape and contains all info:
        if truth_patch.shape == internal_map.shape:
            # Logic: if truth_patch says "Obstacle" (2) or "Free" (1), trust it.
            # If truth_patch is 0 (Unexplored), keep our belief?
            # Or is truth_patch ONLY what was seen?

            # Assuming truth_patch overrides:
            return np.where(truth_patch != 0, truth_patch, internal_map)

        return internal_map
