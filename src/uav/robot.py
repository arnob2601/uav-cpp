import copy
import numpy as np
from .datatypes import Pose, Action, ActionType, RobotState
from .interfaces import BasePlanner


class UnderwaterRobot:
    def __init__(self, start_pose: Pose, planner: BasePlanner, env, noise_model=None):
        self.planner = planner
        self.env = env
        self.noise_model = noise_model
        # Belief State (Where robot thinks it is)
        self.belief_pose = start_pose
        # Sigma: Uncertainty covariance (2x2)
        self.sigma = np.zeros((2, 2))

        # 0=Unexplored, 1=Free, 2=Obstacle. Initialize with 0.
        self.perceived_map = np.zeros((env.rows, env.cols))
        self.accumulated_error = 0.0

        # Mark initial position as covered/free
        self._mark_coverage(self.belief_pose)

    def step(self) -> Action:
        """Consult planner for next move"""
        return self.planner.get_next_action(
            RobotState(self.belief_pose, self.sigma, self.perceived_map)
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
            # Reset uncertainty on surfacing
            self.sigma = np.zeros((2, 2))

            # Merge true map data into perceived map
            if 'true_map_patch' in observation:
                self.perceived_map = observation['true_map_patch']
        else:
            # DEAD RECKONING (Drift accumulates here because we don't know it)
            self.belief_pose = self._apply_kinematics(self.belief_pose, action)

            # Update uncertainty (Sigma)
            if self.noise_model:
                # Add covariance Q_t
                if hasattr(self.noise_model, 'get_drift_covariance'):
                    Q_t = self.noise_model.get_drift_covariance(action)
                    self.sigma += Q_t
                else:
                    # Fallback or simple constant if model doesn't support it
                    pass

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

    def clone(self):
        """
        Create a deep copy of the robot state for branching.
        """
        # 1. New robot instance (init with same static objects)
        # Note: Planner and Env are typically shared/read-only or we should check if they need cloning.
        # For VoI, the Planner policy might be same, but internal state?
        # If Planner has state (queue), we might need to deepcopy it or accept it's reset?
        # In this task, we assume planner is stateless or we just want to branch the belief.

        # We need to construct it carefully to avoid re-init logic if side effects exist.
        # But __init__ sets up start_pose.

        new_robot = UnderwaterRobot(
            start_pose=copy.deepcopy(self.belief_pose),
            planner=self.planner.clone(),  # Deep copy planner
            env=self.env,  # Share environment map
            noise_model=self.noise_model.clone() if self.noise_model else None
        )

        # 2. Copy Internal State
        new_robot.perceived_map = self.perceived_map.copy()
        new_robot.sigma = self.sigma.copy()
        new_robot.accumulated_error = self.accumulated_error

        return new_robot

    def _mark_coverage(self, pose: Pose):
        """
        Mark the cell corresponding to the pose as Free/Covered.
        """
        # Map indices
        r = int(round(pose.y))
        c = int(round(pose.x))

        # Simulate sensor radius = 3
        neighbors = self.env.get_neighbors(r, c)
        rows, cols = self.perceived_map.shape
        for nr, nc in neighbors:
            if 0 <= nr < rows and 0 <= nc < cols:
                self.perceived_map[nr, nc] = 1.0
