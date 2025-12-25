from .datatypes import Pose, ActionType
from .robot import UnderwaterRobot
from .interfaces import NoiseModel
import numpy as np


class CoverageSimulator:
    def __init__(self, env_grid, robot: UnderwaterRobot, noise_model: NoiseModel):
        self.env = env_grid
        self.robot = robot
        self.noise_model = noise_model

        # Ground Truths
        self.true_pose = robot.belief_pose  # Start synced
        self.true_map_coverage = np.zeros((env_grid.rows, env_grid.cols))
        self._update_coverage(self.true_pose)  # Mark initial position as covered
        self.history = [self.true_pose]

    def step(self):
        # 1. Get intention from Robot
        action = self.robot.step()

        # 2. Simulator calculates Physical outcome
        observation = None

        if action.type == ActionType.MOVE:
            # Move Ground Truth (Apply Drift)
            new_pose = self.noise_model.apply_drift(self.true_pose, action)

            # Check for collision/boundary
            r = int(round(new_pose.y))
            c = int(round(new_pose.x))

            if self.env.is_valid(r, c):
                self.true_pose = new_pose
                # Update Ground Truth Coverage
                self._update_coverage(self.true_pose)
                self.history.append(self.true_pose)
            else:
                # Collision! Robot stays put (or slides, but simple block for now)
                pass

        elif action.type == ActionType.RESURFACE:
            # Teleport robot to surface (or simulate ascent)
            # Prepare 'GPS' data and Map correction to send back to robot
            # In this simple sim, we return the Perfect Pose and the Perfect Map (of what was covered)

            # Simulate "Seeing" the map correction?
            # Or just resetting position?
            # User request: "observation contains True Pose and True Map patch"

            observation = {
                'true_pose': self.true_pose,
                'true_map_patch': self.true_map_coverage.copy()  # Simplification: gives entire truth of what's covered
            }
            self.noise_model.drift_prob = 0.0  # Reset localization noise/drift probability after resurfacing

        # 3. Robot updates its belief (Dead Reckoning or Reset)
        self.robot.update_internal_state(action, observation)

        return self._check_mission_status()

    def _update_coverage(self, pose: Pose):
        """
        Mark the ground truth map as covered at this pose.
        """
        r = int(round(pose.y))
        c = int(round(pose.x))
        neighbors = self.env.get_neighbors(r, c, radius=3)
        for nr, nc in neighbors:
            self.true_map_coverage[nr, nc] = 1.0

    def _check_mission_status(self):
        """
        Return status dict
        """
        # Count covered cells
        covered = np.sum(self.true_map_coverage > 0)
        # Total valid cells (approximate based on env)
        # Assuming environment has explicit bounds or we count valid cells
        # env.rows * env.cols might include walls.

        return {
            "covered_cells": covered,
            "true_pose": self.true_pose
        }
