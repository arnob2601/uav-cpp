from .interfaces import NoiseModel
from .datatypes import Pose, Action
import random


class UniformNoiseModel(NoiseModel):
    def __init__(self, drift_prob=0.0, delta_prob=0.0):
        self.drift_prob = drift_prob
        self.delta_prob = delta_prob

    def apply_drift(self, true_pose: Pose, action: Action) -> Pose:
        """
        Applies random drift to the movement.
        """
        # Basic kinematics first
        new_x = true_pose.x + action.vx * action.dt
        new_y = true_pose.y + action.vy * action.dt
        new_theta = true_pose.theta

        # Apply drift
        # Example: 10% chance to drift 1 unit in a random direction
        if random.random() < self.drift_prob:
            # Drift direction
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            new_x += dx
            new_y += dy

        # Increase drift probability for next time
        self.drift_prob = min(1.0, self.drift_prob + self.delta_prob)

        return Pose(new_x, new_y, new_theta)

    def predict_motion(self, belief_pose: Pose, action: Action) -> Pose:
        """
        Robot assumes no drift.
        """
        new_x = belief_pose.x + action.vx * action.dt
        new_y = belief_pose.y + action.vy * action.dt
        return Pose(new_x, new_y, belief_pose.theta)
