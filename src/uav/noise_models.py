from .interfaces import NoiseModel
from .datatypes import Pose, Action
import random
import numpy as np


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


class GaussianAccumulatedNoiseModel(NoiseModel):
    def __init__(self, drift_variance=0.1):
        """
        drift_variance: Variance accumulation per meter traveled.
        """
        self.drift_variance = drift_variance

    def apply_drift(self, true_pose: Pose, action: Action) -> Pose:
        # 1. Deterministic move
        dx = action.vx * action.dt
        dy = action.vy * action.dt

        # 2. Add Gaussian Noise
        # Noise scales with distance? Or per step?
        # "true_pose += action + random_gaussian_noise()"
        # Usually noise is proportional to sqrt(distance) for standard deviation,
        # or variance is proportional to distance.
        # Let's assume the noise added in this step corresponds to the variance accumulated.
        dist = (dx**2 + dy**2)**0.5
        if dist > 0:
            std_dev = (self.drift_variance * dist)**0.5
            noise_x = random.gauss(0, std_dev)
            noise_y = random.gauss(0, std_dev)
        else:
            noise_x = 0
            noise_y = 0

        new_x = true_pose.x + dx + noise_x
        new_y = true_pose.y + dy + noise_y

        return Pose(new_x, new_y, true_pose.theta)

    def predict_motion(self, belief_pose: Pose, action: Action) -> Pose:
        """
        Dead reckoning (mean estimate).
        """
        new_x = belief_pose.x + action.vx * action.dt
        new_y = belief_pose.y + action.vy * action.dt
        return Pose(new_x, new_y, belief_pose.theta)

    def get_drift_covariance(self, action: Action) -> np.ndarray:
        """
        Returns the 2x2 covariance matrix to ADD to the robot's belief sigma.
        Q_t = drift_variance * distance * Identity
        """
        dx = action.vx * action.dt
        dy = action.vy * action.dt
        dist = (dx**2 + dy**2)**0.5

        variance_step = self.drift_variance * dist
        return np.array([
            [variance_step, 0.0],
            [0.0, variance_step]
        ])
