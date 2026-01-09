import numpy as np
from .simulator import CoverageSimulator
from .datatypes import Action, ActionType


class ScenarioRunner:
    def __init__(self):
        pass

    def run_branching_oracle(self, sim: CoverageSimulator, time_steps_ahead: int = 200) -> float:
        """
        Calculates the Value of Information (VoI) by comparing:
        1. Continuing without surfacing (Q_no_surface)
        2. Surfacing now, then continuing (Q_surface)

        Returns:
            voi = Cost(No Surface) - Cost(Surface)
            Positive value means Surfacing is better (saves distance/cost).
        """

        # Branch 1: No Surface
        sim_no_surface = sim.clone()
        # Just continue executing the policy
        cost_no_surface = self._simulate_and_compute_cost(sim_no_surface, time_steps_ahead)

        # Branch 2: Surface Now
        sim_surface = sim.clone()

        # We can just manually call the components of step.
        action = Action(ActionType.RESURFACE, dt=5.0)

        # Physics Step
        observation = None
        # Resurface logic in Sim
        observation = {
            'true_pose': sim_surface.true_pose,
            'true_map_patch': sim_surface.true_map_coverage.copy()
        }
        sim_surface.noise_model.drift_prob = 0.0  # If model has this

        # Robot Update
        sim_surface.robot.update_internal_state(action, observation)

        # Cost of surfacing
        cost_surface_action = 20.0  # Equivalent meters cost (as per specs)

        # Continue simulation
        cost_continue = self._simulate_and_compute_cost(sim_surface, time_steps_ahead)

        total_cost_surface = cost_surface_action + cost_continue

        voi = cost_no_surface - total_cost_surface
        return voi

    def _simulate_and_compute_cost(self, sim: CoverageSimulator, steps: int) -> float:
        """
        Run simulation for N steps (or until finished) and compute total recovery cost.
        """
        distance_traveled = 0.0

        for _ in range(steps):
            # Record pose
            prev_pose = sim.true_pose

            # Step
            _ = sim.step()

            # Calculate distance moved (True distance)
            curr_pose = sim.true_pose
            dist = np.hypot(curr_pose.x - prev_pose.x, curr_pose.y - prev_pose.y)
            distance_traveled += dist

            # Check if done? (Assuming fixed steps or mission completion check)
            # If planner is finished, it returns (0,0) move.
            # We can check action?

        # Add Recovery Cost (The "Hole" Problem)
        dist_recovery = self.calculate_recovery_cost(sim)

        return distance_traveled + dist_recovery

    def calculate_recovery_cost(self, sim: CoverageSimulator) -> float:
        """
        Calculate the cost to visit all "holes" (False Positives).
        Holes are cells that are NOT covered in Truth, but Robot THINKS are covered.

        Actually, specs say: "Compare G_belief vs G_truth"
        "Identify 'holes' (cells in Belief but not in Truth)" -> This implies Robot thinks it covered them (Belief=1), but Truth=0.
        Wait, coverage is usually about ensuring Truth=1.
        If Robot thinks Belief=1 but Truth=0, it skipped it. Correct.
        """
        g_truth = sim.true_map_coverage
        g_belief = sim.robot.perceived_map

        # Mask: Belief says Covered (1) AND Truth says Uncovered (0)
        # Assuming 1.0 is covered, 0.0 is not.
        holes_mask = (g_belief > 0.5) & (g_truth < 0.5)

        hole_indices = np.argwhere(holes_mask)
        if len(hole_indices) == 0:
            return 0.0

        # Plan path to visit all holes
        # Salesman problem. Greedy approximation.
        # Start at current true pose
        current_pos = (sim.true_pose.y, sim.true_pose.x)  # r, c

        remaining_holes = [tuple(h) for h in hole_indices]  # List of (r, c)

        total_dist = 0.0

        # Simple Greedy TSP
        while remaining_holes:
            # Find closest hole
            dists = [np.hypot(h[0] - current_pos[0], h[1] - current_pos[1]) for h in remaining_holes]
            min_idx = np.argmin(dists)
            nearest = remaining_holes[min_idx]
            dist = dists[min_idx]

            total_dist += dist
            current_pos = nearest
            remaining_holes.pop(min_idx)

            # Assuming we cover it by visiting.
            # In reality we have a footprint.
            # But cost estimation can assume point visit or we remove neighbors.
            # For simplicity: point visit.

        return total_dist
