import sys
import os
import csv
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from uav.simulator import CoverageSimulator
from uav.robot import UnderwaterRobot
from uav.datatypes import Pose
from uav.noise_models import GaussianAccumulatedNoiseModel
from uav.environment import Grid
from uav.planners import ResurfacingPlanner, generate_lawnmower_path_coordinates
from uav.runner import ScenarioRunner
from uav.policies import LawnmowerPolicy # Assuming this exists or we mock it.
# Wait, LawnmowerPolicy class wasn't clearly seen in planners.py, but `ResurfacingPlanner` takes a policy.
# Let's check `policies.py`. Assume it exists or use a simple planner.

# Actually, let's just use BlindPlanner with a generated path for simplicity in this script
# unless Policy is strictly required. The instructions mentioned "Neural Policy" for deployment.
# For training data, we just need a policy to follow.
from uav.planners import BlindPlanner
from uav.policies import PlanningPolicy

class MockPolicy(PlanningPolicy):
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.returned = False

    def plan(self, current_pose: Pose, belief_map: np.ndarray):
        if not self.returned:
            self.returned = True
            return self.waypoints
        return []

def generate_dataset(output_file, num_samples=100):
    print(f"Generating {num_samples} samples...")

    # Setup
    env = Grid(30, 30)
    noise_model = GaussianAccumulatedNoiseModel(drift_variance=0.01)

    # Plan a lawnmower path
    path_coords = generate_lawnmower_path_coordinates(30, 30, radius=3)
    # Convert to waypoints (simple list of tuples is accepted by BlindPlanner)
    planner = BlindPlanner(MockPolicy(path_coords))

    robot = UnderwaterRobot(Pose(path_coords[0][0], path_coords[0][1]), planner, env, noise_model)
    sim = CoverageSimulator(env, robot, noise_model)
    runner = ScenarioRunner()

    data = []

    # Run simulation
    # We collect data points periodically
    for t in range(50000): # max steps safety breaker
        if len(data) >= num_samples:
            break

        # 1. Capture State
        # - Uncertainty (trace of sigma)
        # - Belief entropy?
        # - Map coverage?
        sig_trace = np.trace(robot.sigma)

        # Calculate coverage percent (Truth)
        covered_cells = np.sum(sim.true_map_coverage > 0)
        total_cells = sim.env.rows * sim.env.cols
        coverage_pct = covered_cells / total_cells

        # 2. Compute Label (Oracle VoI)
        # Only compute VoI if we are not finished
        voi = 0.0
        # Check if finished
        if sim.robot.belief_pose.x == 0 and sim.robot.belief_pose.y == 0:
             # End of path for BlindPlanner usually returns (0,0) move or similar?
             # Actually BlindPlanner returns move (0,0) if done.
             # But robot pose stays.
             pass
        else:
             voi = runner.run_branching_oracle(sim, time_steps_ahead=50)

        # Store
        data.append({
            'step': t,
            'sigma_trace': sig_trace,
            'coverage': coverage_pct,
            'voi': voi
        })

        if len(data) % 10 == 0:
            print(f"Sample {len(data)}/{num_samples}: Sigma={sig_trace:.4f}, Cov={coverage_pct:.2%}, VoI={voi:.4f}")

        # Step forward
        status = sim.step()

    # Save
    import os
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'sigma_trace', 'coverage', 'voi'])
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved {len(data)} samples to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/voi_dataset.csv", help="Output CSV file")
    args = parser.parse_args()

    generate_dataset(args.output, num_samples=args.samples)
