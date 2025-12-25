import random

import uav
from uav.simulator import CoverageSimulator
from uav.robot import UnderwaterRobot
from uav.planners import ResurfacingPlanner
from uav.policies import LawnmowerPolicy
from uav.noise_models import UniformNoiseModel
from uav.datatypes import Pose


def run_simulation_scenario(drift_prob, delta_prob, output_name, title):
    # Setup 100x100 playable -> 102x102 with walls
    ROWS, COLS = 102, 102
    grid = uav.environment.Grid(ROWS, COLS)

    # 1. Plan (Offline)
    # Generate the ideal path as a list of waypoints
    # Set start pose (we can default to 0,0 or based on scenario)
    start_x, start_y = (2, 2)  # Start slightly inside
    start_pose = Pose(start_x, start_y, 0.0)

    # Policy & Planner
    policy = LawnmowerPolicy(radius=4)
    # Using ResurfacingPlanner. It will generate the path on init/first step.
    planner = ResurfacingPlanner(
        policy=policy,
        start_pose=start_pose,
        grid_config={'rows': ROWS, 'cols': COLS},
        resurface_interval=50
    )

    robot = UnderwaterRobot(start_pose, planner, grid)
    noise_model = UniformNoiseModel(drift_prob=drift_prob, delta_prob=delta_prob)

    sim = CoverageSimulator(grid, robot, noise_model)

    # Execute
    # We run until the planner says it's done (returns 0 velocity) or max steps
    max_steps = 5000  # Increased allowance
    steps = 0

    print(f"[{title}] Simulating...")

    while steps < max_steps:
        # Check if done
        # Ideally the simulator loop runs blindly, but we need a break condition.
        # BlindPlanner returns 0 velocity when done.

        # We can peek at planner state or check action
        # But for strictly proper simulation, we run step()

        status = sim.step()
        steps += 1

        # Check if robot has stopped moving meaningfully
        # This is a bit hacky, normally we have a "MissionComplete" flag.
        # BlindPlanner halts by returning 0 velocity actions.
        if planner.current_waypoint_idx >= len(planner.waypoints) and planner.state == "NAVIGATING":
            # If planner ran out of waypoints AND is not about to resurface/replan, we might be done.
            # But ResurfacingPlanner replans?
            # For simpler termination: if coverage is high enough?
            # Or if replanning returns empty path (handled by policy returning []).
            if not planner.waypoints:
                break

    uav.plotting.plot_results(grid, sim.history, sim.true_map_coverage, title, output_name)

    valid_cells_count = (ROWS - 2) * (COLS - 2)
    scanned_count = status['covered_cells']
    print(f"[{title}] Coverage: {scanned_count}/{valid_cells_count} ({scanned_count / valid_cells_count:.2%}%)")

    # Tolerant success check
    return scanned_count == (valid_cells_count * 1.0)


def run_experiments():
    # 1. No Drift
    print("Running Baseline (No Drift)...")
    perfect = run_simulation_scenario(0.0, 0.0, 'coverage_no_drift.png', 'No Drift')

    # 2. With Drift
    print("\nRunning Experiment (With Drift)...")
    drifted_perfect = run_simulation_scenario(0.0, 0.0001, 'coverage_with_drift.png', 'With Drift (p+=0.0001)')

    print("\nSummary:")
    print(f"No Drift Ideal: {'Pass' if perfect else 'FAIL'}")
    print(f"Drift Impact: {'Holes Detected' if not drifted_perfect else 'Unexpected Perfect Coverage'}")


if __name__ == "__main__":
    random.seed(42)
    run_experiments()
