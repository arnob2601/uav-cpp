import math
import random
import numpy as np

import uav
from uav.simulator import CoverageSimulator
from uav.robot import UnderwaterRobot
from uav.planners import BlindPlanner, generate_lawnmower_path_coordinates
from uav.noise_models import UniformNoiseModel
from uav.datatypes import Pose


def run_simulation_scenario(drift_prob, output_name, title):
    # Setup 100x100 playable -> 102x102 with walls
    ROWS, COLS = 102, 102
    grid = uav.environment.Grid(ROWS, COLS)

    # 1. Plan (Offline)
    # Generate the ideal path as a list of waypoints
    ideal_path_coords = generate_lawnmower_path_coordinates(ROWS, COLS)

    if not ideal_path_coords:
        print("Failed to generate path")
        return False

    # 2. Setup Robot and Simulator
    start_x, start_y = ideal_path_coords[0]
    start_pose = Pose(start_x, start_y, 0.0)

    planner = BlindPlanner(ideal_path_coords)
    robot = UnderwaterRobot(start_pose, planner, grid)
    noise_model = UniformNoiseModel(drift_prob=drift_prob)

    sim = CoverageSimulator(grid, robot, noise_model)

    # Execute
    # We run until the planner says it's done (returns 0 velocity) or max steps
    max_steps = len(ideal_path_coords) * 2 # Safety margin
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
        if planner.current_waypoint_idx >= len(planner.waypoints):
             break

    uav.plotting.plot_results(grid, sim.history, sim.true_map_coverage, title, output_name)

    valid_cells_count = (ROWS - 2) * (COLS - 2)
    scanned_count = status['covered_cells']
    print(f"[{title}] Coverage: {scanned_count}/{valid_cells_count} ({scanned_count/valid_cells_count:.2%}%)")

    # Tolerant success check
    return scanned_count >= (valid_cells_count * 0.95)

def run_experiments():
    # 1. No Drift
    print("Running Baseline (No Drift)...")
    perfect = run_simulation_scenario(0.0, 'coverage_no_drift.png', 'No Drift')

    # 2. With Drift
    print("\nRunning Experiment (With Drift)...")
    drifted_perfect = run_simulation_scenario(0.1, 'coverage_with_drift.png', 'With Drift (p=0.1)')

    print("\nSummary:")
    print(f"No Drift Ideal: {'Pass' if perfect else 'FAIL'}")
    print(f"Drift Impact: {'Holes Detected' if not drifted_perfect else 'Unexpected Perfect Coverage'}")

if __name__ == "__main__":
    random.seed(42)
    run_experiments()
