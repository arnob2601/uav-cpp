import pytest
import uav
from uav.simulator import CoverageSimulator
from uav.robot import UnderwaterRobot
from uav.policies import TSPRegionPolicy
from uav.planners import BlindPlanner
from uav.noise_models import UniformNoiseModel
from uav.datatypes import Pose


def test_blind_planner_full_coverage_no_drift():
    # Setup 30x30 grid (smaller for faster TSP test)
    ROWS, COLS = 32, 32
    grid = uav.environment.Grid(ROWS, COLS)

    start_pose = Pose(2, 2, 0.0)

    # 1. Setup Robot with BlindPlanner and TSP Policy
    # The BlindPlanner will invoke the policy once at start.
    policy = TSPRegionPolicy(radius=3)
    planner = BlindPlanner(policy, start_pose)
    robot = UnderwaterRobot(start_pose, planner, grid)

    # 2. No Drift
    noise_model = UniformNoiseModel(drift_prob=0.0, delta_prob=0.0)

    sim = CoverageSimulator(grid, robot, noise_model)

    # 3. Run Simulation
    # Allow enough steps
    max_steps = ROWS * COLS
    steps = 0

    while steps < max_steps:
        # Check if done
        if planner.initial_planning_done and planner.current_waypoint_idx >= len(planner.waypoints):
            break

        sim.step()
        steps += 1

    # 4. Verify Coverage

    # Check valid cells only (exclude borders if grid logic handles them)
    # The Grid has a 1-pixel border.

    # We inspect the true_map coverage just to be sure
    scanned_in_valid = 0
    valid_cells = 0
    rows, cols = sim.true_map_coverage.shape
    for r in range(rows):
        for c in range(cols):
            if grid.is_valid(r, c):
                valid_cells += 1
                if sim.true_map_coverage[r, c] > 0:
                    scanned_in_valid += 1

    coverage_ratio = scanned_in_valid / valid_cells if valid_cells > 0 else 0
    print(f"Coverage: {scanned_in_valid}/{valid_cells} ({coverage_ratio:.2%})")

    uav.plotting.plot_results(grid, sim.history, sim.true_map_coverage,
                              "Blind Planner TSP Coverage No Drift", "data/full_coverage_tsp_no_drift.png",
                              surface_indices=sim.surface_indices)

    # TSP Policy should theoretically achieve high coverage
    assert coverage_ratio > 0.95, f"Expected high coverage, got {coverage_ratio:.2%}"
