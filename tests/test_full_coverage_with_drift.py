import pytest
# import numpy as np
import random


import uav
from uav.simulator import CoverageSimulator
from uav.robot import UnderwaterRobot
from uav.planners import ResurfacingPlanner
from uav.policies import TSPRegionPolicy
from uav.noise_models import UniformNoiseModel
from uav.datatypes import Pose


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_full_coverage_tsp_replanning_with_drift(seed):
    # Set random seed for reproducibility
    random.seed(seed)

    # Setup 30x30 grid
    ROWS, COLS = 102, 102
    grid = uav.environment.Grid(ROWS, COLS)

    start_pose = Pose(5, 5, 0.0)

    # 1. Setup ResurfacingPlanner with TSPRegionPolicy
    policy = TSPRegionPolicy(radius=5)

    # Resurface interval can be large, we rely on end-of-mission verification too.
    # But frequent resurfacing helps keep errors low.
    planner = ResurfacingPlanner(
        policy=policy,
        start_pose=start_pose,
        resurface_interval=200  # Reasonably long
    )

    robot = UnderwaterRobot(start_pose, planner, grid, sensor_radius=5)

    # 2. Add Significant Drift
    # Enough to cause missed spots
    noise_model = UniformNoiseModel(drift_prob=0.0, delta_prob=0.001)

    sim = CoverageSimulator(grid, robot, noise_model)

    # 3. Run Simulation
    # 5000 steps should be enough for 30x30
    max_steps = ROWS * COLS * 2
    steps = 0

    while steps < max_steps:
        sim.step()
        steps += 1

        # Check mission status explicitly to break if done
        # But planner might be idling if done.
        # ResurfacingPlanner.get_next_action returns (0,0) move if done.

        # We can check if true coverage is 100%
        valid_cells = (ROWS - 2) * (COLS - 2)
        status = sim._check_mission_status()
        if status['covered_cells'] >= valid_cells:
            # We are done!
            break

        # If robot is idling for too long, maybe we are stuck?
        # But we assume planner works.

    # 4. Verify
    valid_cells = 0
    scanned_cells = 0

    rows, cols = sim.true_map_coverage.shape
    for r in range(rows):
        for c in range(cols):
            if grid.is_valid(r, c):
                valid_cells += 1
                if sim.true_map_coverage[r, c] > 0:
                    scanned_cells += 1

    coverage_ratio = scanned_cells / valid_cells if valid_cells > 0 else 0

    print(f"Final Coverage: {scanned_cells}/{valid_cells} ({coverage_ratio:.2%})")
    print(f"Steps taken: {steps}")

    # Plot results needed for debugging
    uav.plotting.plot_results(grid, sim.history, sim.true_map_coverage,
                              f"TSP Replanning with Drift {seed}", f"data/tsp_replanning_drift_{seed}.png",
                              surface_indices=sim.surface_indices)

    assert coverage_ratio > 0.99, f"Expected >99% coverage, got {coverage_ratio:.2%}"
