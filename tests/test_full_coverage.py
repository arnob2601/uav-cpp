import pytest
import uav
from uav.simulator import CoverageSimulator
from uav.robot import UnderwaterRobot
from uav.planners import BlindPlanner, generate_lawnmower_path_coordinates
from uav.noise_models import UniformNoiseModel
from uav.datatypes import Pose


def test_blind_planner_full_coverage_no_drift():
    # Setup 50x50 grid
    ROWS, COLS = 52, 52
    grid = uav.environment.Grid(ROWS, COLS)

    # 1. Generate full lawnmower path
    path_coords = generate_lawnmower_path_coordinates(ROWS, COLS, radius=4)
    assert len(path_coords) > 0, "Path generation failed"

    start_x, start_y = path_coords[0]
    start_pose = Pose(start_x, start_y, 0.0)

    # 2. Setup Robot with BlindPlanner (follows path exactly)
    planner = BlindPlanner(path_coords)
    robot = UnderwaterRobot(start_pose, planner, grid)

    # 3. No Drift
    noise_model = UniformNoiseModel(drift_prob=0.0, delta_prob=0.0)

    sim = CoverageSimulator(grid, robot, noise_model)

    # 4. Run Simulation
    max_steps = len(path_coords) * 2
    steps = 0

    while steps < max_steps:
        # Check if done
        if planner.current_waypoint_idx >= len(planner.waypoints):
            break

        sim.step()
        steps += 1

    # 5. Verify Coverage
    # Calculate valid cells (excluding walls if any, assuming standard Grid has walls)
    # The Grid class likely puts walls on the border?
    # Let's check environment.py or assume from previous context (WALL=0, UNEXPLORED=1 in plotting, but in sim logic?)
    # Simulator: true_map_coverage is zeros. _update_coverage sets to 1.0.

    # Check if we covered all "reachable" cells.
    # Since we generated a path for the whole grid, we should cover substantial amount.
    # Ideally 100% of the interior.

    # Determine valid area:
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

    print(f"Coverage: {scanned_cells}/{valid_cells} ({coverage_ratio:.2%})")

    # Allow small margin of error due to discretization or edge cases,
    # but "No Drift" + "Full Path" should be very close to 100%.
    # plot the coverage
    uav.plotting.plot_results(grid, sim.history, sim.true_map_coverage,
                              "Blind Planner Full Coverage No Drift", "data/full_coverage_no_drift.png",
                              surface_indices=sim.surface_indices)

    assert coverage_ratio == 1.0, f"Expected 100% coverage, got {coverage_ratio:.2%}"
