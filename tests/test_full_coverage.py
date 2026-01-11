import pytest
import uav
from uav.simulator import CoverageSimulator
from uav.robot import UnderwaterRobot
from uav.policies import TSPRegionPolicy
from uav.planners import BlindPlanner
from uav.noise_models import UniformNoiseModel
from uav.datatypes import Pose


@pytest.mark.parametrize(
    "grid_size, radius",
    [
        ((32, 32), 3),
        ((32, 32), 4),
        ((32, 32), 5),
        ((32, 32), 6),
        ((32, 32), 7),
        ((64, 64), 3),
        ((64, 64), 4),
        ((64, 64), 5),
        ((64, 64), 6),
        ((64, 64), 7),
        ((128, 128), 3),
        ((128, 128), 4),
        ((128, 128), 5),
        ((128, 128), 6),
        ((128, 128), 7),
    ]
)
def test_blind_planner_full_coverage_no_drift(grid_size, radius):
    # Setup 30x30 grid (smaller for faster TSP test)
    ROWS, COLS = grid_size
    grid = uav.environment.Grid(ROWS, COLS)

    start_pose = Pose(2, 2, 0.0)

    # 1. Setup Robot with BlindPlanner and TSP Policy
    # The BlindPlanner will invoke the policy once at start.
    policy = TSPRegionPolicy(radius=radius)
    planner = BlindPlanner(policy)
    robot = UnderwaterRobot(start_pose, planner, grid, sensor_radius=radius)

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

    # add grid information and radius to filename
    filename = f"data/full_coverage_tsp_no_drift_{ROWS}x{COLS}_r{radius}.png"
    uav.plotting.plot_results(grid, sim.history, sim.true_map_coverage,
                              "Blind Planner TSP Coverage No Drift", filename,
                              surface_indices=sim.surface_indices)

    # TSP Policy should theoretically achieve high coverage
    assert coverage_ratio == 1.0, f"Expected high coverage, got {coverage_ratio:.2%}"
