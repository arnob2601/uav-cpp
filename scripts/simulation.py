import math
import random
import numpy as np

import uav


def generate_lawnmower_path(grid_rows, grid_cols, radius=5):
    """
    Generates a list of coordinates using the ported planner.
    Valid area is [1, rows-2] x [1, cols-2].
    Path is interpolated to ensure dense coverage.
    """
    # Define polygon for the valid area
    # Expand polygon by radius to ensure path hugs the edges (planner keeps centers inside)
    # We will clamp the actual path points later.
    margin = radius
    min_r, max_r = 1 - margin, grid_rows - 2 + margin
    min_c, max_c = 1 - margin, grid_cols - 2 + margin

    ox = [min_c, max_c, max_c, min_c, min_c]
    oy = [min_r, min_r, max_r, max_r, min_r]

    # Resolution:
    # Radius 5 => Diameter 10.
    # We need significant overlap to ensure coverage.
    resolution = 1.0 * radius

    # Plan
    rx, ry = uav.planner.planning(ox, oy, resolution)

    # Interpolate to create dense path
    dense_path = []

    if not rx:
        return []

    # Start point
    curr_r = int(round(ry[0]))
    curr_c = int(round(rx[0]))

    # Clamp
    curr_r = max(0, min(curr_r, grid_rows - 1))
    curr_c = max(0, min(curr_c, grid_cols - 1))

    dense_path.append((curr_r, curr_c))

    for i in range(1, len(rx)):
        target_r = int(round(ry[i]))
        target_c = int(round(rx[i]))

        target_r = max(0, min(target_r, grid_rows - 1))
        target_c = max(0, min(target_c, grid_cols - 1))

        # Walk from curr to target
        # Simple walk: move along one axis then other?
        # Or Bresenham?
        # Planner usually moves Manhattan (along X or along Y), but diagonal turns possible.
        # Let's simple step towards.

        while (curr_r, curr_c) != (target_r, target_c):
            dr = target_r - curr_r
            dc = target_c - curr_c

            # Step size 1
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_c = 0 if dc == 0 else (1 if dc > 0 else -1)

            # If diagonal move needed, do it? or one by one?
            # Planner `SweepSearcher` moves 8-neighbor?
            # `find_safe_turning_grid` allows diagonal.
            # Simulation `UAV.move_towards` handles drift on steps.
            # Ideally we feed 1-step increments.

            curr_r += step_r
            curr_c += step_c
            dense_path.append((curr_r, curr_c))

    return dense_path


def run_simulation_scenario(drift_prob, output_name, title):
    # Setup 100x100 playable -> 102x102 with walls
    ROWS, COLS = 102, 102
    grid = uav.environment.Grid(ROWS, COLS)

    # Plan
    ideal_path = generate_lawnmower_path(ROWS, COLS)

    start_pos = ideal_path[0] # Start at the beginning of the path
    robot = uav.robot.UAV(start_pos, grid, drift_prob=drift_prob)

    # Convert to moves
    moves = []
    # Initial move from current (1,1) to first point of path (1,1) -> No move
    # So we iterate from the SECOND point
    # Wait, if start_pos IS ideal_path[0], we start moving to ideal_path[1]

    # Actually, generate_lawnmower_path generates all points.
    # If we are already at [0], next target is [1].

    for i in range(1, len(ideal_path)):
        prev = ideal_path[i-1]
        curr = ideal_path[i]
        dr, dc = curr[0] - prev[0], curr[1] - prev[1]
        moves.append((dr, dc))

    # Execute
    print(f"[{title}] Simulating {len(moves)} steps...")

    for dr, dc in moves:
        current_r, current_c = robot.pos
        # Open Loop Assumption: We think we are at the ideal previous location?
        # Or we just blindly apply delta to CURRENT position?
        # User prompt: "assuming it is always on track".
        # This implies we apply the PLAN'S relative move to the CURRENT position.

        target_r, target_c = current_r + dr, current_c + dc
        robot.move_towards((target_r, target_c))

    uav.plotting.plot_results(grid, robot, title, output_name)

    valid_cells_count = (ROWS - 2) * (COLS - 2)
    scanned_count = len(robot.scanned_cells)
    print(f"[{title}] Coverage: {scanned_count}/{valid_cells_count} ({scanned_count/valid_cells_count:.2%}%)")
    return scanned_count == valid_cells_count

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
