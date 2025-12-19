import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

import planner


class Grid:
    def __init__(self, rows, cols):
        # rows, cols include the walls.
        # Playable area is [1, rows-2] x [1, cols-2]
        self.rows = rows
        self.cols = cols

    def __str__(self):
        return f"Grid({self.rows}x{self.cols})"

    def is_valid(self, r, c):
        # Check within bounds
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        # Check walls (1-cell border)
        # 0 is wall, rows-1 is wall
        # 0 is wall, cols-1 is wall
        if r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1:
            return False
        return True

    def get_neighbors(self, r, c, radius=5):
        # Get neighbors within Euclidean distance
        neighbors = []
        # Optimization: scan bounding box
        r_min = max(0, r - radius)
        r_max = min(self.rows - 1, r + radius)
        c_min = max(0, c - radius)
        c_max = min(self.cols - 1, c + radius)

        for i in range(r_min, r_max + 1):
            for j in range(c_min, c_max + 1):
                if not self.is_valid(i, j):
                    continue
                # Euclidean distance check (center to center)
                dist = math.sqrt((i - r)**2 + (j - c)**2)
                if dist <= radius:
                    neighbors.append((i, j))
        return neighbors

class UAV:
    def __init__(self, start_pos, grid, drift_prob=0.2):
        self.pos = start_pos # (row, col)
        self.grid = grid
        self.drift_prob = drift_prob
        self.path_history = [start_pos]
        self.scanned_cells = set()

        # Initial scan
        self.scan()

    def scan(self):
        neighbors = self.grid.get_neighbors(*self.pos, radius=5)
        for h in neighbors:
            self.scanned_cells.add(h)

    def move_towards(self, target_pos):
        """
        Attempts to move to target_pos.
        Drift is restricted to the 3 cells in the forward direction:
        Forward (Target), Forward-Left, Forward-Right.
        """
        curr_r, curr_c = self.pos
        target_r, target_c = target_pos

        # Calculate movement vector
        dr = target_r - curr_r
        dc = target_c - curr_c

        # If staying put (dr=0, dc=0), just scan and return
        if dr == 0 and dc == 0:
            self.scan()
            return self.pos

        forward = (target_r, target_c)

        # Determine Perpendiculars for Drift
        left_dr, left_dc = -dc, dr
        right_dr, right_dc = dc, -dr

        candidates = []

        # 1. Forward (Target)
        if self.grid.is_valid(*forward):
            candidates.append(forward)

        # 2. Forward-Left
        f_left = (target_r + left_dr, target_c + left_dc)
        if self.grid.is_valid(*f_left):
            candidates.append(f_left)

        # 3. Forward-Right
        f_right = (target_r + right_dr, target_c + right_dc)
        if self.grid.is_valid(*f_right):
            candidates.append(f_right)

        actual_move = forward # Default

        # Only drift if we have alternatives
        alternatives = [c for c in candidates if c != forward]

        if alternatives and random.random() < self.drift_prob:
            actual_move = random.choice(alternatives)
        elif not self.grid.is_valid(*forward):
            # If strictly forward is blocked, force drift if valid
            if alternatives:
                actual_move = random.choice(alternatives)
            else:
                actual_move = self.pos # Staying put

        self.pos = actual_move
        self.path_history.append(self.pos)
        self.scan()
        return self.pos



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
    rx, ry = planner.planning(ox, oy, resolution)

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

def plot_results(grid, uav, title, filename):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a dense grid for visualization
    # Values: 0 = Wall (Black), 1 = Unexplored (Grey), 2 = Scanned (White)
    vis_grid = np.zeros((grid.rows, grid.cols), dtype=int)

    # Fill defaults
    # Set all valid cells to Unexplored (1)
    # Walls (0) will remain 0
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.is_valid(r, c):
                vis_grid[r, c] = 1
            else:
                vis_grid[r, c] = 0

    # Mark scanned
    for r, c in uav.scanned_cells:
        if 0 <= r < grid.rows and 0 <= c < grid.cols:
            vis_grid[r, c] = 2

    # Define Colormap
    # 0 -> Black (Wall)
    # 1 -> Grey (Unexplored)
    # 2 -> White (Scanned)
    cmap = ListedColormap(['black', 'grey', 'white'])

    # Plot heatmap
    ax.imshow(vis_grid, cmap=cmap, origin='upper', extent=[0, grid.cols, grid.rows, 0])

    # Plot Trajectory with Gradient
    py, px = zip(*uav.path_history)
    # Shift to center of cells
    px = np.array([x + 0.5 for x in px])
    py = np.array([y + 0.5 for y in py])

    # Create segments for LineCollection
    points = np.array([px, py]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create LineCollection with Gradient
    norm = plt.Normalize(0, len(px))
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(np.arange(len(px)))
    lc.set_linewidth(3) # 3x thicker

    ax.add_collection(lc)

    # Start/End Markers
    ax.plot(px[-1], py[-1], 'r*', markersize=15, label='End', zorder=10)
    ax.plot(px[0], py[0], 'go', markersize=10, label='Start', zorder=10)

    valid_cells_count = (grid.rows - 2) * (grid.cols - 2)
    scanned_count = len(uav.scanned_cells)
    coverage_pct = scanned_count / valid_cells_count * 100

    ax.set_title(f"{title}\nCoverage: {scanned_count}/{valid_cells_count} ({coverage_pct:.2f}%)")
    plt.legend(loc='upper right')

    plt.savefig(filename)
    plt.close()

    unscanned = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            if vis_grid[r, c] == 1: # Unexplored
                unscanned.append((r, c))

    if unscanned:
        print(f"[{title}] First 20 Missed Cells: {unscanned[:20]}")

def run_simulation_scenario(drift_prob, output_name, title):
    # Setup 100x100 playable -> 102x102 with walls
    ROWS, COLS = 102, 102
    grid = Grid(ROWS, COLS)

    # Plan
    ideal_path = generate_lawnmower_path(ROWS, COLS)

    start_pos = ideal_path[0] # Start at the beginning of the path
    uav = UAV(start_pos, grid, drift_prob=drift_prob)

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
        current_r, current_c = uav.pos
        # Open Loop Assumption: We think we are at the ideal previous location?
        # Or we just blindly apply delta to CURRENT position?
        # User prompt: "assuming it is always on track".
        # This implies we apply the PLAN'S relative move to the CURRENT position.

        target_r, target_c = current_r + dr, current_c + dc
        uav.move_towards((target_r, target_c))

    plot_results(grid, uav, title, output_name)

    valid_cells_count = (ROWS - 2) * (COLS - 2)
    scanned_count = len(uav.scanned_cells)
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
