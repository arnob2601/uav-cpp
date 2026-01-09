import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
matplotlib.use('Agg')


def plot_results(grid, path_history, coverage_grid, title, filename, surface_indices=None):
    """
    path_history: list of (x, y) tuples or Pose objects
    coverage_grid: 2D numpy array where >0 is covered
    surface_indices: list of indices in path_history where surfacing occurred
    """
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

    # Mark scanned from coverage_grid
    rows, cols = coverage_grid.shape
    for r in range(rows):
        for c in range(cols):
            if coverage_grid[r, c] > 0 and 0 <= r < grid.rows and 0 <= c < grid.cols:
                vis_grid[r, c] = 2

    # Define Colormap
    # 0 -> Black (Wall)
    # 1 -> Grey (Unexplored)
    # 2 -> White (Scanned)
    cmap = ListedColormap(['black', 'grey', 'white'])

    # Plot heatmap
    ax.imshow(vis_grid, cmap=cmap, origin='upper', extent=[0, grid.cols, grid.rows, 0])

    # Plot Trajectory with Gradient
    # path_history can be tuples or Pose objects
    px = []
    py = []
    for p in path_history:
        if hasattr(p, 'x'):
            px.append(p.x + 0.5)
            py.append(p.y + 0.5)
        else:
            px.append(p[0] + 0.5)
            py.append(p[1] + 0.5)

    px = np.array(px)
    py = np.array(py)

    # Create segments for LineCollection
    points = np.array([px, py]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create LineCollection with Gradient
    norm = plt.Normalize(0, len(px))
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(np.arange(len(px)))
    lc.set_linewidth(3)  # 3x thicker

    ax.add_collection(lc)

    # Start/End Markers
    ax.plot(px[-1], py[-1], 'r*', markersize=15, label='End', zorder=10)
    ax.plot(px[0], py[0], 'go', markersize=10, label='Start', zorder=10)

    # Surface Markers
    if surface_indices:
        traj_cmap = plt.get_cmap('viridis')
        for idx in surface_indices:
            if 0 <= idx < len(px):
                # Retrieve color from trajectory colormap based on index
                color = traj_cmap(norm(idx))
                # Plot marker ("^" for surface) with black edge for visibility
                ax.plot(px[idx], py[idx], marker='^', color=color, markeredgecolor='black', markersize=12, zorder=11,
                        label='Surface' if idx == surface_indices[0] else "")

    valid_cells_count = (grid.rows - 2) * (grid.cols - 2)
    scanned_count = np.sum(coverage_grid > 0)
    coverage_pct = scanned_count / valid_cells_count * 100

    # Calculate Path Length (Cost)
    path_len = 0.0
    if len(px) > 1:
        dists = np.sqrt(np.diff(px)**2 + np.diff(py)**2)
        path_len = np.sum(dists)

    ax.set_title(f"{title}\nCoverage: {scanned_count}/{valid_cells_count} ({coverage_pct:.2f}%) | Cost: {path_len:.2f}")

    # Handle duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.savefig(filename)
    plt.close()

    unscanned = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            if vis_grid[r, c] == 1:  # Unexplored
                unscanned.append((r, c))

    if unscanned:
        print(f"[{title}] First 20 Missed Cells: {unscanned[:20]}")
