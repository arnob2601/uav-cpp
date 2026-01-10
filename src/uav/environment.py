import math


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

    def get_neighbors(self, r, c, radius=3):
        # Get neighbors within Euclidean distance
        neighbors = []
        # Optimization: scan bounding box
        r_min = max(0, int(r - radius))
        r_max = min(self.rows - 1, int(r + radius))
        c_min = max(0, int(c - radius))
        c_max = min(self.cols - 1, int(c + radius))

        for i in range(r_min, r_max + 1):
            for j in range(c_min, c_max + 1):
                if not self.is_valid(i, j):
                    continue
                # Euclidean distance check (center to center)
                dist = math.sqrt((i - r)**2 + (j - c)**2)
                if dist <= radius:
                    neighbors.append((i, j))
        return neighbors
