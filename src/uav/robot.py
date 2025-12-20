import random


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
        neighbors = self.grid.get_neighbors(*self.pos, radius=3)
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