"""
Grid based sweep planner
Ported from PythonRobotics for UAV Simulation
"""

import math
from enum import IntEnum
import numpy as np

# --- Utils ---
def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s], [s, c]])

class FloatGrid:
    def __init__(self, init_val=0.0):
        self.data = init_val

    def get_float_data(self):
        return self.data

    def __eq__(self, other):
        if not isinstance(other, FloatGrid):
            return NotImplemented
        return self.get_float_data() == other.get_float_data()

    def __lt__(self, other):
        if not isinstance(other, FloatGrid):
            return NotImplemented
        return self.get_float_data() < other.get_float_data()
    
    def __ge__(self, other):
        if not isinstance(other, FloatGrid):
            return NotImplemented
        return self.get_float_data() >= other.get_float_data()

# --- Grid Map ---
class GridMap:
    def __init__(self, width, height, resolution, center_x, center_y, init_val=FloatGrid(0.0)):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.center_x = center_x
        self.center_y = center_y

        self.left_lower_x = self.center_x - self.width / 2.0 * self.resolution
        self.left_lower_y = self.center_y - self.height / 2.0 * self.resolution

        self.n_data = self.width * self.height
        self.data = [init_val] * self.n_data
        self.data_type = type(init_val)

    def get_value_from_xy_index(self, x_ind, y_ind):
        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)
        if 0 <= grid_ind < self.n_data:
            return self.data[grid_ind]
        else:
            return None

    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        return int(y_ind * self.width + x_ind)

    def calc_xy_index_from_position(self, pos, lower_pos, max_index):
        ind = int(np.floor((pos - lower_pos) / self.resolution))
        if 0 <= ind <= max_index:
            return ind
        else:
            return None

    def calc_grid_central_xy_position_from_xy_index(self, x_ind, y_ind):
        x_pos = self.left_lower_x + x_ind * self.resolution + self.resolution / 2.0
        y_pos = self.left_lower_y + y_ind * self.resolution + self.resolution / 2.0
        return x_pos, y_pos

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        if (x_ind is None) or (y_ind is None):
            return False
        grid_ind = int(y_ind * self.width + x_ind)
        if 0 <= grid_ind < self.n_data and isinstance(val, self.data_type):
            self.data[grid_ind] = val
            return True
        return False

    def check_occupied_from_xy_index(self, x_ind, y_ind, occupied_val):
        val = self.get_value_from_xy_index(x_ind, y_ind)
        if val is None or val >= occupied_val:
            return True
        return False

    def set_value_from_polygon(self, pol_x, pol_y, val, inside=True):
        if (pol_x[0] != pol_x[-1]) or (pol_y[0] != pol_y[-1]):
            pol_x.append(pol_x[0])
            pol_y.append(pol_y[0])

        for x_ind in range(self.width):
            for y_ind in range(self.height):
                x_pos, y_pos = self.calc_grid_central_xy_position_from_xy_index(x_ind, y_ind)
                flag = self.check_inside_polygon(x_pos, y_pos, pol_x, pol_y)
                if flag is inside:
                    self.set_value_from_xy_index(x_ind, y_ind, val)

    @staticmethod
    def check_inside_polygon(iox, ioy, x, y):
        n_point = len(x) - 1
        inside = False
        for i1 in range(n_point):
            i2 = (i1 + 1) % (n_point + 1)
            if x[i1] >= x[i2]:
                min_x, max_x = x[i2], x[i1]
            else:
                min_x, max_x = x[i1], x[i2]
            if not min_x <= iox < max_x:
                continue
            tmp1 = (y[i2] - y[i1]) / (x[i2] - x[i1])
            if (y[i1] + tmp1 * (iox - x[i1]) - ioy) > 0.0:
                inside = not inside
        return inside
    
    def expand_grid(self, occupied_val=FloatGrid(1.0)):
        x_inds, y_inds, values = [], [], []

        for ix in range(self.width):
            for iy in range(self.height):
                if self.check_occupied_from_xy_index(ix, iy, occupied_val):
                    x_inds.append(ix)
                    y_inds.append(iy)
                    values.append(self.get_value_from_xy_index(ix, iy))

        for (ix, iy, value) in zip(x_inds, y_inds, values):
            self.set_value_from_xy_index(ix + 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy + 1, val=value)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy - 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=value)


# --- Sweep Planner ---
class SweepSearcher:
    class SweepDirection(IntEnum):
        UP = 1
        DOWN = -1

    class MovingDirection(IntEnum):
        RIGHT = 1
        LEFT = -1

    def __init__(self, moving_direction, sweep_direction, x_inds_goal_y, goal_y):
        self.moving_direction = moving_direction
        self.sweep_direction = sweep_direction
        self.turing_window = []
        self.update_turning_window()
        self.x_indexes_goal_y = x_inds_goal_y
        self.goal_y = goal_y

    def update_turning_window(self):
        self.turing_window = [
            (self.moving_direction, 0.0),
            (self.moving_direction, self.sweep_direction),
            (0, self.sweep_direction),
            (-self.moving_direction, self.sweep_direction),
        ]

    def swap_moving_direction(self):
        self.moving_direction *= -1
        self.update_turning_window()

    def check_occupied(self, c_x_index, c_y_index, grid_map, occupied_val=FloatGrid(0.5)):
        return grid_map.check_occupied_from_xy_index(c_x_index, c_y_index, occupied_val)

    def find_safe_turning_grid(self, c_x_index, c_y_index, grid_map):
        for (d_x_ind, d_y_ind) in self.turing_window:
            next_x_ind = int(d_x_ind + c_x_index)
            next_y_ind = int(d_y_ind + c_y_index)
            if not self.check_occupied(next_x_ind, next_y_ind, grid_map):
                return next_x_ind, next_y_ind
        return None, None

    def move_target_grid(self, c_x_index, c_y_index, grid_map):
        n_x_index = self.moving_direction + c_x_index
        n_y_index = c_y_index

        if not self.check_occupied(n_x_index, n_y_index, grid_map):
            return n_x_index, n_y_index
        else:
            next_c_x_index, next_c_y_index = self.find_safe_turning_grid(c_x_index, c_y_index, grid_map)
            
            if (next_c_x_index is None) and (next_c_y_index is None):
                # moving backward
                next_c_x_index = -self.moving_direction + c_x_index
                next_c_y_index = c_y_index
                if self.check_occupied(next_c_x_index, next_c_y_index, grid_map, FloatGrid(1.0)):
                    return None, None
            else:
                while not self.check_occupied(next_c_x_index + self.moving_direction, next_c_y_index, grid_map):
                    next_c_x_index += self.moving_direction
                self.swap_moving_direction()
            return next_c_x_index, next_c_y_index

    def is_search_done(self, grid_map):
        for ix in self.x_indexes_goal_y:
            if not self.check_occupied(ix, self.goal_y, grid_map):
                return False
        return True

    def search_start_grid(self, grid_map):
        x_inds = []
        y_ind = 0
        if self.sweep_direction == self.SweepDirection.DOWN:
            x_inds, y_ind = search_free_grid_index_at_edge_y(grid_map, from_upper=True)
        elif self.sweep_direction == self.SweepDirection.UP:
            x_inds, y_ind = search_free_grid_index_at_edge_y(grid_map, from_upper=False)

        if self.moving_direction == self.MovingDirection.RIGHT:
            return min(x_inds), y_ind
        elif self.moving_direction == self.MovingDirection.LEFT:
            return max(x_inds), y_ind
        raise ValueError("self.moving direction is invalid ")

def search_free_grid_index_at_edge_y(grid_map, from_upper=False):
    y_index = None
    x_indexes = []

    if from_upper:
        x_range = range(grid_map.height)[::-1]
        y_range = range(grid_map.width)[::-1]
    else:
        x_range = range(grid_map.height)
        y_range = range(grid_map.width)

    for iy in x_range:
        for ix in y_range:
            if not grid_map.check_occupied_from_xy_index(ix, iy, FloatGrid(0.5)):
                y_index = iy
                x_indexes.append(ix)
        if y_index is not None:
            break
    return x_indexes, y_index

def find_sweep_direction_and_start_position(ox, oy):
    max_dist = 0.0
    vec = [0.0, 0.0]
    sweep_start_pos = [0.0, 0.0]
    for i in range(len(ox) - 1):
        dx = ox[i + 1] - ox[i]
        dy = oy[i + 1] - oy[i]
        d = np.hypot(dx, dy)
        if d > max_dist:
            max_dist = d
            vec = [dx, dy]
            sweep_start_pos = [ox[i], oy[i]]
    return vec, sweep_start_pos

def convert_grid_coordinate(ox, oy, sweep_vec, sweep_start_position):
    tx = [ix - sweep_start_position[0] for ix in ox]
    ty = [iy - sweep_start_position[1] for iy in oy]
    th = math.atan2(sweep_vec[1], sweep_vec[0])
    converted_xy = np.stack([tx, ty]).T @ rot_mat_2d(th)
    return converted_xy[:, 0], converted_xy[:, 1]

def convert_global_coordinate(x, y, sweep_vec, sweep_start_position):
    th = math.atan2(sweep_vec[1], sweep_vec[0])
    converted_xy = np.stack([x, y]).T @ rot_mat_2d(-th)
    rx = [ix + sweep_start_position[0] for ix in converted_xy[:, 0]]
    ry = [iy + sweep_start_position[1] for iy in converted_xy[:, 1]]
    return rx, ry

def setup_grid_map(ox, oy, resolution, sweep_direction, offset_grid=10):
    width = math.ceil((max(ox) - min(ox)) / resolution) + offset_grid
    height = math.ceil((max(oy) - min(oy)) / resolution) + offset_grid
    center_x = (np.max(ox) + np.min(ox)) / 2.0
    center_y = (np.max(oy) + np.min(oy)) / 2.0

    grid_map = GridMap(width, height, resolution, center_x, center_y)
    grid_map.set_value_from_polygon(ox, oy, FloatGrid(1.0), inside=False)
    grid_map.expand_grid()

    x_inds_goal_y = []
    goal_y = 0
    if sweep_direction == SweepSearcher.SweepDirection.UP:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(grid_map, from_upper=True)
    elif sweep_direction == SweepSearcher.SweepDirection.DOWN:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(grid_map, from_upper=False)

    return grid_map, x_inds_goal_y, goal_y

def sweep_path_search(sweep_searcher, grid_map):
    c_x_index, c_y_index = sweep_searcher.search_start_grid(grid_map)
    if not grid_map.set_value_from_xy_index(c_x_index, c_y_index, FloatGrid(0.5)):
        print("Cannot find start grid")
        return [], []

    x, y = grid_map.calc_grid_central_xy_position_from_xy_index(c_x_index, c_y_index)
    px, py = [x], [y]

    while True:
        c_x_index, c_y_index = sweep_searcher.move_target_grid(c_x_index, c_y_index, grid_map)

        if sweep_searcher.is_search_done(grid_map) or (c_x_index is None or c_y_index is None):
            break

        x, y = grid_map.calc_grid_central_xy_position_from_xy_index(c_x_index, c_y_index)
        px.append(x)
        py.append(y)
        grid_map.set_value_from_xy_index(c_x_index, c_y_index, FloatGrid(0.5))

    return px, py

def planning(ox, oy, resolution,
             moving_direction=SweepSearcher.MovingDirection.RIGHT,
             sweeping_direction=SweepSearcher.SweepDirection.UP):
    sweep_vec, sweep_start_position = find_sweep_direction_and_start_position(ox, oy)
    rox, roy = convert_grid_coordinate(ox, oy, sweep_vec, sweep_start_position)
    grid_map, x_inds_goal_y, goal_y = setup_grid_map(rox, roy, resolution, sweeping_direction)
    sweep_searcher = SweepSearcher(moving_direction, sweeping_direction, x_inds_goal_y, goal_y)
    px, py = sweep_path_search(sweep_searcher, grid_map)
    rx, ry = convert_global_coordinate(px, py, sweep_vec, sweep_start_position)
    return list(rx), list(ry)
