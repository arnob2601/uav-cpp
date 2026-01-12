import numpy as np

from uav import planner
from uav.datatypes import Pose


class TestTSPPlanner:
    def test_clustering_single_block(self):
        """Test clustering on a simple 50x50 map with one 10x10 block unvisited"""
        rows, cols = 50, 50
        belief_map = np.ones((rows, cols))
        # Create unvisited block in middle
        belief_map[20:30, 20:30] = 0

        start_pose = Pose(0, 0, 0)

        path = planner.plan_coverage_tsp(belief_map, start_pose, radius=1)

        assert len(path) > 0, "Should generate path for unvisited block"

        # Verify points are within bounds
        for x, y in path:
            assert 0 <= x < cols
            assert 0 <= y < rows

    def test_clustering_separated_blocks(self):
        """Test clustering with two separated blocks"""
        rows, cols = 50, 50
        belief_map = np.ones((rows, cols))

        # Block 1 (Top Left approx)
        belief_map[5:15, 5:15] = 0

        # Block 2 (Bottom Right approx)
        belief_map[35:45, 35:45] = 0

        start_pose = Pose(0, 0, 0)

        path = planner.plan_coverage_tsp(belief_map, start_pose, radius=1)

        assert len(path) > 0

        # We expect the path to jump between regions.
        # Check if we visit both regions?
        # A simple check is that the path covers cells in both regions.

        visited_b1 = False
        visited_b2 = False

        for x, y in path:
            r, c = int(y), int(x)
            if 5 <= r < 15 and 5 <= c < 15:
                visited_b1 = True
            if 35 <= r < 45 and 35 <= c < 45:
                visited_b2 = True

        assert visited_b1, "Should visit Block 1"
        assert visited_b2, "Should visit Block 2"

    def test_wall_handling(self):
        """Test that unvisited walls are ignored"""
        rows, cols = 50, 50
        belief_map = np.ones((rows, cols))

        # Set walls as unvisited (0)
        belief_map[0, :] = 0
        belief_map[rows - 1, :] = 0
        belief_map[:, 0] = 0
        belief_map[:, cols - 1] = 0

        start_pose = Pose(5, 5, 0)

        path = planner.plan_coverage_tsp(belief_map, start_pose, radius=1)

        # Should be empty as only walls are 0
        assert len(path) == 0, "Should ignore walls"
