import inspect
from uav.environment import Grid
from uav.policies import TSPRegionPolicy
from uav.simulator import CoverageSimulator


def test_coverage_radius_consistency():
    """
    Ensures that the Robot/Simulator and Planner use the same coverage radius.
    """
    # 1. Get Simulator/Environment default radius
    # Simulator calls env.get_neighbors(r, c) without radius arg, relying on default.
    grid_sig = inspect.signature(Grid.get_neighbors)
    sim_radius = grid_sig.parameters['radius'].default

    # 2. Get Planner Policy default radius
    # TSPRegionPolicy passes radius to planner
    policy_sig = inspect.signature(TSPRegionPolicy.__init__)
    policy_default_radius = policy_sig.parameters['radius'].default

    print(f"Simulator (Grid) Radius: {sim_radius}")
    print(f"Policy Default Radius: {policy_default_radius}")

    assert sim_radius == policy_default_radius, \
        f"Radius mismatch! Grid uses {sim_radius}, Policy uses {policy_default_radius}. They must match."


def test_planning_resolution_matches_radius():
    """
    Verify that TSPRegionPolicy passes its radius as resolution to planner.
    """
    policy = TSPRegionPolicy(radius=5)
    # We can mock planner.plan_coverage_tsp to check arguments

    import uav.planner
    from unittest.mock import MagicMock

    orig_planner = uav.planner.plan_coverage_tsp
    uav.planner.plan_coverage_tsp = MagicMock(return_value=[])

    try:
        policy.plan(None, None)

        # Check calls
        uav.planner.plan_coverage_tsp.assert_called_once()
        call_args = uav.planner.plan_coverage_tsp.call_args
        kwargs = call_args.kwargs

        passed_radius = kwargs.get('radius')
        assert passed_radius == 5, f"Policy did not pass configured radius! Passed: {passed_radius}"

    finally:
        uav.planner.plan_coverage_tsp = orig_planner
