"""
Test harness for the Ball Drop Simulator.

This module provides unit tests using `pytest` to validate the behavior of the
Ball Drop Simulator. It includes tests for various scenarios such as different
initial conditions for balls, varying environments (with or without air resistance),
and corner cases like zero initial height or an empty ball list.
"""
from dataclasses import dataclass
import sys
import math
import argparse
import readchar
import vpython as vp
import pytest

sys.path.append('.')
from ball_drop_sim import BallDropSimulator
from ball import Ball
from environment import Environment, EnvironmentDefaults
from ball_spec import BallSpec, BallSpecDefaults

__author__ = 'Jim Tooker'


@dataclass
class ExpectedResults:
    """
    Data class to hold the expected results of a Ball Drop Simulator test run.

    Attributes:
        total_time (float): Total simulation time until all balls stop.
        init_height (list[float]): List of initial heights for each ball.
        max_speed (list[float]): List of maximum speeds reached by each ball.
        terminal_vel_reached (list[bool]): List indicating if each ball reached terminal velocity.
        terminal_velocity (list[float]): List of terminal velocities for each ball.
        first_impact_time (list[float]): List of times for each ball's first ground impact.
        stop_time (list[float]): List of times when each ball stopped moving.
    """
    total_time: float
    init_height: list[float]
    max_speed: list[float]
    terminal_vel_reached: list[bool]
    terminal_velocity: list[float]
    first_impact_time: list[float]
    stop_time: list[float]


class TestBallDropSimulator:
    """
    Class for `pytest` testing of the Ball Drop Simulator.
    
    This class contains test methods that simulate various conditions, including 
    different ball masses, drag coefficients, and environments. It verifies that the 
    BallDropSimulator produces results that match the expected values for total simulation 
    time, ball speeds, and other metrics.
    """
    tolerance = 0.01  # Tolerance for float comparisons

    def check_results(self, sim: BallDropSimulator, expected: ExpectedResults) -> None:
        """
        Checks the results of a Ball Drop Simulator run against expected values.

        Args:
            sim (BallDropSimulator): The simulation instance being checked.
            expected (ExpectedResults): The expected results to compare against.

        Raises:
            AssertionError: If any of the actual results deviate from the expected values.
        """
        assert sim.total_time == pytest.approx(expected.total_time, abs=self.tolerance)
        for i, ball in enumerate(sim.balls):
            assert ball.init_height == pytest.approx(expected.init_height[i], abs=self.tolerance)
            assert ball.max_speed == pytest.approx(expected.max_speed[i], abs=self.tolerance)
            assert ball.terminal_vel_reached == expected.terminal_vel_reached[i]
            assert ball.terminal_velocity == pytest.approx(expected.terminal_velocity[i], abs=self.tolerance)
            assert ball.first_impact_time == pytest.approx(expected.first_impact_time[i], abs=self.tolerance)
            assert ball.stop_time == pytest.approx(expected.stop_time[i], abs=self.tolerance)

    def test_2_ball(self) -> None:
        """
        Test case with two balls having different masses, radii, and environments.
        """
        # Create the Ball Specs
        ball1_spec: BallSpec = BallSpec(
            mass=300,
            radius=5,
            drag_coefficient=BallSpecDefaults.SPHERE_DRAG_COEFFICIENT
        )
        ball2_spec: BallSpec = BallSpec(
            mass=100,
            radius=3,
            drag_coefficient=BallSpecDefaults.SPHERE_DRAG_COEFFICIENT/2
        )

        # Create two different environments
        env1: Environment = Environment(
            gravity=EnvironmentDefaults.EARTH_GRAVITY,
            air_density=EnvironmentDefaults.EARTH_AIR_DENSITY,
            cor=0.9
        )
        env2: Environment = Environment(
            gravity=EnvironmentDefaults.EARTH_GRAVITY,
            air_density=0.3,
            cor=EnvironmentDefaults.DEFAULT_COR
        )

        # Create two balls with different environments
        ball1: Ball = Ball(
            specs=ball1_spec,
            env=env1,
            init_height=50,
            color=vp.color.blue
        )
        ball2: Ball = Ball(
            specs=ball2_spec,
            env=env2,
            init_height=30,
            color=vp.color.red
        )

        # Expected Results
        expected_results = ExpectedResults(total_time=15.74,
                                           init_height=[50,30],
                                           max_speed=[11.4,21.07],
                                           terminal_vel_reached=[True,False],
                                           terminal_velocity=[11.41,31.37],
                                           first_impact_time=[5.19,2.6],
                                           stop_time=[14.78,15.74])

        # Run simulation
        sim = BallDropSimulator([ball1, ball2])
        sim.run()
        self.check_results(sim, expected_results)

    def test_default_ball(self) -> None:
        """
        Test case with a single default ball using default environment and ball specs.
        """
        # Create default ball
        ball1: Ball = Ball()

        # Expected Results
        expected_results = ExpectedResults(total_time=4.68,
                                           init_height=[10],
                                           max_speed=[3.29],
                                           terminal_vel_reached=[True],
                                           terminal_velocity=[3.29],
                                           first_impact_time=[3.27],
                                           stop_time=[4.68])

        # Run simulation
        sim = BallDropSimulator([ball1])
        sim.run()
        self.check_results(sim, expected_results)

    def test_3_ball(self) -> None:
        """
        Test case with three balls, mixing default and custom configurations.
        """
        # Create the Ball Specs
        ball1_spec: BallSpec = BallSpec(
            mass=300,
            radius=5,
            drag_coefficient=BallSpecDefaults.SPHERE_DRAG_COEFFICIENT
        )
        ball2_spec: BallSpec = BallSpec(
            mass=100,
            radius=3,
            drag_coefficient=BallSpecDefaults.SPHERE_DRAG_COEFFICIENT/2
        )

        # Create two different environments
        env1: Environment = Environment(
            gravity=EnvironmentDefaults.EARTH_GRAVITY,
            air_density=EnvironmentDefaults.EARTH_AIR_DENSITY,
            cor=0.9
        )
        env2: Environment = Environment(
            gravity=EnvironmentDefaults.EARTH_GRAVITY,
            air_density=0.3,
            cor=EnvironmentDefaults.DEFAULT_COR
        )

        # Create three balls with different environments
        ball1: Ball = Ball(
            specs=ball1_spec,
            env=env1,
            init_height=50,
            color=vp.color.blue
        )
        ball2: Ball = Ball(
            specs=ball2_spec,
            env=env2,
            init_height=30,
            color=vp.color.green
        )
        ball3: Ball = Ball()

        # Expected Results
        expected_results = ExpectedResults(total_time=15.74,
                                           init_height=[50,30,10],
                                           max_speed=[11.4,21.07,3.29],
                                           terminal_vel_reached=[True,False,True],
                                           terminal_velocity=[11.41,31.37,3.29],
                                           first_impact_time=[5.19,2.6,3.27],
                                           stop_time=[14.78,15.74,4.68])

        # Run simulation
        sim = BallDropSimulator([ball1, ball2, ball3])
        sim.run()
        self.check_results(sim, expected_results)

    def test_no_air(self) -> None:
        """
        Test case with a single ball in a vacuum (no air resistance).
        """
        # Create ball specification
        ball1_spec: BallSpec = BallSpec()

        # Create Environment
        env1: Environment = Environment(
            gravity=EnvironmentDefaults.EARTH_GRAVITY,
            air_density=0,
            cor=0.8
        )

        # Create Ball
        ball1: Ball = Ball(
            specs=ball1_spec,
            env=env1
        )

        # Expected Results
        expected_results = ExpectedResults(total_time=12.11,
                                           init_height=[10],
                                           max_speed=[14.02],
                                           terminal_vel_reached=[False],
                                           terminal_velocity=[math.inf],
                                           first_impact_time=[1.43],
                                           stop_time=[12.11])

        # Run simulation
        sim = BallDropSimulator([ball1])
        sim.run()
        self.check_results(sim, expected_results)

    def test_empty_ball_list(self) -> None:
        """
        Test case where no balls are provided to the simulator.
        """
        with pytest.raises(ValueError):
            BallDropSimulator([])

    def test_zero_init_height(self) -> None:
        """
        Test case where a ball is initialized with zero height.
        """
        with pytest.raises(ValueError):
            ball = Ball(init_height=0)
            BallDropSimulator([ball])


def main() -> None:
    """
    Main entry point for the test harness.

    This function handles command-line arguments and executes the tests 
    with or without the GUI depending on the options provided.
    """
    parser = argparse.ArgumentParser(description='Ball Drop Simulator Tester')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI')
    parser.add_argument('-k', help='Only run tests which match the given substring expression')
    args = parser.parse_args()

    if args.no_gui:
        BallDropSimulator.disable_gui(True)

    pytest_args = ["tests/test_ball_drop_sim.py"]
    if args.k:
        pytest_args.extend(["-k", args.k])

    result = pytest.main(pytest_args)

    if args.no_gui:
        sys.exit(result)
    else:
        print("Press any key to exit...")
        readchar.readkey()
        BallDropSimulator.quit_simulation()

if __name__ == '__main__':
    main()
