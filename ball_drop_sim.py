"""
FIXME - Need Module docstring
"""
from re import M
from typing import Final, Tuple, Optional
import argparse
import readchar
import vpython as vp
from ball_specs import BallSpecs, BallSpecsDefaults
from environment import Environment, EnvironmentDefaults
from ball import Ball


class BallDropSimulator:
    """
    FIXME
    """
    # Class constants
    _LABEL_RANGE: Final[int] = 10
    _LABEL_STEP: Final[float] = 0.7
    _LABEL_Y_OVERHEAD: Final[int] = 3
    _GRAPH_WIDTH: Final[int] = 600
    _GRAPH_HEIGHT: Final[int] = 400
    _MAIN_CANVAS_SIZE: Final[tuple[int, int]] = (600, 600)

    # Flag to indicate whether the GUI should be disabled (True = no GUI)
    _no_gui: bool = False

    def __init__(self, balls: list[Ball]) -> None:
        """
        Initialize simulation with list of balls.

        Args:
            list[Ball]: List of Ball objects to simulate dropping
        """
        self._validate_balls(balls)
        self._balls: list[Ball] = balls

        # Variables to keep track of time
        self._total_time: float = 0
        self._time_label: Optional[vp.label] = None

        # Initialize graph variables (Will be None for now)
        self._velocity_graph: Optional[vp.graph] = None
        self._acceleration_graph: Optional[vp.graph] = None
        self._position_graph: Optional[vp.graph] = None

        if BallDropSimulator._no_gui is False:
            # Initialize canvases
            self._canvas: vp.canvas = self._create_main_canvas()
            self._runtime_canvas: vp.canvas = self._create_runtime_canvas()
            self._parameter_canvas: vp.canvas = self._create_parameter_canvas()

            # Initialize all label containers
            self._height_labels: list[vp.label] = []
            self._speed_labels: list[vp.label] = []
            self._max_speed_labels: list[vp.label] = []
            self._terminal_velocity_labels: list[vp.label] = []
            self._first_impact_labels: list[vp.label] = []
            self._stop_time_labels: list[vp.label] = []

            # Initialize plot containers
            self._velocity_plots: list[vp.gcurve] = []
            self._acceleration_plots: list[vp.gcurve] = []
            self._position_plots: list[vp.gcurve] = []

            # Setup simulation
            self._setup_simulation()
        else:
            for i, ball in enumerate(balls):
                print(f'\nBall{i+1}:')
                print(f'  {ball.specs}')
                print(f'  {ball.env}')
                print(f'  Initial Height: {ball.init_height:.2f} m')

    @staticmethod
    def quit_simulation() -> None:
        """Stop the VPython server."""
        if BallDropSimulator._no_gui is False:
            # We don't import vp_services until needed, because importing it will start
            # the server, if not started already.
            import vpython.no_notebook as vp_services  # type: ignore[import-untyped]
            vp_services.stop_server()

    @classmethod
    def disable_gui(cls, no_gui: bool) -> None:
        """
        Enables or disables the GUI.

        Args:
            no_gui (bool): Flag to indicate where GUI should be disabled (True = disable GUI).
        """
        cls._no_gui = no_gui

    @property
    def balls(self) -> list[Ball]:
        """return balls object"""
        return self._balls

    @property
    def total_time(self) -> float:
        """return total time"""
        return self._total_time

    @property
    def _max_height(self) -> float:
        """Calculate maximum height among all balls."""
        return max(ball.position.y for ball in self._balls)

    @property
    def _grid_range(self) -> int:
        """Calculate grid range based on maximum height."""
        return int(self._max_height)

    def run(self) -> None:
        """Run the simulation."""
        FPS: Final[int] = 100
        dt: float = 1/FPS
        t: float = 0

        self._update_labels(t)

        while True:
            vp.rate(FPS)
            t += dt

            for ball in self._balls:
                ball.update(dt, t)

            self._update_labels(t)

            if all(ball.has_stopped for ball in self._balls):
                msg: str = f'Total Time: {t:.2f} s'
                if self._time_label:
                    self._time_label.text = msg
                else:
                    print(f'\n{msg}')
                break

        self._total_time = t

        if BallDropSimulator._no_gui is True:
            for i, ball in enumerate(self._balls):
                print(f'\nBall{i+1}:')
                print(f'  Max speed: {ball.max_speed:.2f} m/s')
                print(f'  Terminal velocity reached?: {ball.terminal_vel_reached}. ({ball.terminal_velocity:.2f} m/s)')
                print(f'  Time for 1st impact: {ball.first_impact_time:.2f} s')
                print(f'  Time to stop: {ball.stop_time:.2f} s')

    def _validate_balls(self, balls: list[Ball]) -> None:
        """Validate the balls parameter."""
        if not isinstance(balls, list):
            raise ValueError("'balls' parameter must be a list")
        if not all(isinstance(ball, Ball) for ball in balls):
            raise ValueError("All elements in 'balls' must be instances of Ball")
        if not balls:
            raise ValueError("'balls' list cannot be empty")

    def _create_main_canvas(self) -> vp.canvas:
        """Create and return the main simulation canvas."""
        return vp.canvas(
            title='Ball Drop BallDropSimulator',
            width=self._MAIN_CANVAS_SIZE[0],
            height=self._MAIN_CANVAS_SIZE[1],
            background=vp.color.white,
            align='left'
        )

    def _create_runtime_canvas(self) -> vp.canvas:
        """Create and return the runtime information canvas."""
        return vp.canvas(
            width=self._MAIN_CANVAS_SIZE[0],
            height=self._MAIN_CANVAS_SIZE[1],
            background=vp.color.white,
            align='left'
        )

    def _create_parameter_canvas(self) -> vp.canvas:
        """Create and return the parameter information canvas."""
        return vp.canvas(
            width=self._MAIN_CANVAS_SIZE[0],
            height=self._MAIN_CANVAS_SIZE[1],
            background=vp.color.white,
            align='left'
        )

    def _setup_simulation(self) -> None:
        """Set up all simulation components."""
        # Calculate and set x-positions for balls
        x_positions = self._calculate_x_positions()
        for ball, x_pos in zip(self._balls, x_positions):
            ball.position.x = x_pos

        # Create simulation components
        self._create_grid()
        self._create_ball_visuals()
        self._create_runtime_labels()
        self._create_parameters_labels()
        self._create_graphs()

    def _calculate_x_positions(self) -> list[float]:
        """Calculate x-positions for all balls."""
        grid_range: int = self._grid_range
        num_balls: int = len(self._balls)
        segment_width: float = 2 * grid_range / (num_balls + 1)
        return [-grid_range + segment_width * (i + 1) for i in range(num_balls)]

    def _create_grid(self) -> None:
        """Create the visual grid."""
        grid_range: int = self._grid_range
        step: int = int(grid_range / 10)

        self._canvas.select()

        # Create vertical lines
        for x in vp.arange(-grid_range, grid_range + step, step):
            vp.curve(
                pos=[vp.vector(x, 0, 0), vp.vector(x, grid_range, 0)],
                color=vp.color.gray(0.7)
            )

        # Create horizontal lines
        for y in vp.arange(0, grid_range + step, step):
            vp.curve(
                pos=[vp.vector(-grid_range, y, 0), vp.vector(grid_range, y, 0)],
                color=vp.color.gray(0.7)
            )

            # Add height labels every 2 units
            if y % 2 == 0:
                vp.label(
                    pos=vp.vector(-grid_range - step, y, 0),
                    text=f'{y:.0f}',
                    box=False
                )

        # Add time label below grid
        self._time_label = vp.label(
            pos=vp.vector(-2*step, -step, 0),
            align='left',
            box=False
        )

    def _create_ball_visuals(self) -> None:
        """Create visual representations for all balls."""
        for ball in self._balls:
            ball.create_visual(self._canvas)

    def _create_runtime_labels(self) -> None:
        """Create runtime information labels."""
        line_num: float = self._LABEL_RANGE + self._LABEL_Y_OVERHEAD

        self._runtime_canvas.select()

        for i, ball in enumerate(self._balls):
            # Create ball header label
            vp.label(
                pos=vp.vector(-self._LABEL_RANGE, line_num * self._LABEL_STEP, 0),
                text=f'Ball {i+1}:',
                align='left',
                box=False,
                color=ball.color
            )
            line_num -= 1

            # Create initial height label
            vp.label(
                pos=vp.vector(-self._LABEL_RANGE, line_num * self._LABEL_STEP, 0),
                text=f'  Initial Height: {ball.position.y:.2f} m',
                align='left',
                box=False,
                color=ball.color
            )
            line_num -= 1

            # Create dynamic labels
            self._create_dynamic_labels(ball, line_num)
            line_num -= 6  # Space for next ball's labels

        self._canvas.select()

    def _create_dynamic_labels(self, ball: Ball, start_line: float) -> None:
        """Create dynamic labels for a single ball."""
        line_num = start_line
        label_positions = [
            ('height', '  Height: '),
            ('speed', '  Speed: '),
            ('max_speed', '  Max Speed: '),
            ('terminal_velocity', '  Terminal Velocity Reached? '),
            ('first_impact', '  Time for first impact: '),
            ('stop_time', '  Time to stop: ')
        ]

        for label_type, prefix in label_positions:
            label = vp.label(
                pos=vp.vector(-self._LABEL_RANGE, line_num * self._LABEL_STEP, 0),
                text=prefix,
                align='left',
                box=False,
                color=ball.color
            )
            getattr(self, f'_{label_type}_labels').append(label)
            line_num -= 1

    def _create_parameters_labels(self) -> None:
        """Create parameter information labels."""
        line_num: float = self._LABEL_RANGE + self._LABEL_Y_OVERHEAD

        self._parameter_canvas.select()

        for i, ball in enumerate(self._balls):
            params = [
                (f'Ball {i+1}:', 0),
                ('  Specifications:', 0),
                (f'    Mass: {ball.specs.mass:.4g} kg', 0),
                (f'    Radius: {ball.specs.radius:.2f} m', 0),
                (f'    Drag Coefficient: {ball.specs.drag_coefficient:.2f}', 0),
                ('  Environment:', 0),
                (f'    Gravity: {ball.env.gravity:.2f} m/s²', 0),
                (f'    Air Density: {ball.env.air_density:.2f} kg/m³', 0),
                (f'    CoR: {ball.env.cor:.2f}', 0)
            ]

            for text, extra_space in params:
                vp.label(
                    pos=vp.vector(-self._LABEL_RANGE, line_num * self._LABEL_STEP, 0),
                    text=text,
                    align='left',
                    box=False,
                    color=ball.color
                )
                line_num -= (1 + extra_space)

        self._canvas.select()

    def _create_graphs(self) -> None:
        """Create all graphs for the simulation."""
        # Create velocity graph
        self._velocity_graph = vp.graph(
            title="Velocity vs Time",
            xtitle="Time (s)",
            ytitle="Velocity (m/s)",
            width=self._GRAPH_WIDTH,
            height=self._GRAPH_HEIGHT,
            align='left'
        )

        # Create acceleration graph
        self._acceleration_graph = vp.graph(
            title="Acceleration vs Time",
            xtitle="Time (s)",
            ytitle="Acceleration (m/s²)",
            width=self._GRAPH_WIDTH,
            height=self._GRAPH_HEIGHT,
            align='left'
        )

        # Create position graph
        self._position_graph = vp.graph(
            title="Position vs Time",
            xtitle="Time (s)",
            ytitle="Height (m)",
            width=self._GRAPH_WIDTH,
            height=self._GRAPH_HEIGHT,
            align='left'
        )

        # Create plot curves for each ball
        for i, ball in enumerate(self._balls):
            self._velocity_plots.append(
                vp.gcurve(graph=self._velocity_graph, color=ball.color, label=f'Ball {i+1}')
            )
            self._acceleration_plots.append(
                vp.gcurve(graph=self._acceleration_graph, color=ball.color, label=f'Ball {i+1}')
            )
            self._position_plots.append(
                vp.gcurve(graph=self._position_graph, color=ball.color, label=f'Ball {i+1}')
            )

    def _update_labels(self, t: float) -> None:
        """Update all dynamic labels with current values."""
        if BallDropSimulator._no_gui is False:
            if self._time_label:
                self._time_label.text = f'Time: {t:.2f} s'

            for i, ball in enumerate(self._balls):
                # Update plots
                self._velocity_plots[i].plot(t, ball.velocity.y)
                self._acceleration_plots[i].plot(t, ball.acceleration.y)
                self._position_plots[i].plot(t, ball.position.y)

                # Update labels
                self._height_labels[i].text = f'  Height: {ball.position.y:.2f} m'
                self._speed_labels[i].text = f'  Speed: {abs(ball.velocity.y):.2f} m/s'
                self._max_speed_labels[i].text = f'  Max Speed: {ball.max_speed:.2f} m/s'

                # Update terminal velocity status
                self._terminal_velocity_labels[i].text = (
                    f'  Terminal velocity reached? '
                    f'{"Yes" if ball.terminal_vel_reached else "No"} '
                    f'({ball.terminal_velocity:.2f} m/s)'
                )

                # Update impact and stop times
                if ball.has_hit_ground and ball.first_impact_time is not None:
                    self._first_impact_labels[i].text = (
                        f'  Time for first impact: {ball.first_impact_time:.2f} s'
                    )

                if ball.has_stopped and ball.stop_time is not None:
                    self._stop_time_labels[i].text = (
                        f'  Time to stop: {ball.stop_time:.2f} s'
                    )


def main() -> None:
    """Main function to run the simulation."""
    def _get_user_input() -> list[Ball]:
        def _get_float(prompt: str,
                       default_value: Optional[float] = None,
                       min_value: float = float('-inf'),
                       max_value: float = float('inf')) -> float:
            while True:
                try:
                    value = float(input(prompt))
                    if min_value <= value <= max_value:
                        return value
                    else:
                        print(f"Please enter a value between {min_value} and {max_value}.")
                except ValueError:
                    if default_value is not None:
                        return default_value
                    else:
                        print("Please enter a valid number.")

        def _get_ball_spec(ball_num):
            mass = _get_float(f'Enter mass for Ball {ball_num} (kg): <{
                              BallSpecsDefaults.MASS}> ', BallSpecsDefaults.MASS)
            radius = _get_float(f'Enter radius for Ball {ball_num} (m): <{
                                BallSpecsDefaults.RADIUS}> ', BallSpecsDefaults.RADIUS)
            drag_coeff = _get_float(f'Enter drag coefficient for Ball {ball_num}: <{
                                    BallSpecsDefaults.SPHERE_DRAG_COEFFICIENT}> ',
                                    BallSpecsDefaults.SPHERE_DRAG_COEFFICIENT)
            return BallSpecs(mass, radius, drag_coeff)

        def _get_env(ball_num):
            gravity = _get_float(f'Enter gravity for Ball {
                                 ball_num} (m/s²): <{EnvironmentDefaults.EARTH_GRAVITY}> ',
                                 EnvironmentDefaults.EARTH_GRAVITY)
            air_density = _get_float(f'Enter air density for Ball {
                                     ball_num} (kg/m³): <{EnvironmentDefaults.EARTH_AIR_DENSITY}> ',
                                     EnvironmentDefaults.EARTH_AIR_DENSITY)
            cor = _get_float(f'Enter CoR for Ball {ball_num}: <{
                             EnvironmentDefaults.DEFAULT_COR}> ', EnvironmentDefaults.DEFAULT_COR)
            return Environment(gravity, air_density, cor)

        choice: str = 'y'
        balls: list[Ball] = []
        i: int = 0
        MAX_BALLS: Final[int] = 3
        colors: list[vp.vector] = [vp.color.blue, vp.color.red, vp.color.magenta]

        while choice == 'y' and i < MAX_BALLS:
            balls.append(Ball())
            balls[i].init_height = _get_float(f'Enter Height for Ball {i+1}: <10> ', 10)
            balls[i].specs = _get_ball_spec(i+1)
            balls[i].env = _get_env(i+1)
            balls[i].color = colors[i]
            i += 1
            if i < MAX_BALLS:
                choice = input("Do you want to create another Ball (y/n, max 3) <n>: ").lower()
            else:
                choice = 'n'

        return balls

    parser = argparse.ArgumentParser(description='Ball Drop Simulator')
    parser.add_argument('--test', action='store_true', help='Run with pre-defined test cases')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI')
    args = parser.parse_args()

    if args.no_gui is True:
        BallDropSimulator.disable_gui(True)

    if args.test:
        # Create the Ball Specs
        ball1_spec: BallSpecs = BallSpecs(
            mass=300,
            radius=5,
            drag_coefficient=BallSpecsDefaults.SPHERE_DRAG_COEFFICIENT
        )
        ball2_spec: BallSpecs = BallSpecs(
            mass=100,
            radius=3,
            drag_coefficient=BallSpecsDefaults.SPHERE_DRAG_COEFFICIENT/2
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
        balls: list[Ball] = [ball1, ball2]
    else:
        balls: list[Ball] = _get_user_input()

    # Create BallDropSimulator with both balls and run it
    sim = BallDropSimulator(balls)
    sim.run()

    if args.no_gui is False:
        print("Press any key to exit...")
        readchar.readkey()
        BallDropSimulator.quit_simulation()


if __name__ == "__main__":
    main()
