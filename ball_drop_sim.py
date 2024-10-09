"""
Module to simulate the dropping of balls under gravity. The simulation can handle
multiple balls, track their velocity, acceleration, position over time, and display
these values graphically.
"""
from typing import Final, Optional
import argparse
import readchar
import vpython as vp
from ball_spec import BallSpec, BallSpecDefaults
from environment import Environment, EnvironmentDefaults
from ball import Ball

__author__ = 'Jim Tooker'


class BallDropSimulator:
    """
    A class to simulate the drop of multiple balls under the influence of gravity
    and environmental factors. It supports both graphical simulation with VPython
    and command-line output.
    """

    # Class constants for graphical and display parameters
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
        Initialize the BallDropSimulator with a list of Ball objects.

        Args:
            balls (list[Ball]): List of Ball objects to be simulated.
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
            # Initialize canvases for graphical output
            self._canvas: vp.canvas = self._create_main_canvas()
            self._runtime_canvas: vp.canvas = self._create_runtime_canvas()
            self._parameter_canvas: vp.canvas = self._create_parameter_canvas()

            # Initialize label containers for graphical display
            self._height_labels: list[vp.label] = []
            self._speed_labels: list[vp.label] = []
            self._max_speed_labels: list[vp.label] = []
            self._terminal_velocity_labels: list[vp.label] = []
            self._first_impact_labels: list[vp.label] = []
            self._stop_time_labels: list[vp.label] = []

            # Initialize plot containers for graphical data
            self._velocity_plots: list[vp.gcurve] = []
            self._acceleration_plots: list[vp.gcurve] = []
            self._position_plots: list[vp.gcurve] = []

            # Setup simulation components
            self._setup_simulation()
        else:
            # If no GUI, print details of each ball to the console
            for i, ball in enumerate(balls):
                print(f'\nBall{i+1}:')
                print(f'  {ball.specs}')
                print(f'  {ball.env}')
                print(f'  Initial Height: {ball.init_height:.2f} m')

    @staticmethod
    def quit_simulation() -> None:
        """
        Stop the VPython server and quit the simulation.
        """
        if BallDropSimulator._no_gui is False:
            # We don't import vp_services until needed, because importing it will start
            # the server, if not started already.
            import vpython.no_notebook as vp_services  # type: ignore[import-untyped]
            vp_services.stop_server()

    @classmethod
    def disable_gui(cls, no_gui: bool) -> None:
        """
        Enables or disables the GUI based on user input.

        Args:
            no_gui (bool): If True, disable the GUI.
        """
        cls._no_gui = no_gui

    @property
    def balls(self) -> list[Ball]:
        """
        Retrieve the list of balls being simulated.

        Returns:
            list[Ball]: The list of balls in the simulation.
        """
        return self._balls

    @property
    def total_time(self) -> float:
        """
        Retrieve the total time of the simulation.

        Returns:
            float: The total time elapsed in the simulation.
        """
        return self._total_time

    @property
    def _max_height(self) -> float:
        """
        Calculate and return the maximum height among all the balls.

        Returns:
            float: The maximum height of any ball in the simulation.
        """
        return max(ball.position.y for ball in self._balls)

    @property
    def _grid_range(self) -> int:
        """
        Determine the grid range for the simulation based on the maximum height.

        Returns:
            int: The calculated grid range.
        """
        return int(self._max_height)

    def run(self) -> None:
        """
        Start and run the ball drop simulation. If GUI is enabled, the simulation
        will display real-time graphical information. Otherwise, the results will be printed.
        """
        FPS: Final[int] = 100  # Simulation frame rate (Frames Per Second)
        dt: float = 1/FPS  # Time step for the simulation
        t: float = 0  # Initial time

        # Initialize and update the labels with the starting time
        self._update_labels(t)

        # Run the simulation until all balls have stopped
        while True:
            vp.rate(FPS)
            t += dt

            # Update each ball's state
            for ball in self._balls:
                ball.update(dt, t)

            # Update the labels and graphical data
            self._update_labels(t)

            # Stop the simulation if all balls have stopped
            if all(ball.has_stopped for ball in self._balls):
                msg: str = f'Total Time: {t:.2f} s'
                if self._time_label:
                    self._time_label.text = msg
                else:
                    print(f'\n{msg}')
                break

        # Store the total simulation time
        self._total_time = t

        # If no GUI, print ball data to the console
        if BallDropSimulator._no_gui is True:
            for i, ball in enumerate(self._balls):
                print(f'\nBall{i+1}:')
                print(f'  Max speed: {ball.max_speed:.2f} m/s')
                print(f'  Terminal velocity reached?: {ball.terminal_vel_reached}. '
                      f'({ball.terminal_velocity:.2f} m/s)')
                print(f'  Time for 1st impact: {ball.first_impact_time:.2f} s')
                print(f'  Time to stop: {ball.stop_time:.2f} s')

    def _validate_balls(self, balls: list[Ball]) -> None:
        """
        Validate that the balls parameter is a list of Ball objects.

        Args:
            balls (list[Ball]): The list of Ball objects to validate.

        Raises:
            ValueError: If the input is not a list or contains invalid elements.
        """
        if not isinstance(balls, list):
            raise ValueError("'balls' parameter must be a list")
        if not all(isinstance(ball, Ball) for ball in balls):
            raise ValueError("All elements in 'balls' must be instances of Ball")
        if not balls:
            raise ValueError("'balls' list cannot be empty")

    def _create_main_canvas(self) -> vp.canvas:
        """Create and return the main simulation canvas."""
        return vp.canvas(
            title='Ball Drop Simulator',
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
        """Set up all components of the simulation including the grid, ball visuals, labels, and graphs."""
        # Calculate and set x-positions for all balls
        x_positions = self._calculate_x_positions()
        for ball, x_pos in zip(self._balls, x_positions):
            ball.position.x = x_pos  # Set the x-position for each ball

        # Create various components needed for the simulation
        self._create_grid()
        self._create_ball_visuals()
        self._create_runtime_labels()
        self._create_parameters_labels()
        self._create_graphs()

    def _calculate_x_positions(self) -> list[float]:
        """Calculate evenly spaced x-positions for all balls based on the grid size.

        Returns:
            list[float]: The calculated x-positions for each ball.
        """
        grid_range: int = self._grid_range
        num_balls: int = len(self._balls)
        segment_width: float = 2 * grid_range / (num_balls + 1)

        # Calculate and return x-positions for each ball by placing them within segments along the x-axis
        return [-grid_range + segment_width * (i + 1) for i in range(num_balls)]

    def _create_grid(self) -> None:
        """Create the visual grid with both horizontal and vertical lines, and add height labels."""
        grid_range: int = self._grid_range
        step: int = int(grid_range / 10)

        self._canvas.select()  # Activate the canvas for drawing

        # Draw vertical lines across the grid
        for x in vp.arange(-grid_range, grid_range + step, step):
            vp.curve(
                pos=[vp.vector(x, 0, 0), vp.vector(x, grid_range, 0)],
                color=vp.color.gray(0.7)  # Light gray grid lines
            )

        # Draw horizontal lines and add height labels
        for y in vp.arange(0, grid_range + step, step):
            vp.curve(
                pos=[vp.vector(-grid_range, y, 0), vp.vector(grid_range, y, 0)],
                color=vp.color.gray(0.7)
            )

            # Add height labels on every second line
            if y % 2 == 0:
                vp.label(
                    pos=vp.vector(-grid_range - step, y, 0),
                    text=f'{y:.0f}',  # Display height value
                    box=False
                )

        # Add the time label beneath the grid
        self._time_label = vp.label(
            pos=vp.vector(-2 * step, -step, 0),
            align='left',
            box=False
        )

    def _create_ball_visuals(self) -> None:
        """Create visual representations of the balls in the simulation."""
        for ball in self._balls:
            ball.create_visual(self._canvas)  # Each ball creates its own visual

    def _create_runtime_labels(self) -> None:
        """Create labels that display runtime information such as height, speed, and impact times."""
        line_num: float = self._LABEL_RANGE + self._LABEL_Y_OVERHEAD  # Start label positioning

        self._runtime_canvas.select()  # Select the canvas for runtime labels

        for i, ball in enumerate(self._balls):
            # Header label for each ball
            vp.label(
                pos=vp.vector(-self._LABEL_RANGE, line_num * self._LABEL_STEP, 0),
                text=f'Ball {i + 1}:',
                align='left',
                box=False,
                color=ball.color  # Color matching the ball
            )
            line_num -= 1

            # Label for initial height
            vp.label(
                pos=vp.vector(-self._LABEL_RANGE, line_num * self._LABEL_STEP, 0),
                text=f'  Initial Height: {ball.position.y:.2f} m',
                align='left',
                box=False,
                color=ball.color
            )
            line_num -= 1

            # Create dynamic labels for the ball (e.g., height, speed)
            self._create_dynamic_labels(ball, line_num)
            line_num -= 6  # Space for labels of the next ball

        self._canvas.select()  # Return control to the main canvas

    def _create_dynamic_labels(self, ball: Ball, start_line: float) -> None:
        """Create dynamic labels for displaying real-time ball properties (e.g., height, speed).

        Args:
            ball (Ball): The ball for which to create the dynamic labels.
            start_line (float): The starting vertical position for the labels.
        """
        line_num = start_line
        label_positions = [
            ('height', '  Height: '),
            ('speed', '  Speed: '),
            ('max_speed', '  Max Speed: '),
            ('terminal_velocity', '  Terminal Velocity Reached? '),
            ('first_impact', '  Time for first impact: '),
            ('stop_time', '  Time to stop: ')
        ]

        # Create and store labels for each ball property
        for label_type, prefix in label_positions:
            label = vp.label(
                pos=vp.vector(-self._LABEL_RANGE, line_num * self._LABEL_STEP, 0),
                text=prefix,
                align='left',
                box=False,
                color=ball.color
            )
            # Store the label in the corresponding list (e.g., _height_labels, _speed_labels)
            getattr(self, f'_{label_type}_labels').append(label)
            line_num -= 1

    def _create_parameters_labels(self) -> None:
        """Create labels to display the ball specifications and environment parameters."""
        line_num: float = self._LABEL_RANGE + self._LABEL_Y_OVERHEAD  # Initial label position

        self._parameter_canvas.select()  # Select canvas for parameter labels

        for i, ball in enumerate(self._balls):
            # List of ball specification and environment labels to display
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

            # Display each parameter label
            for text, extra_space in params:
                vp.label(
                    pos=vp.vector(-self._LABEL_RANGE, line_num * self._LABEL_STEP, 0),
                    text=text,
                    align='left',
                    box=False,
                    color=ball.color
                )
                line_num -= (1 + extra_space)  # Adjust line for spacing

        self._canvas.select()  # Return to the main canvas

    def _create_graphs(self) -> None:
        """Create graphs for velocity, acceleration, and position over time."""
        # Create graph for velocity vs time
        self._velocity_graph = vp.graph(
            title="Velocity vs Time",
            xtitle="Time (s)",
            ytitle="Velocity (m/s)",
            width=self._GRAPH_WIDTH,
            height=self._GRAPH_HEIGHT,
            align='left'
        )

        # Create graph for acceleration vs time
        self._acceleration_graph = vp.graph(
            title="Acceleration vs Time",
            xtitle="Time (s)",
            ytitle="Acceleration (m/s²)",
            width=self._GRAPH_WIDTH,
            height=self._GRAPH_HEIGHT,
            align='left'
        )

        # Create graph for position vs time
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
        """Update the dynamic labels and plots for each ball during the simulation.

        Args:
            t (float): The current time in the simulation.
        """
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

                # If ball has stopped, print the stop time
                if ball.has_stopped and ball.stop_time is not None:
                    self._stop_time_labels[i].text = (
                        f'  Time to stop: {ball.stop_time:.2f} s'
                    )


def main() -> None:
    """
    Main function to run the Ball Drop Simulation.
    
    The function first retrieves user input or a test case to initialize 
    balls with specific properties. It then creates a BallDropSimulator instance 
    and runs the simulation, with or without a graphical user interface (GUI) 
    based on the provided arguments.
    """
    
    def _get_user_input() -> list[Ball]:
        """
        Prompts the user to enter the properties for up to three balls to be
        simulated. The user provides the initial height, ball specifications, 
        and environment parameters for each ball.

        Returns:
            list[Ball]: A list of balls created with user-provided attributes.
        """

        def _get_float(prompt: str,
                       default_value: Optional[float] = None,
                       min_value: float = float('-inf'),
                       max_value: float = float('inf')) -> float:
            """
            Utility function to get a floating-point number from the user within a specified range.
            
            Args:
                prompt (str): The message to display for input.
                default_value (Optional[float]): The default value if no input is given.
                min_value (float): The minimum acceptable value.
                max_value (float): The maximum acceptable value.
            
            Returns:
                float: The validated floating-point value from the user.
            """
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

        def _get_ball_spec(ball_num: int) -> BallSpec:
            """
            Retrieves the ball specifications (mass, radius, drag coefficient) from the user.

            Args:
                ball_num (int): The ball number for display in the prompt.

            Returns:
                BallSpec: The ball specification object with user input or defaults.
            """
            mass: float = _get_float(f'Enter mass for Ball {ball_num} (kg): <{BallSpecDefaults.MASS}> ', 
                                     BallSpecDefaults.MASS)
            radius: float = _get_float(f'Enter radius for Ball {ball_num} (m): <{BallSpecDefaults.RADIUS}> ', 
                                       BallSpecDefaults.RADIUS)
            drag_coeff: float = _get_float(f'Enter drag coefficient for Ball {ball_num}: <{BallSpecDefaults.SPHERE_DRAG_COEFFICIENT}> ', 
                                           BallSpecDefaults.SPHERE_DRAG_COEFFICIENT)
            return BallSpec(mass, radius, drag_coeff)

        def _get_env(ball_num: int) -> Environment:
            """
            Retrieves the environment properties (gravity, air density, CoR) from the user.

            Args:
                ball_num (int): The ball number for display in the prompt.

            Returns:
                Environment: The environment object with user input or defaults.
            """
            gravity: float = _get_float(f'Enter gravity for Ball {ball_num} (m/s²): <{EnvironmentDefaults.EARTH_GRAVITY}> ', 
                                        EnvironmentDefaults.EARTH_GRAVITY)
            air_density: float = _get_float(f'Enter air density for Ball {ball_num} (kg/m³): <{EnvironmentDefaults.EARTH_AIR_DENSITY}> ', 
                                            EnvironmentDefaults.EARTH_AIR_DENSITY)
            cor: float = _get_float(f'Enter CoR for Ball {ball_num}: <{EnvironmentDefaults.DEFAULT_COR}> ', 
                                    EnvironmentDefaults.DEFAULT_COR)
            return Environment(gravity, air_density, cor)

        # User can add up to 3 balls
        choice: str = 'y'
        balls: list[Ball] = []
        i: int = 0
        MAX_BALLS: Final[int] = 3  # Maximum allowed balls
        colors: list[vp.vector] = [vp.color.blue, vp.color.red, vp.color.magenta]  # Color options for balls

        # Loop to gather ball input
        while choice == 'y' and i < MAX_BALLS:
            balls.append(Ball())  # Add a new ball
            balls[i].init_height = _get_float(f'Enter Height for Ball {i+1}: <10> ', 10)
            balls[i].specs = _get_ball_spec(i+1)
            balls[i].env = _get_env(i+1)
            balls[i].color = colors[i]  # Assign color
            i += 1  # Increment ball count

            if i < MAX_BALLS:
                choice = input("Do you want to create another Ball (y/n, max 3) <n>: ").lower()
            else:
                choice = 'n'  # Stop if max balls reached

        return balls

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Ball Drop Simulator')
    parser.add_argument('--test', action='store_true', help='Run with pre-defined test cases')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI')
    args = parser.parse_args()

    # Disable GUI if specified in arguments
    if args.no_gui is True:
        BallDropSimulator.disable_gui(True)

    balls: list[Ball]  # Placeholder for balls

    if args.test:
        # Test mode: create pre-defined ball specs and environments
        ball1_spec: BallSpec = BallSpec(
            mass=300,
            radius=5,
            drag_coefficient=BallSpecDefaults.SPHERE_DRAG_COEFFICIENT
        )
        ball2_spec: BallSpec = BallSpec(
            mass=100,
            radius=3,
            drag_coefficient=BallSpecDefaults.SPHERE_DRAG_COEFFICIENT / 2
        )

        # Two different environments for the balls
        env1: Environment = Environment(
            gravity=EnvironmentDefaults.EARTH_GRAVITY,
            air_density=EnvironmentDefaults.EARTH_AIR_DENSITY,
            cor=0.9
        )
        env2: Environment = Environment(
            gravity=EnvironmentDefaults.EARTH_GRAVITY,
            air_density=0.3,  # Less air density
            cor=EnvironmentDefaults.DEFAULT_COR
        )

        # Create two balls with different specs and environments
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
        balls = [ball1, ball2]  # Pre-defined test case with two balls

    else:
        # Normal mode: get user input for ball creation
        balls = _get_user_input()

    # Create a BallDropSimulator instance with the balls and start the simulation
    sim = BallDropSimulator(balls)
    sim.run()

    # If GUI is enabled, wait for the user to press a key before exiting
    if args.no_gui is False:
        print("Press any key to exit...")
        readchar.readkey()  # Wait for key press to exit
        BallDropSimulator.quit_simulation()


if __name__ == "__main__":
    main()
