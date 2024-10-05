"""
FIXME - Need Module docstring
"""
from typing import Final
import vpython as vp
from ball_specs import BallSpecs, BallSpecsDefaults
from environment import Environment, EnvironmentDefaults
from ball import Ball


class Simulation:
    """
    FIXME
    """
    # Class constants
    _LABEL_RANGE: Final[int] = 10
    _LABEL_STEP: Final[float] = 0.75
    _GRAPH_WIDTH: Final[int] = 600
    _GRAPH_HEIGHT: Final[int] = 400
    _MAIN_CANVAS_SIZE: Final[tuple[int, int]] = (600, 600)

    def __init__(self, balls: list[Ball]) -> None:
        """
        Initialize simulation with list of balls.

        Args:
            list[Ball]: List of Ball objects to simulate dropping
        """
        self._validate_balls(balls)
        self._balls: list[Ball] = balls

        # Initialize canvases
        self._canvas: vp.canvas = self._create_main_canvas()
        self._runtime_canvas: vp.canvas = self._create_runtime_canvas()
        self._parameter_canvas: vp.canvas = self._create_parameter_canvas()

        # Initialize all label containers
        self._time_label: vp.label
        self._height_labels: list[vp.label] = []
        self._speed_labels: list[vp.label] = []
        self._max_speed_labels: list[vp.label] = []
        self._terminal_velocity_labels: list[vp.label] = []
        self._first_impact_labels: list[vp.label] = []
        self._stop_time_labels: list[vp.label] = []

        # Initialize plot containers
        self._velocity_graph: vp.graph
        self._acceleration_graph: vp.graph
        self._position_graph: vp.graph
        self._velocity_plots: list[vp.gcurve] = []
        self._acceleration_plots: list[vp.gcurve] = []
        self._position_plots: list[vp.gcurve] = []

        # Setup simulation
        self._setup_simulation()

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
            title='Ball Drop Simulation',
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

    @property
    def _max_height(self) -> float:
        """Calculate maximum height among all balls."""
        return max(ball.position.y for ball in self._balls)

    @property
    def _grid_range(self) -> int:
        """Calculate grid range based on maximum height."""
        return int(self._max_height)

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

        # Add time label
        self._time_label = vp.label(
            pos=vp.vector(0, -step, 0),
            align='center',
            box=False
        )

    def _create_ball_visuals(self) -> None:
        """Create visual representations for all balls."""
        for ball in self._balls:
            ball.create_visual(self._canvas)

    def _create_runtime_labels(self) -> None:
        """Create runtime information labels."""
        line_num: float = self._LABEL_RANGE

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
                text=f'  Initial Height: {ball.position.y:.3g} m',
                align='left',
                box=False,
                color=ball.color
            )
            line_num -= 1

            # Create dynamic labels
            self._create_dynamic_labels(ball, line_num)
            line_num -= 8  # Space for next ball's labels

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
        line_num: float = self._LABEL_RANGE

        self._parameter_canvas.select()

        for i, ball in enumerate(self._balls):
            params = [
                (f'Ball {i+1}:', 0),
                ('  Specifications:', 0),
                (f'    Mass: {ball.specs.mass:.4g} kg', 0),
                (f'    Radius: {ball.specs.radius:.3g} m', 0),
                (f'    Drag Coefficient: {ball.specs.drag_coefficient:.3g}', 0),
                ('  Environment:', 0),
                (f'    Gravity: {ball.env.gravity:.3g} m/s²', 0),
                (f'    Air Density: {ball.env.air_density:.3g} kg/m³', 0),
                (f'    CoR: {ball.env.cor:.3g}', 1)
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
        if self._time_label:
            self._time_label.text = f'Time: {t:.3g} secs'

        for i, ball in enumerate(self._balls):
            # Update plots
            self._velocity_plots[i].plot(t, ball.velocity.y)
            self._acceleration_plots[i].plot(t, ball.acceleration.y)
            self._position_plots[i].plot(t, ball.position.y)

            # Update labels
            self._height_labels[i].text = f'  Height: {ball.position.y:.3g} m'
            self._speed_labels[i].text = f'  Speed: {abs(ball.velocity.y):.3g} m/s'
            self._max_speed_labels[i].text = f'  Max Speed: {ball.v_max:.3g} m/s'

            # Update terminal velocity status
            self._terminal_velocity_labels[i].text = (
                f'  Terminal velocity reached? '
                f'{"Yes" if ball.terminal_vel_reached else "No"} '
                f'({ball.terminal_velocity:.3g} m/s)'
            )

            # Update impact and stop times
            if ball.has_hit_ground and ball.first_impact_time is not None:
                self._first_impact_labels[i].text = (
                    f'  Time for first impact: {ball.first_impact_time:.3g} secs'
                )

            if ball.has_stopped and ball.stop_time is not None:
                self._stop_time_labels[i].text = (
                    f'  Time to stop: {ball.stop_time:.3g} secs'
                )

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
                if self._time_label:
                    self._time_label.text = f'Total Time: {t:.3g} secs'
                break


def main() -> None:
    """Main function to run the simulation."""
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
        air_density=0.30,
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

    # Create Simulation with both balls
    sim = Simulation([ball1, ball2])
    sim.run()
    print('Done')

if __name__ == "__main__":
    main()
