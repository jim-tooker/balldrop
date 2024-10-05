"""
FIXME - Need Module docstring
"""
import math
from dataclasses import dataclass
from typing import Final, Optional
import vpython as vp

class BallSpecsDefaults:
    MASS: Final[float] = 1.0  # kg
    RADIUS: Final[float] = 1.0  # meters
    SPHERE_DRAG_COEFFICIENT: Final[float] = 0.47

@dataclass
class BallSpecs:
    mass: float = BallSpecsDefaults.MASS
    radius: float = BallSpecsDefaults.RADIUS
    drag_coefficient: float = BallSpecsDefaults.SPHERE_DRAG_COEFFICIENT

class EnvironmentDefaults:
    EARTH_GRAVITY: Final[float] = 9.80665  # m/s²
    EARTH_AIR_DENSITY: Final[float] = 1.225  # kg/m³
    DEFAULT_COR: Final[float] = 0.8  # Coefficient of Restitution

@dataclass
class Environment:
    gravity: float = EnvironmentDefaults.EARTH_GRAVITY
    air_density: float = EnvironmentDefaults.EARTH_AIR_DENSITY
    cor: float = EnvironmentDefaults.DEFAULT_COR


class Ball:
    _MIN_VISUAL_RADIUS: Final[float] = 0.02

    def __init__(self,
                 specs: BallSpecs,
                 env: Environment,
                 init_height: float,
                 color: vp.vector) -> None:
        """
        Initialize a Ball object with given specifications and environment.

        Args:
            specs: Ball specifications including mass, radius, and drag coefficient
            env: Environment specifications including gravity, air density, and coefficient of restitution
            init_height: Initial height of the ball
            color: Color vector for the ball's visual representation
        """
        self._validate_inputs(specs, env, init_height, color)
        
        # Protected instance variables
        self._specs: BallSpecs = specs
        self._env: Environment = env
        self._init_height: float = init_height
        self._radius: float = specs.radius
        self._color: vp.vector = color
        self._mass: float = specs.mass
        
        self._position: vp.vector = vp.vector(0, init_height, 0)
        self._velocity: vp.vector = vp.vector(0, 0, 0)
        self._v_max: float = 0.0
        self._terminal_vel_reached: bool = False
        self._has_hit_ground: bool = False
        self._first_impact_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._has_stopped: bool = False
        self._sphere: Optional[vp.sphere] = None

    def _validate_inputs(self, specs: BallSpecs, env: Environment, 
                        init_height: float, color: vp.vector) -> None:
        """Validate input parameters."""
        if not isinstance(specs, BallSpecs):
            raise ValueError("'specs' parameter must be an instance of BallSpecs")
        if not isinstance(env, Environment):
            raise ValueError("'env' parameter must be an instance of Environment")
        if not isinstance(init_height, (int, float)):
            raise ValueError("'init_height' parameter must be a numeric value")
        if init_height <= 0:
            raise ValueError("'init_height' parameter must be positive")
        if not isinstance(color, vp.vector):
            raise ValueError("'color' parameter must be a valid vp.vector object")

    @property
    def specs(self) -> BallSpecs:
        """Get ball specifications."""
        return self._specs

    @property
    def env(self) -> Environment:
        """Get environment specifications."""
        return self._env

    @property
    def color(self) -> vp.vector:
        """Get ball color."""
        return self._color

    @property
    def has_stopped(self) -> bool:
        """Get ball stopped status."""
        return self._has_stopped

    @property
    def has_hit_ground(self) -> bool:
        """Get ball ground hit status."""
        return self._has_hit_ground

    @property
    def first_impact_time(self) -> Optional[float]:
        """Get time of first ground impact."""
        return self._first_impact_time

    @property
    def stop_time(self) -> Optional[float]:
        """Get time when ball stopped."""
        return self._stop_time

    @property
    def v_max(self) -> float:
        """Get maximum velocity reached."""
        return self._v_max

    @property
    def terminal_vel_reached(self) -> bool:
        """Get terminal velocity reached status."""
        return self._terminal_vel_reached

    @property
    def position(self) -> vp.vector:
        """Get current position."""
        return self._position

    @property
    def velocity(self) -> vp.vector:
        """Get current velocity."""
        return self._velocity

    @property
    def visual_radius(self) -> float:
        """Calculate the visual radius for rendering."""
        return max(self._radius, self._init_height * self._MIN_VISUAL_RADIUS)

    @property
    def sphere_pos(self) -> vp.vector:
        """Calculate sphere position accounting for visual radius."""
        return self._position + vp.vector(0, self.visual_radius, 0)

    @property
    def cross_section_area(self) -> float:
        """Calculate cross-sectional area of the sphere."""
        return math.pi * self._radius**2

    @property
    def speed(self) -> float:
        """Calculate current speed magnitude."""
        return float(vp.mag(self._velocity))

    @property
    def air_resistance(self) -> float:
        """Calculate air resistance force."""
        return (0.5 * self.cross_section_area * self.speed**2 * 
                self._env.air_density * self._specs.drag_coefficient)

    @property
    def acceleration(self) -> vp.vector:
        """Calculate current acceleration vector."""
        # If we've stopped, accelerate is 0
        if self._has_stopped:
            return vp.vector(0,0,0)

        gravity_acc = vp.vector(0, -self._env.gravity, 0)
        drag_acc = (-self._velocity.norm() * self.air_resistance / self._mass 
                   if self.speed > 0 else vp.vector(0, 0, 0))
        return gravity_acc + drag_acc

    @property
    def terminal_velocity(self) -> float:
        """Calculate theoretical terminal velocity."""
        if (self._env.air_density == 0 or 
            self.cross_section_area == 0 or 
            self._specs.drag_coefficient == 0):
            return float('inf')
        return math.sqrt((2 * self._mass * self._env.gravity) /
                        (self._env.air_density * self.cross_section_area * 
                         self._specs.drag_coefficient))

    def create_visual(self, canvas: vp.canvas) -> None:
        """Create visual representation of the ball."""
        self._sphere = vp.sphere(
            canvas=canvas,
            pos=self.sphere_pos,
            radius=self.visual_radius,
            color=self._color
        )

    def update(self, dt: float, current_time: float) -> None:
        """
        Update ball physics for the current time step.

        Args:
            dt: Time step duration
            current_time: Current simulation time
        """
        # Update velocity using acceleration
        self._velocity += self.acceleration * dt

        # Update physical position
        self._position += self._velocity * dt

        # Update visual position
        if self._sphere is not None:
            self._sphere.pos = self.sphere_pos

        # Update max speed
        current_speed = abs(self._velocity.y)
        self._v_max = max(self._v_max, current_speed)

        # Check if terminal velocity has been reached
        if not self._terminal_vel_reached and math.isclose(
            current_speed, self.terminal_velocity, abs_tol=0.005):
            self._terminal_vel_reached = True

        if self._position.y <= 0:
            # Ensure position is at ground level
            self._position.y = 0

            # Update visual position
            if self._sphere is not None:
                self._sphere.pos = self.sphere_pos

            if not self._has_hit_ground:
                self._has_hit_ground = True
                self._first_impact_time = current_time

            # Check for minimum speed
            MIN_SPEED: Final[float] = self._env.gravity * dt
            if abs(self._velocity.y) <= MIN_SPEED:
                self._velocity.y = 0
                if not self._has_stopped:
                    self._has_stopped = True
                    self._stop_time = current_time
            else:
                # Apply coefficient of restitution
                self._velocity.y = -self._velocity.y * self._env.cor


class Simulation:
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
            balls: List of Ball objects to simulate
        """
        self._validate_balls(balls)
        self._balls: list[Ball] = balls

        # Initialize canvases
        self._canvas: vp.canvas = self._create_main_canvas()
        self._runtime_canvas: vp.canvas = self._create_runtime_canvas()
        self._parameter_canvas: vp.canvas = self._create_parameter_canvas()

        # Initialize all label containers
        self._time_label: Optional[vp.label] = None
        self._height_labels: list[vp.label] = []
        self._speed_labels: list[vp.label] = []
        self._max_speed_labels: list[vp.label] = []
        self._terminal_velocity_labels: list[vp.label] = []
        self._first_impact_labels: list[vp.label] = []
        self._stop_time_labels: list[vp.label] = []

        # Initialize plot containers
        self._velocity_graph: Optional[vp.graph] = None
        self._acceleration_graph: Optional[vp.graph] = None
        self._position_graph: Optional[vp.graph] = None
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
            self._create_dynamic_labels(ball, i, line_num)
            line_num -= 8  # Space for next ball's labels

        self._canvas.select()

    def _create_dynamic_labels(self, ball: Ball, index: int, start_line: float) -> None:
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
