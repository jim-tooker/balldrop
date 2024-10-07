"""
FIXME - Need Module docstring
"""
import math
from typing import Final, Optional
import vpython as vp
from ball_specs import BallSpecs
from environment import Environment


class Ball:
    """
    FIXME
    """
    _MIN_VISUAL_RADIUS: Final[float] = 0.02

    def __init__(self,
                 specs: BallSpecs = BallSpecs(),
                 env: Environment = Environment(),
                 init_height: float = 10,
                 color: vp.vector = vp.color.red) -> None:
        """
        Initialize a Ball object with given specifications and environment.

        Args:
            specs: Ball specifications including mass, radius, and drag coefficient
            env: Environment specifications including gravity, air density, and coefficient of restitution
            init_height: Initial height of the ball
            color: Color vector for the ball's visual representation (Default red)
        """
        self._validate_inputs(specs, env, init_height, color)

        self._specs: BallSpecs = specs
        self._env: Environment = env
        self._init_height: float = init_height
        self._radius: float = specs.radius
        self._color: vp.vector = color
        self._mass: float = specs.mass

        self._position: vp.vector = vp.vector(0, init_height, 0)
        self._velocity: vp.vector = vp.vector(0, 0, 0)
        self._max_speed: float = 0
        self._terminal_vel_reached: bool = False
        self._has_hit_ground: bool = False
        self._first_impact_time: float = 0
        self._has_stopped: bool = False
        self._stop_time: float = 0
        self._sphere: Optional[vp.sphere] = None

    @property
    def specs(self) -> BallSpecs:
        """Get ball specifications."""
        return self._specs

    @property
    def env(self) -> Environment:
        """Get environment specifications."""
        return self._env

    @property
    def init_height(self) -> float:
        """Get Initial height."""
        return self._init_height

    @property
    def color(self) -> vp.vector:
        """Get ball color."""
        return self._color

    @property
    def position(self) -> vp.vector:
        """Get current position."""
        return self._position

    @property
    def velocity(self) -> vp.vector:
        """Get current velocity."""
        return self._velocity

    @property
    def max_speed(self) -> float:
        """Get maximum speed reached."""
        return self._max_speed

    @property
    def terminal_vel_reached(self) -> bool:
        """Get terminal velocity reached status."""
        return self._terminal_vel_reached

    @property
    def has_hit_ground(self) -> bool:
        """Get ball ground hit status."""
        return self._has_hit_ground

    @property
    def first_impact_time(self) -> float:
        """Get time of first ground impact."""
        return self._first_impact_time

    @property
    def has_stopped(self) -> bool:
        """Get ball stopped status."""
        return self._has_stopped

    @property
    def stop_time(self) -> float:
        """Get time when ball stopped."""
        return self._stop_time

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
            return math.inf
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
        self._max_speed = max(self._max_speed, current_speed)

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
