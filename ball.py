"""
This module defines the Ball class, which is responsible for representing a physical ball
in the ball drop simulation. The Ball class handles physics calculations like velocity, 
acceleration, and air resistance, and provides visual rendering using the Vpython library.

The Ball class interacts with the Environment and BallSpec classes to incorporate 
environmental and physical properties such as gravity, air resistance, mass, and drag.
"""
import math
from typing import Final, Optional
import vpython as vp
from ball_spec import BallSpec
from environment import Environment

__author__ = 'Jim Tooker'


class Ball:
    """
    Represents a ball in the ball drop simulation with physics and visual rendering capabilities.

    The Ball class updates its physical properties (position, velocity, etc.) and renders
    a visual representation using the Vpython library. It incorporates gravity, air resistance,
    and restitution, and tracks whether the ball has hit the ground or stopped moving.
    """
    _MIN_VISUAL_RADIUS: Final[float] = 0.02  # Minimum visual radius for rendering

    def __init__(self,
                 specs: BallSpec = BallSpec(),
                 env: Environment = Environment(),
                 init_height: float = 10,
                 color: vp.vector = vp.color.red) -> None:
        """
        Initialize a Ball object with given specifications and environment.

        Args:
            specs: Ball specifications including mass, radius, and drag coefficient.
            env: Environment specifications including gravity, air density, and coefficient of restitution.
            init_height: Initial height of the ball in the simulation.
            color: Color vector for the ball's visual representation (Default: red).
        """
        self._validate_inputs(specs, env, init_height, color)

        # Ball physical properties
        self.specs: BallSpec = specs
        self.env: Environment = env
        self.init_height: float = init_height
        self.color: vp.vector = color

        # Ball state variables
        self._position: vp.vector = vp.vector(0, init_height, 0)  # Initial position
        self._velocity: vp.vector = vp.vector(0, 0, 0)  # Initial velocity
        self._max_speed: float = 0  # Track maximum speed reached
        self._terminal_vel_reached: bool = False  # Track if terminal velocity has been reached
        self._has_hit_ground: bool = False  # Track if ball has hit the ground
        self._first_impact_time: float = 0  # Time of first ground impact
        self._has_stopped: bool = False  # Track if ball has stopped
        self._stop_time: float = 0  # Time when the ball stopped
        self._sphere: Optional[vp.sphere] = None  # vpython sphere for visual rendering

    @property
    def position(self) -> vp.vector:
        """
        Get the current position of the ball.

        Returns:
            vp.vector: The current position of the ball in the simulation.
        """
        return self._position

    @property
    def velocity(self) -> vp.vector:
        """
        Get the current velocity of the ball.

        Returns:
            vp.vector: The current velocity of the ball as a vector.
        """
        return self._velocity

    @property
    def max_speed(self) -> float:
        """
        Get the maximum speed the ball has reached.

        Returns:
            float: The maximum speed reached by the ball in the simulation.
        """
        return self._max_speed

    @property
    def terminal_vel_reached(self) -> bool:
        """
        Check if the ball has reached terminal velocity.

        Returns:
            bool: True if the ball has reached terminal velocity, False otherwise.
        """
        return self._terminal_vel_reached

    @property
    def has_hit_ground(self) -> bool:
        """
        Check if the ball has hit the ground.

        Returns:
            bool: True if the ball has hit the ground, False otherwise.
        """
        return self._has_hit_ground

    @property
    def first_impact_time(self) -> float:
        """
        Get the time of the ball's first ground impact.

        Returns:
            float: The time (in seconds) of the ball's first impact with the ground.
        """
        return self._first_impact_time

    @property
    def has_stopped(self) -> bool:
        """
        Check if the ball has stopped moving.

        Returns:
            bool: True if the ball has stopped moving, False otherwise.
        """
        return self._has_stopped

    @property
    def stop_time(self) -> float:
        """
        Get the time when the ball stopped.

        Returns:
            float: The time (in seconds) when the ball stopped moving.
        """
        return self._stop_time

    @property
    def visual_radius(self) -> float:
        """
        Calculate and return the visual radius of the ball for rendering.

        Returns:
            float: The visual radius of the ball, ensuring it's not smaller than
            a minimum value for rendering purposes.
        """
        return max(self.specs.radius, self.init_height * self._MIN_VISUAL_RADIUS)

    @property
    def sphere_pos(self) -> vp.vector:
        """
        Get the adjusted sphere position accounting for the visual radius.

        Returns:
            vp.vector: The adjusted position of the ball's visual sphere, taking 
            the visual radius into account.
        """
        return self._position + vp.vector(0, self.visual_radius, 0)

    @property
    def cross_section_area(self) -> float:
        """
        Calculate and return the cross-sectional area of the ball.

        Returns:
            float: The cross-sectional area of the ball, calculated based on its radius.
        """
        return math.pi * self.specs.radius**2

    @property
    def speed(self) -> float:
        """
        Calculate and return the current speed of the ball.

        Returns:
            float: The current speed magnitude of the ball in the simulation.
        """
        return float(vp.mag(self._velocity))

    @property
    def air_resistance(self) -> float:
        """
        Calculate and return the air resistance force on the ball.

        Returns:
            float: The air resistance force acting on the ball, calculated based 
            on its speed, cross-sectional area, air density, and drag coefficient.
        """
        return (0.5 * self.cross_section_area * self.speed**2 *
                self.env.air_density * self.specs.drag_coefficient)

    @property
    def acceleration(self) -> vp.vector:
        """
        Calculate and return the current acceleration of the ball.

        Returns:
            vp.vector: The current acceleration of the ball, which is the 
            combination of gravity and air resistance. Returns a zero vector if the 
            ball has stopped.
        """
        if self._has_stopped:
            return vp.vector(0, 0, 0)

        gravity_acc = vp.vector(0, -self.env.gravity, 0)
        drag_acc = (-self._velocity.norm() * self.air_resistance / self.specs.mass
                    if self.speed > 0 else vp.vector(0, 0, 0))
        return gravity_acc + drag_acc

    @property
    def terminal_velocity(self) -> float:
        """
        Calculate and return the theoretical terminal velocity of the ball.

        Returns:
            float: The terminal velocity of the ball, which is the speed where the 
            force of air resistance equals the force of gravity. Returns infinity 
            if there is no air resistance (e.g., in a vacuum).
        """
        if (self.env.air_density == 0 or
            self.cross_section_area == 0 or
            self.specs.drag_coefficient == 0):
            return math.inf  # No terminal velocity in a vacuum
        return math.sqrt((2 * self.specs.mass * self.env.gravity) /
                         (self.env.air_density * self.cross_section_area *
                          self.specs.drag_coefficient))
    def create_visual(self, canvas: vp.canvas) -> None:
        """
        Create a visual representation of the ball in the simulation canvas.

        Args:
            canvas: vpython canvas to draw the ball on.
        """
        self._sphere = vp.sphere(
            canvas=canvas,
            pos=self.sphere_pos,
            radius=self.visual_radius,
            color=self.color
        )

    def update(self, dt: float, current_time: float) -> None:
        """
        Update the ball's physics and position for the current time step.

        Args:
            dt: Time step duration in seconds.
            current_time: Current simulation time in seconds.
        """
        # Update velocity based on acceleration
        self._velocity += self.acceleration * dt

        # Update physical position based on velocity
        self._position += self._velocity * dt

        # Update visual position
        if self._sphere is not None:
            self._sphere.pos = self.sphere_pos

        # Track the maximum speed reached
        current_speed = abs(self._velocity.y)
        self._max_speed = max(self._max_speed, current_speed)

        # Check if terminal velocity has been reached
        if not self._terminal_vel_reached and math.isclose(
            current_speed, self.terminal_velocity, abs_tol=0.005):
            self._terminal_vel_reached = True

        # Handle ball hitting the ground
        if self._position.y <= 0:
            self._position.y = 0  # Ensure ball stays at ground level

            # Update visual position to reflect hitting the ground
            if self._sphere is not None:
                self._sphere.pos = self.sphere_pos

            if not self._has_hit_ground:
                self._has_hit_ground = True
                self._first_impact_time = current_time

            # Check if the ball has come to rest
            MIN_SPEED: Final[float] = self.env.gravity * dt
            if abs(self._velocity.y) <= MIN_SPEED:
                self._velocity.y = 0  # Stop ball movement
                if not self._has_stopped:
                    self._has_stopped = True
                    self._stop_time = current_time
            else:
                # Apply the coefficient of restitution for bouncing
                self._velocity.y = -self._velocity.y * self.env.cor

    def _validate_inputs(self, specs: BallSpec, env: Environment,
                         init_height: float, color: vp.vector) -> None:
        """
        Validate the inputs provided during ball initialization.

        Args:
            specs: Ball specifications.
            env: Environmental parameters.
            init_height: Initial height for the ball.
            color: Color vector for ball's visual representation.

        Raises:
            ValueError: If any input is invalid.
        """
        if not isinstance(specs, BallSpec):
            raise ValueError("'specs' parameter must be an instance of BallSpec")
        if not isinstance(env, Environment):
            raise ValueError("'env' parameter must be an instance of Environment")
        if not isinstance(init_height, (int, float)):
            raise ValueError("'init_height' parameter must be a numeric value")
        if init_height <= 0:
            raise ValueError("'init_height' parameter must be positive")
        if not isinstance(color, vp.vector):
            raise ValueError("'color' parameter must be a valid vp.vector object")
