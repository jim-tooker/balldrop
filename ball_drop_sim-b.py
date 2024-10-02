import vpython as vp
import math
from dataclasses import dataclass
from typing import Final

# Constants
FPS = 100
DEFAULT_BALL_RADIUS = 1.0  # meters
DEFAULT_BALL_COLOR = vp.color.red

@dataclass
class Environment:
    gravity: float = 9.8  # m/s²
    air_density: float = 1.225  # kg/m³
    cor: float = 0.8

class Ball:
    SPHERE_DRAG_COEFFICIENT: Final[float] = 0.47

    # Create a minimum visual size of ball.  Otherwise, for large heights, ball won't be vis1ble.
    MIN_VISUAL_RADIUS: Final[float] = 0.02  # 2% of scene height

    def __init__(self, env, y, radius=DEFAULT_BALL_RADIUS, color=DEFAULT_BALL_COLOR, mass=1):
        self.env = env
        self._physical_radius = radius
        visual_radius = max(radius, y * self.MIN_VISUAL_RADIUS)
        self.sphere = vp.sphere(pos=vp.vector(0, y, 0), radius=visual_radius, color=color)
        self.velocity = vp.vector(0, 0, 0)
        self.last_velocity = self.velocity
        self.mass = mass  # kg

    @property
    def physical_radius(self) -> float:
        return self._physical_radius

    @property
    def area(self) -> float:
        # Return cross-sectional area of a sphere
        return math.pi * self._physical_radius**2

    @property
    def speed(self) -> float:
        """Calculate and return the speed of the ball."""
        return float(vp.mag(self.velocity))

    @property
    def air_resistance(self) -> float:
        # Return drag force
        return 0.5 * self.area * self.speed**2 * self.env.air_density * self.SPHERE_DRAG_COEFFICIENT

    @property
    def acceleration(self) -> vp.vector:
        """Calculate acceleration based on forces."""
        gravity_acc = vp.vector(0, -self.env.gravity, 0)
        drag_acc = -self.velocity.norm() * self.air_resistance / self.mass if self.speed > 0 else vp.vector(0, 0, 0)
        return gravity_acc + drag_acc

    @property
    def terminal_velocity(self) -> float:
        """Calculate the theoretical terminal velocity."""
        return math.sqrt((2 * self.mass * self.env.gravity) /
                        (self.env.air_density * self.area * self.SPHERE_DRAG_COEFFICIENT))

    def update(self, dt):
        # Update velocity using acceleration
        self.velocity += self.acceleration * dt

        # Update position
        self.sphere.pos += self.velocity * dt

        # Check for ground collision
        if self.sphere.pos.y - self._physical_radius <= 0:
            # Set ball location at ground level (account for radius of ball)
            self.sphere.pos.y = self._physical_radius

            # Check if we've hit minimum speed
            min_speed: Final[float] = self.env.gravity * dt
            if abs(self.velocity.y) <= min_speed:
                self.velocity.y = 0
            else:
                # Reverse velocity and multiply by cor (elasticity) factor
                self.velocity.y = -self.velocity.y * self.env.cor

class Simulation:
    def __init__(self, ball_y, ball_radius=DEFAULT_BALL_RADIUS,
                 ball_color=DEFAULT_BALL_COLOR, ball_mass=1):
        self.canvas = vp.canvas(title='Ball Drop Simulation',
                               width=900, height=600,
                               background=vp.color.white)

        self.env = Environment()
        self.ball = Ball(self.env, ball_y, ball_radius, ball_color, ball_mass)
        self.v_max: float = 0
        self.terminal_vel_reached: bool = False

        # Add the grid
        self._create_grid_and_labels()

        # Create graphs
        graph_width: Final[int]  = 600
        graph_height: Final[int] = 400

        self.velocity_graph = vp.graph(title="Velocity vs Time",
                                       xtitle="Time (s)", ytitle="Velocity (m/s)",
                                       width=graph_width, height=graph_height,
                                       align='left')
        self.velocity_plot = vp.gcurve(color=vp.color.blue)

        self.acceleration_graph = vp.graph(title="Acceleration vs Time",
                                           xtitle="Time (s)", ytitle="Acceleration (m/s²)",
                                           width=graph_width, height=graph_height,
                                           align='left')
        self.acceleration_plot = vp.gcurve(color=vp.color.red)

        self.position_graph = vp.graph(title="Position vs Time",
                                       xtitle="Time (s)", ytitle="Height (m)",
                                       width=graph_width, height=graph_height,
                                       align='left')
        self.position_plot = vp.gcurve(color=vp.color.green)

    def _create_grid_and_labels(self) -> None:
        """Create a grid pattern in the simulation scene."""
        # Adjust the grid range and step based on the ball's initial height and the ground level
        grid_range: int = int(self.ball.sphere.pos.y)
        step: int = int(grid_range / 10)

        for x in vp.arange(-grid_range, grid_range + step, step):
            vp.curve(pos=[vp.vector(x, 0, 0), vp.vector(x, grid_range, 0)], color=vp.color.gray(0.7))
        for y in vp.arange(0, grid_range + step, step):
            vp.curve(pos=[vp.vector(-grid_range, y, 0), vp.vector(grid_range, y, 0)], color=vp.color.gray(0.7))

        for y in vp.arange(0, grid_range + step, step):
            if y % 2 == 0:  # Add labels every 2 units
                vp.label(pos=vp.vector(-grid_range - step, y, 0), 
                        text=f'{y:.0f}', box=False)

        line_num: int = 1

        # Add label for initial height
        vp.label(pos=vp.vector(-grid_range, -line_num*step, 0),
                text=f'Initial Height: {self.ball.sphere.pos.y:.1f} m',
                align='left', box=False)
        line_num += 1

        # Add label for current height
        self.current_height_label = vp.label(pos=vp.vector(-grid_range, -line_num*step, 0),
                                            align='left', box=False)
        line_num += 1

        # Add label for current speed
        self.current_speed_label = vp.label(pos=vp.vector(-grid_range, -line_num*step, 0),
                                            align='left', box=False)
        line_num += 1

        # Add label for max speed
        self.max_speed_label = vp.label(pos=vp.vector(-grid_range, -line_num*step, 0),
                                                align='left', box=False)
        line_num += 1

        # Add label for terminal velocity status
        self.terminal_velocity_label = vp.label(pos=vp.vector(-grid_range, -line_num*step, 0),
                                            align='left', box=False)
        line_num += 1

        # Add label for time
        self.time_label = vp.label(pos=vp.vector(-grid_range, -line_num*step, 0),
                                   align='left', box=False)
        line_num += 1

        # Add label for first drop time
        self.first_drop_label = vp.label(pos=vp.vector(-grid_range, -line_num*step, 0),
                                         align='left', box=False)
        line_num += 1

    def _update_labels(self, t):
        # Update time label
        self.time_label.text = f'Time: {t:.2f} secs'

        # Plot velocity, acceleration, and position
        self.velocity_plot.plot(t, self.ball.velocity.y)
        self.acceleration_plot.plot(t, self.ball.acceleration.y)
        self.position_plot.plot(t, self.ball.sphere.pos.y)

        # Update height label
        self.current_height_label.text = f'Height: {self.ball.sphere.pos.y:.2f} m'

        # Update speed label
        current_speed = abs(self.ball.velocity.y)
        self.current_speed_label.text = f'Speed: {current_speed:.2f} m/s'

        # Update max speed label
        self.v_max = max(self.v_max, current_speed)
        self.max_speed_label.text = f'Max Speed: {self.v_max:.2f} m/s'

        # Check if terminal velocity is reached (within 1% of theoretical)
        if not self.terminal_vel_reached and math.isclose(current_speed,
                                                            self.ball.terminal_velocity,
                                                            rel_tol=0.01):
            self.terminal_vel_reached = True

        # Update terminal velocity status
        self.terminal_velocity_label.text = (f'Terminal Velocity Reached? '
                                                f'{"Yes" if self.terminal_vel_reached else "No"} '
                                                f'(Theory: {self.ball.terminal_velocity:.2f} m/s)')

    def run(self):
        dt: float = 1/FPS
        t: float = 0
        first_drop: bool = True

        self._update_labels(t)

        while True:
            vp.rate(FPS)

            self.ball.update(dt)
            self._update_labels(t)

            # If we've hit the ground
            if self.ball.sphere.pos.y - self.ball.physical_radius <= 0:
                if first_drop:
                    self.first_drop_label.text = f'Time for first impact: {t:.2f} secs'
                    first_drop = False

                # if the ball has stopped
                if self.ball.velocity.y == 0:
                    self.time_label.text = f'Total Time: {t:.2f} secs'
                    break

            t += dt

def main():
    sim = Simulation(ball_y=10, ball_mass=20, ball_radius=1)
    sim.run()
    print('Done')

if __name__ == "__main__":
    main()
