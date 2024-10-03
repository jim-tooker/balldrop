import vpython as vp
import math
from dataclasses import dataclass
from typing import Final, Optional

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
    MIN_VISUAL_RADIUS: Final[float] = 0.02

    def __init__(self, env, init_height, radius=DEFAULT_BALL_RADIUS, color=DEFAULT_BALL_COLOR, mass=1):
        self.env = env
        self.init_height = init_height
        self.radius = radius # m
        self.color = color
        self.mass = mass  # kg

        self.position = vp.vector(0, init_height, 0)  # This will be the bottom of the ball
        self.velocity = vp.vector(0, 0, 0)
        self.v_max: float = 0
        self.terminal_vel_reached: bool = False
        self.has_hit_ground: bool = False
        self.first_impact_time: Optional[float] = None
        self.has_stopped: bool = False
        self.sphere: vp.sphere

    @property
    def visual_radius(self) -> float:
        return max(self.radius, self.init_height * self.MIN_VISUAL_RADIUS)

    @property
    def sphere_pos(self) -> vp.vector:
        return self.position + vp.vector(0, self.visual_radius, 0)

    @property
    def cross_section_area(self) -> float:
        # Return cross-sectional area of a sphere
        return math.pi * self.radius**2

    @property
    def speed(self) -> float:
        """Calculate and return the speed of the ball."""
        return float(vp.mag(self.velocity))

    @property
    def air_resistance(self) -> float:
        # Return drag force
        return 0.5 * self.cross_section_area * self.speed**2 * self.env.air_density * self.SPHERE_DRAG_COEFFICIENT

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
                         (self.env.air_density * self.cross_section_area * self.SPHERE_DRAG_COEFFICIENT))

    def create_visual(self, canvas):        
        self.sphere = vp.sphere(canvas=canvas,
                                pos=self.sphere_pos,
                                radius=self.visual_radius,
                                color=self.color)

    def update(self, dt, current_time):
        # Update velocity using acceleration
        self.velocity += self.acceleration * dt

        # Update physical position
        self.position += self.velocity * dt

        # Update visual position
        self.sphere.pos = self.sphere_pos

        # Update max speed
        current_speed = abs(self.velocity.y)
        self.v_max = max(self.v_max, current_speed)

        # Check if terminal velocity has been reached
        if not self.terminal_vel_reached and math.isclose(current_speed,
                                                          self.terminal_velocity,
                                                          abs_tol=0.005):
            self.terminal_vel_reached = True

        # Check for ground collision
        if self.position.y <= 0:
            # Make sure physical position is at ground level
            self.position.y = 0

            # Update visual position to ground level
            self.sphere.pos = self.sphere_pos

            if not self.has_hit_ground:
                self.has_hit_ground = True
                self.first_impact_time = current_time

            # Check if we've hit minimum speed
            min_speed: Final[float] = self.env.gravity * dt
            if abs(self.velocity.y) <= min_speed:
                self.velocity.y = 0
                self.has_stopped = True
            else:
                # Reverse velocity and multiply by cor (elasticity) factor
                self.velocity.y = -self.velocity.y * self.env.cor


class Simulation:
    def __init__(self,
                 ball: Ball):
        self.ball: Ball = ball

        self.canvas: vp.canvas = vp.canvas(title='Ball Drop Simulation',
                                           width=900, height=600,
                                           background=vp.color.white)

        # Create ball's visual representation
        self.ball.create_visual(self.canvas)

        # Add the grid
        self._create_grid_and_labels()

        # Create graphs
        self._create_graphs()

    def _create_grid_and_labels(self) -> None:
        """Create a grid pattern in the simulation scene."""
        # Adjust the grid range and step based on the ball's initial height
        grid_range: int = int(self.ball.position.y)
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
                text=f'Initial Height: {self.ball.position.y:.1f} m',
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

    def _create_graphs(self):
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

    def _update_labels(self, t):
        # Update time label
        self.time_label.text = f'Time: {t:.2f} secs'

        # Plot velocity, acceleration, and position
        self.velocity_plot.plot(t, self.ball.velocity.y)
        self.acceleration_plot.plot(t, self.ball.acceleration.y)
        self.position_plot.plot(t, self.ball.position.y)

        # Update height label
        self.current_height_label.text = f'Height: {self.ball.position.y:.2f} m'

        # Update speed label
        current_speed = abs(self.ball.velocity.y)
        self.current_speed_label.text = f'Speed: {current_speed:.2f} m/s'

        # Update max speed label
        self.max_speed_label.text = f'Max Speed: {self.ball.v_max:.2f} m/s'

        # Update terminal velocity status
        self.terminal_velocity_label.text = (f'Terminal Velocity Reached? '
                                             f'{"Yes" if self.ball.terminal_vel_reached else "No"} '
                                             f'(Theory: {self.ball.terminal_velocity:.2f} m/s)')

    def run(self):
        dt: float = 1/FPS
        t: float = 0

        self._update_labels(t)

        while True:
            vp.rate(FPS)

            t += dt
            self.ball.update(dt, t)
            self._update_labels(t)

            if self.ball.has_hit_ground and self.ball.first_impact_time is not None:
                self.first_drop_label.text = f'Time for first impact: {self.ball.first_impact_time:.2f} secs'

            if self.ball.has_stopped:
                self.time_label.text = f'Total Time: {t:.2f} secs'
                break


def main():
    # Create the Environment
    env = Environment()
    #env.gravity = 9.8/4
    #env.air_density = 0
    #env.cor = 0.5

    # Create ball
    ball = Ball(env=env,
                init_height=100,
                mass=20,
                radius=1)

    # Create Simulation
    sim = Simulation(ball)
    sim.run()
    print('Done')

if __name__ == "__main__":
    main()
