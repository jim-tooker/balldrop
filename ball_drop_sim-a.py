import vpython as vp
import math
from dataclasses import dataclass
from typing import Final, Optional

# Constants
FPS: int = 100

class BallSpecsDefaults:
    MASS: Final[float] = 1.0  # meters
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
    # Create a minimum visual size of ball.  Otherwise, for large heights, ball won't be vis1ble.
    MIN_VISUAL_RADIUS: Final[float] = 0.02

    def __init__(self,
                 specs: BallSpecs,
                 env: Environment,
                 init_height: float,
                 color: vp.vector):
        self.specs: BallSpecs = specs
        self.env: Environment = env
        self.init_height: float = init_height
        self.radius: float = specs.radius # m
        self.color: vp.vector = color
        self.mass: float = specs.mass  # kg

        self.position: vp.vector = vp.vector(0, init_height, 0)  # This will be the bottom of the ball
        self.velocity: vp.vector = vp.vector(0, 0, 0)
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
        return 0.5 * self.cross_section_area * self.speed**2 * self.env.air_density * self.specs.drag_coefficient

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
                         (self.env.air_density * self.cross_section_area * self.specs.drag_coefficient))

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
    def __init__(self, balls: list[Ball]):
        self.balls: list[Ball] = balls

        height: float = self.balls[0].init_height
        for ball in self.balls:
            if ball.init_height != height:
                raise ValueError("All balls must have the same initial height.")

        self.canvas: vp.canvas = vp.canvas(title='Ball Drop Simulation',
                                           width=900, height=600,
                                           background=vp.color.white)

        # Create the ball's visual representation
        for ball in self.balls:
            ball.create_visual(self.canvas)

        # Add the grid
        self._create_grid()

        # Add the labels
        self._create_labels()

        # Create graphs
        self._create_graphs()

    def _create_grid(self) -> None:
        # Adjust the grid range and step based on the ball's initial height
        grid_range: int = int(self.balls[0].position.y)
        step: int = int(grid_range / 10)

        for x in vp.arange(-grid_range, grid_range + step, step):
            vp.curve(pos=[vp.vector(x, 0, 0), vp.vector(x, grid_range, 0)], color=vp.color.gray(0.7))
        for y in vp.arange(0, grid_range + step, step):
            vp.curve(pos=[vp.vector(-grid_range, y, 0), vp.vector(grid_range, y, 0)], color=vp.color.gray(0.7))

        for y in vp.arange(0, grid_range + step, step):
            if y % 2 == 0:  # Add labels every 2 units
                vp.label(pos=vp.vector(-grid_range - step, y, 0), 
                        text=f'{y:.0f}', box=False)

    def _create_labels(self) -> None:
        # Adjust the label range and step based on the ball's initial height
        label_range: int = int(self.balls[0].position.y)
        step: int = int(label_range / 10)

        self.height_labels = []
        self.speed_labels = []
        self.max_speed_labels = []
        self.terminal_velocity_labels = []
        self.first_impact_labels = []
        line_num: int = 1

        # Add label for initial height
        vp.label(pos=vp.vector(-label_range, label_range+step, 0),
                 text=f'Initial Height: {self.balls[0].position.y:.1f} m',
                 align='left', box=False)
        #line_num += 1

        for ball in self.balls:
            color: vp.vector = ball.color

            # Add label for height
            height_label = vp.label(pos=vp.vector(-label_range, -line_num*step, 0),
                                    align='left', box=False, color=color)
            self.height_labels.append(height_label)
            line_num += 1

            # Add label for speed
            speed_label = vp.label(pos=vp.vector(-label_range, -line_num*step, 0),
                                   align='left', box=False, color=color)
            self.speed_labels.append(speed_label)
            line_num += 1

            # Add label for max speed
            max_speed_label = vp.label(pos=vp.vector(-label_range, -line_num*step, 0),
                                       align='left', box=False, color=color)
            self.max_speed_labels.append(max_speed_label)
            line_num += 1

            # Add label for terminal velocity
            terminal_velocity_label = vp.label(pos=vp.vector(-label_range, -line_num*step, 0),
                                               align='left', box=False, color=color)
            self.terminal_velocity_labels.append(terminal_velocity_label)
            line_num += 1

            # Add label for first impact time
            first_impact_label = vp.label(pos=vp.vector(-label_range, -line_num*step, 0),
                                          align='left', box=False, color=color)
            self.first_impact_labels.append(first_impact_label)
            line_num += 1

        # Add label for time
        self.time_label = vp.label(pos=vp.vector(-label_range, -line_num*step, 0),
                                   align='left', box=False)
        line_num += 1

    def _create_graphs(self):
        graph_width: Final[int]  = 600
        graph_height: Final[int] = 400

        self.velocity_graph = vp.graph(title="Velocity vs Time",
                                       xtitle="Time (s)", ytitle="Velocity (m/s)",
                                       width=graph_width, height=graph_height,
                                       align='left')
        self.velocity_plots = [vp.gcurve(color=ball.color) for ball in self.balls]

        self.acceleration_graph = vp.graph(title="Acceleration vs Time",
                                           xtitle="Time (s)", ytitle="Acceleration (m/s²)",
                                           width=graph_width, height=graph_height,
                                           align='left')
        self.acceleration_plots = [vp.gcurve(color=ball.color) for ball in self.balls]

        self.position_graph = vp.graph(title="Position vs Time",
                                       xtitle="Time (s)", ytitle="Height (m)",
                                       width=graph_width, height=graph_height,
                                       align='left')
        self.position_plots = [vp.gcurve(color=ball.color) for ball in self.balls]

    def _update_labels(self, t):
        # Update time label
        self.time_label.text = f'Time: {t:.2f} secs'

        # Update labels and plots for each ball
        for i, ball in enumerate(self.balls):
            # Plot velocity, acceleration, and position
            self.velocity_plots[i].plot(t, ball.velocity.y)
            self.acceleration_plots[i].plot(t, ball.acceleration.y)
            self.position_plots[i].plot(t, ball.position.y)

            # Update height label
            self.height_labels[i].text = f'Ball {i+1} Height: {ball.position.y:.2f} m'

            # Update speed label
            current_speed = abs(ball.velocity.y)
            self.speed_labels[i].text = f'Ball {i+1} Speed: {current_speed:.2f} m/s'

            # Update max speed label
            self.max_speed_labels[i].text = f'Ball {i+1} Max Speed: {ball.v_max:.2f} m/s'

            # Update terminal velocity status
            self.terminal_velocity_labels[i].text = (f'Ball {i+1} Terminal Velocity Reached? '
                                                f'{"Yes" if ball.terminal_vel_reached else "No"} '
                                                f'({ball.terminal_velocity:.2f} m/s)')

            # Update first impact time if applicable
            if ball.has_hit_ground and ball.first_impact_time is not None:
                self.first_impact_labels[i].text = \
                    f'Ball {i+1} Time for first impact: {ball.first_impact_time:.2f} secs'

    def run(self):
        dt: float = 1/FPS
        t: float = 0

        self._update_labels(t)

        while True:
            vp.rate(FPS)

            t += dt

            for ball in self.balls:
                ball.update(dt, t)

            self._update_labels(t)

            a_ball_still_moving: bool = False
            for ball in self.balls:
                if not a_ball_still_moving and ball.has_stopped is False:
                    a_ball_still_moving = True
            if not a_ball_still_moving:
                self.time_label.text = f'Total Time: {t:.2f} secs'
                break


def main():
    # Height to drop ball from
    height: float = 100

    # Create the Ball Specs
    ball1_spec: BallSpecs = BallSpecs(mass=20,
                                      radius=1,
                                      drag_coefficient=BallSpecsDefaults.SPHERE_DRAG_COEFFICIENT)
    ball2_spec: BallSpecs = BallSpecs(mass=100,
                                      radius=3,
                                      drag_coefficient=BallSpecsDefaults.SPHERE_DRAG_COEFFICIENT)

    # Create two different environments
    env1: Environment = Environment(gravity=EnvironmentDefaults.EARTH_GRAVITY,
                                    air_density=EnvironmentDefaults.EARTH_AIR_DENSITY,
                                    cor=EnvironmentDefaults.DEFAULT_COR)
    env2: Environment = Environment(gravity=EnvironmentDefaults.EARTH_GRAVITY,
                                    air_density=EnvironmentDefaults.EARTH_AIR_DENSITY,
                                    cor=0.9)

    # Create two balls with different environments
    ball1: Ball = Ball(specs=ball1_spec, env=env1, init_height=height, color=vp.color.blue)
    ball2: Ball = Ball(specs=ball2_spec, env=env2, init_height=height, color=vp.color.red)

    # Create Simulation with both balls
    sim = Simulation([ball1, ball2])
    sim.run()
    print('Done')

if __name__ == "__main__":
    main()
