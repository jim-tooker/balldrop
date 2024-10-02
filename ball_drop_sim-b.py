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

    def __init__(self, env, y, radius=DEFAULT_BALL_RADIUS, color=DEFAULT_BALL_COLOR, mass=1):
        self.env = env
        self.sphere = vp.sphere(pos=vp.vector(0, y, 0), radius=radius, color=color)
        self.velocity = vp.vector(0, 0, 0)
        self.last_velocity = self.velocity
        self.mass = mass  # kg

    @property
    def area(self) -> float:
        # Return cross-sectional area of a sphere
        return math.pi * self.sphere.radius**2

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

    def update(self, dt):
        min_speed: Final[float] = self.env.gravity * dt

        # Update velocity using acceleration
        self.velocity += self.acceleration * dt

        # Update position
        self.sphere.pos += self.velocity * dt

        # Check for ground collision
        if self.sphere.pos.y - self.sphere.radius <= 0:
            # Set ball location at ground level (account for radius of ball)
            self.sphere.pos.y = self.sphere.radius

            # Check if we've hit minimum speed
            if abs(self.velocity.y) <= min_speed:
                self.velocity.y = 0
            else:
                # Reverse velocity and multiply by cor (elasticity) factor
                self.velocity.y = -self.velocity.y * self.env.cor

class Simulation:
    def __init__(self, ball_y, ball_radius=DEFAULT_BALL_RADIUS,
                 ball_color=DEFAULT_BALL_COLOR, ball_mass=1):
        self.canvas = vp.canvas(title='Ball Drop Simulation',
                               width=800, height=600,
                               background=vp.color.white)

        self.env = Environment()
        self.ball = Ball(self.env, ball_y, ball_radius, ball_color, ball_mass)

        # Create ground
        vp.box(pos=vp.vector(0, -0.1, 0),
               size=vp.vector(20, 0.1, 20),
               color=vp.color.green)

        # Add dashed horizontal line for initial height
        dash_length = 0.5
        for x in range(-10, 11, 1):
            vp.curve(pos=[vp.vector(x, ball_y, 0),
                         vp.vector(x + dash_length, ball_y, 0)],
                    color=vp.color.blue)

        # Add label for initial height
        vp.label(pos=vp.vector(11, ball_y, 0),
                text=f'Initial height: {ball_y:.2f} m',
                align='left', box=False)

        # Add labels for time
        self.first_drop_label = vp.label(pos=vp.vector(-10, -2, 0),
                                        align='left', box=False)
        self.time_label = vp.label(pos=vp.vector(-10, -3, 0),
                                  align='left', box=False)

        # Create graphs
        graph_width: Final[int]  = 400
        graph_height: Final[int] = 300

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

    def run(self):
        dt = 1/FPS
        t = 0
        first_drop = True

        while True:
            vp.rate(FPS)

            # Update time label
            self.time_label.text = f'Time: {t:.2f} secs'

            # Plot velocity, acceleration, and position before update
            self.velocity_plot.plot(t, self.ball.velocity.y)
            self.acceleration_plot.plot(t, self.ball.acceleration.y)
            self.position_plot.plot(t, self.ball.sphere.pos.y)

            self.ball.update(dt)
            t += dt

            # If we've hit the ground
            if self.ball.sphere.pos.y - self.ball.sphere.radius <= 0:
                if first_drop:
                    self.first_drop_label.text = f'Time for first impact: {t:.2f} secs'
                    first_drop = False

                if self.ball.velocity.y == 0:
                    self.time_label.text = f'Total Time: {t:.2f} secs'
                    break

def main():
    sim = Simulation(ball_y=11, ball_mass=100, ball_radius=1)
    sim.run()
    print('Done')

if __name__ == "__main__":
    main()
