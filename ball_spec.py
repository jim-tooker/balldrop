"""
This module defines the specifications for a ball in the ball drop simulation.

It includes the `BallSpecDefaults` class, which holds the default values for 
the ball's physical properties such as mass, radius, and drag coefficient, and 
the `BallSpec` data class that can be used to represent these properties for individual balls.
"""

from dataclasses import dataclass
from typing import Final

__author__ = 'Jim Tooker'


class BallSpecDefaults:
    """
    Default values for ball specifications.

    Attributes:
        MASS (float): The default mass of the ball in kilograms.
        RADIUS (float): The default radius of the ball in meters.
        SPHERE_DRAG_COEFFICIENT (float): The default drag coefficient for a spherical ball.
    """
    MASS: Final[float] = 1.0  # kg
    RADIUS: Final[float] = 1.0  # meters
    SPHERE_DRAG_COEFFICIENT: Final[float] = 0.47  # Drag coefficient for a sphere in air


@dataclass
class BallSpec:
    """
    Specifications for the ball, including mass, radius, and drag coefficient.

    Attributes:
        mass (float): The mass of the ball in kilograms. Defaults to `BallSpecDefaults.MASS`.
        radius (float): The radius of the ball in meters. Defaults to `BallSpecDefaults.RADIUS`.
        drag_coefficient (float): The drag coefficient of the ball. Defaults to 
                                  `BallSpecDefaults.SPHERE_DRAG_COEFFICIENT`.
    """
    mass: float = BallSpecDefaults.MASS
    radius: float = BallSpecDefaults.RADIUS
    drag_coefficient: float = BallSpecDefaults.SPHERE_DRAG_COEFFICIENT
